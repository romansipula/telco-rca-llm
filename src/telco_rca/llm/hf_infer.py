"""Batched greedy generation with transformers (GPU fp16 reference, or CPU)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..io import read_jsonl
from .common import load_model, load_tokenizer, offload_to_cpu, offload_to_cpu_target_fix, peak_gpu_mb
from .data import end_of_turn, render_prompt

log = logging.getLogger(__name__)


def stop_ids(tok, cfg) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    eot = (cfg.get("end_of_turn") or end_of_turn(tok, cfg)).strip()
    if eot:
        enc = tok(eot, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.add(int(enc[0]))
    for t in cfg.get("extra_stop_tokens", []):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return sorted(ids)


def _generate(model, tok, prompts, dev, max_new_tokens, stops):
    """Greedy batch generation; on CUDA OOM the batch is split in half and retried (results are
    re-padded to one tensor so the caller sees a single batch)."""
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    try:
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id)
        return gen, enc
    except torch.OutOfMemoryError:
        if len(prompts) == 1:
            raise
        del enc
        torch.cuda.empty_cache()
        log.warning("OOM at batch %d; splitting", len(prompts))
        h = len(prompts) // 2
        parts = [_generate(model, tok, prompts[:h], dev, max_new_tokens, stops),
                 _generate(model, tok, prompts[h:], dev, max_new_tokens, stops)]
        # left-pad prompts / right-pad generations to a common shape
        P = max(e["input_ids"].shape[1] for _, e in parts)
        G = max(g.shape[1] - e["input_ids"].shape[1] for g, e in parts)
        gens, ids, masks = [], [], []
        for g, e in parts:
            p = e["input_ids"].shape[1]
            new = g[:, p:]
            new = torch.nn.functional.pad(new, (0, G - new.shape[1]), value=tok.pad_token_id)
            pid = torch.nn.functional.pad(e["input_ids"], (P - p, 0), value=tok.pad_token_id)
            msk = torch.nn.functional.pad(e["attention_mask"], (P - p, 0), value=0)
            gens.append(torch.cat([pid, new], 1))
            ids.append(pid)
            masks.append(msk)
        return torch.cat(gens), {"input_ids": torch.cat(ids), "attention_mask": torch.cat(masks)}


def infer(cfg: dict, examples_file: str, out_file: str, adapter: str | None = None, limit: int | None = None,
          batch_size: int = 8, max_new_tokens: int = 512, quant: str | None = None) -> None:
    import torch

    tok = load_tokenizer(cfg)
    tok.padding_side = "left"
    examples = read_jsonl(examples_file)
    if limit:
        examples = examples[:limit]
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.open() if line.strip()}
    todo = [e for e in examples if e["case_id"] not in done]
    if not todo:
        log.info("all %d predictions already present in %s", len(examples), out)
        return
    quant = quant or cfg.get("infer_quant", "none")
    device_map = cfg.get("infer_device_map")
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 and \
            cfg.get("infer_multi_gpu", False) else None
    dtype = cfg.get("infer_dtype", cfg.get("compute_dtype", "float16")) if torch.cuda.is_available() else "float32"
    if cfg.get("cpu_modules") and torch.cuda.is_available() and quant == "none":
        # load in host RAM, keep the big lookup tables there, move the rest to the GPU
        # (the full fp16 Gemma 4 E4B does not fit a T4 while loading)
        model = load_model(cfg, quant="none", device_map={"": "cpu"}, compute_dtype=dtype)
        offload_to_cpu(model, cfg["cpu_modules"])
        model.to("cuda:0")
        offload_to_cpu_target_fix(model, cfg["cpu_modules"])
    else:
        model = load_model(cfg, quant=quant, device_map=device_map, compute_dtype=dtype)
        if cfg.get("cpu_modules") and torch.cuda.is_available():
            offload_to_cpu(model, cfg["cpu_modules"])
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    stops = stop_ids(tok, cfg)
    prompts = [(e, render_prompt(tok, e["messages"], cfg)) for e in todo]
    prompts.sort(key=lambda x: len(x[1]))
    emb = model.get_input_embeddings()
    dev = emb.weight.device if hasattr(emb, "weight") else next(model.parameters()).device
    with out.open("a") as f:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            t0 = time.time()
            gen, enc = _generate(model, tok, [p for _, p in chunk], dev, max_new_tokens, stops)
            dt = time.time() - t0
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, (e, _) in enumerate(chunk):
                ids = new[j].tolist()
                n_out = 0
                for t in ids:
                    n_out += 1
                    if t in stops:
                        break
                text = tok.decode(ids[:n_out], skip_special_tokens=True)
                row = {"case_id": e["case_id"], "output": text,
                       "prompt_tokens": int(enc["attention_mask"][j].sum()), "output_tokens": n_out,
                       "seconds": dt / len(chunk), "batch_seconds": dt, "batch_size": len(chunk),
                       "backend": "transformers", "adapter": adapter, "quant": quant}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            log.info("%d/%d done (%.1fs/batch, peak %.0f MB)", min(i + batch_size, len(prompts)), len(prompts), dt,
                     peak_gpu_mb() or 0)
