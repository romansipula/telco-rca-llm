"""Batched greedy generation with transformers (GPU fp16 reference, or CPU)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..io import read_jsonl
from .common import load_model, load_tokenizer, peak_gpu_mb
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
    model = load_model(cfg, quant=quant, device_map=device_map,
                       compute_dtype=cfg.get("infer_dtype", cfg.get("compute_dtype", "float16"))
                       if torch.cuda.is_available() else "float32")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    stops = stop_ids(tok, cfg)
    prompts = [(e, render_prompt(tok, e["messages"], cfg)) for e in todo]
    prompts.sort(key=lambda x: len(x[1]))
    dev = next(model.parameters()).device
    with out.open("a") as f:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tok([p for _, p in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(dev) for k, v in enc.items()}
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stops,
                                     pad_token_id=tok.pad_token_id)
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
