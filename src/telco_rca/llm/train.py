"""QLoRA supervised fine-tuning: (system, user=evidence) -> assistant=strict JSON.

* 4-bit NF4 base weights, fp16 compute (T4 has no bf16), LoRA adapters in fp32.
* Loss only on the assistant tokens (prompt tokens are masked with -100).
* Before training, a forward pass over a few real examples checks that fp16 logits are finite
  (known Gemma-on-T4 overflow issue). If they are not and ``fp32_fallback`` is true, the model is
  reloaded with fp32 compute and the decision is written to ``train_log.json``.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from ..io import read_jsonl, write_json
from .common import finite_probe, gpu_info, load_model, load_tokenizer, peak_gpu_mb
from .data import render_full

log = logging.getLogger(__name__)

DEFAULT_TARGETS = (r"^(?!.*(vision|audio|multi_modal|mm_|embed_vision|embed_audio)).*\."
                   r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


class SFTDataset:
    def __init__(self, examples, tokenizer, cfg, max_len):
        self.items = []
        self.dropped = 0
        for ex in examples:
            prompt, full = render_full(tokenizer, ex["messages"], ex["target"], cfg)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            if f_ids[: len(p_ids)] != p_ids:
                # tokenisation merged across the boundary; fall back to concatenating id lists
                t_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
                f_ids = p_ids + t_ids
            if len(f_ids) > max_len:
                self.dropped += 1
                continue
            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            self.items.append({"input_ids": f_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def _peft(model, cfg):
    from peft import LoraConfig, get_peft_model

    lc = cfg.get("lora", {})
    pc = LoraConfig(r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.05),
                    target_modules=lc.get("target_modules", DEFAULT_TARGETS), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, pc)


def train(cfg: dict, train_file: str, output_dir: str, eval_file: str | None = None, max_steps: int | None = None,
          quant: str | None = None) -> dict:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcfg = cfg.get("train", {})
    tok = load_tokenizer(cfg)
    examples = read_jsonl(train_file)
    ds = SFTDataset(examples, tok, cfg, tcfg.get("max_seq_len", 2048))
    log.info("SFT examples: %d (dropped %d over max length)", len(ds), ds.dropped)
    quant = quant or cfg.get("train_quant", "4bit")
    on_gpu = torch.cuda.is_available()
    compute = cfg.get("compute_dtype", "float16") if on_gpu else "float32"
    dmap = None
    if on_gpu and cfg.get("train_multi_gpu") and torch.cuda.device_count() > 1:
        dmap = "balanced"  # naive model parallelism over both GPUs (memory, not speed)
    model = load_model(cfg, quant=quant if on_gpu else "none", compute_dtype=compute, device_map=dmap)
    probe_texts = [tok.decode(ds[i]["input_ids"]) for i in range(min(3, len(ds)))]
    probe = finite_probe(model, tok, probe_texts)
    fallback = False
    if not probe["finite"] and compute == "float16" and cfg.get("fp32_fallback", True):
        log.warning("fp16 forward not finite (%s); reloading with fp32 compute", probe["losses"])
        del model
        torch.cuda.empty_cache()
        compute, fallback = "float32", True
        model = load_model(cfg, quant=quant if on_gpu else "none", compute_dtype=compute, device_map=dmap)
        probe = finite_probe(model, tok, probe_texts)
    model.config.use_cache = False
    if tcfg.get("gradient_checkpointing", True) and on_gpu:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = _peft(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    losses = []

    class LossLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append({"step": state.global_step, "loss": logs["loss"], "lr": logs.get("learning_rate")})
                if len(losses) >= 5 and all(not math.isfinite(x["loss"]) or x["loss"] == 0 for x in losses[-5:]):
                    log.error("loss non-finite for 5 logging steps; stopping")
                    control.should_training_stop = True

    bs, ga = tcfg.get("batch_size", 2), tcfg.get("grad_accum", 8)
    epochs = tcfg.get("epochs", 2)
    total_steps = max_steps or math.ceil(len(ds) / (bs * ga)) * epochs
    warmup = int(tcfg.get("warmup_ratio", 0.05) * total_steps)
    args = TrainingArguments(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        max_steps=max_steps or tcfg.get("max_steps", -1),
        learning_rate=tcfg.get("lr", 2e-4),
        lr_scheduler_type=tcfg.get("scheduler", "cosine"),
        warmup_steps=warmup,
        weight_decay=0.0,
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        logging_steps=tcfg.get("logging_steps", 5),
        save_strategy="no",
        report_to=[],
        fp16=on_gpu and compute == "float16",
        bf16=False,
        optim=tcfg.get("optim", "paged_adamw_8bit") if on_gpu else "adamw_torch",
        seed=cfg.get("seed", 42),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=not on_gpu,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id), callbacks=[LossLog()])
    trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    info = {
        "model": cfg["hf_id"], "train_file": str(train_file), "n_examples": len(ds), "dropped_too_long": ds.dropped,
        "trainable_params": n_train, "compute_dtype": compute, "fp32_fallback_used": fallback,
        "quant": quant if on_gpu else "none", "fp16_probe": probe, "loss_history": losses,
        "final_loss": losses[-1]["loss"] if losses else None, "train_seconds": time.time() - t0,
        "peak_gpu_mb": peak_gpu_mb(), "gpu": gpu_info(), "train_cfg": tcfg, "lora": cfg.get("lora", {}),
    }
    write_json(out / "train_log.json", info)
    log.info("training done: %s", json.dumps({k: info[k] for k in ("n_examples", "final_loss", "train_seconds",
                                                                     "peak_gpu_mb")}))
    return info
