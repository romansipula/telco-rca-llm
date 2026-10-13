"""Merge a LoRA adapter into its base model and export GGUF (f16 -> Q8_0, Q4_K_M) with llama.cpp.

Runs on CPU (needs ~2x the fp16 model size in RAM). ``llama_cpp`` is a llama.cpp checkout with
``convert_hf_to_gguf.py`` and a built ``build/bin/llama-quantize``.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..io import write_json

log = logging.getLogger(__name__)

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json", "generation_config.json", "processor_config.json",
                   "preprocessor_config.json", "tekken.json", "params.json", "SYSTEM_PROMPT.txt", "vocab.json",
                   "merges.txt", "added_tokens.json"]


def merge_adapter(cfg: dict, adapter: str | None, out_dir: Path) -> Path:
    import torch
    from huggingface_hub import snapshot_download

    from .common import load_model

    merged = out_dir / "merged"
    if (merged / "config.json").exists():
        return merged
    model = load_model(cfg, quant="none", compute_dtype="float16", device_map={"": "cpu"})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model = model.to(torch.float16)
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="4GB")
    # copy tokenizer / template / processor files from the original snapshot so the converter sees
    # exactly the upstream layout
    snap = Path(snapshot_download(cfg["hf_id"], allow_patterns=TOKENIZER_FILES))
    for name in TOKENIZER_FILES:
        if (snap / name).exists() and not (merged / name).exists():
            shutil.copy(snap / name, merged / name)
    del model
    return merged


def export_gguf(cfg: dict, adapter: str | None, out_dir: str, llama_cpp: str, quants=("Q8_0", "Q4_K_M"),
                keep_merged: bool = False) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lc = Path(llama_cpp)
    merged = merge_adapter(cfg, adapter, out)
    name = cfg.get("short_name", Path(cfg["hf_id"]).name)
    f16 = out / f"{name}-f16.gguf"
    info = {"model": cfg["hf_id"], "adapter": adapter, "files": {}}
    if not f16.exists():
        cmd = [sys.executable, str(lc / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16), "--outtype", "f16"]
        log.info("%s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    quant_bin = _find_bin(lc, "llama-quantize")
    for q in quants:
        dst = out / f"{name}-{q}.gguf"
        if not dst.exists():
            subprocess.run([str(quant_bin), str(f16), str(dst), q], check=True)
        info["files"][q] = {"path": dst.name, "bytes": dst.stat().st_size}
    info["files"]["f16"] = {"path": f16.name, "bytes": f16.stat().st_size}
    if not keep_merged:
        shutil.rmtree(merged, ignore_errors=True)
    if os.environ.get("TELCO_RCA_DROP_F16"):
        f16.unlink()
    info["seconds"] = time.time() - t0
    write_json(out / "export_info.json", info)
    return info


def _find_bin(lc: Path, name: str) -> Path:
    for cand in (lc / "build" / "bin" / name, lc / name, Path(shutil.which(name) or "")):
        if cand and cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found under {lc}; build llama.cpp first")
