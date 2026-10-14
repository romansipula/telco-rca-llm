"""CPU-only inference of a GGUF model with llama.cpp's ``llama-server``.

The prompt is rendered and tokenised with the model's own Hugging Face chat template/tokenizer
(tokenizer files only, no weights) and sent to the server as token ids, so the CPU run sees
byte-identical prompts to the GPU run. Per case we record prompt/output tokens, wall seconds and
server timings; peak RSS of the server process (VmHWM) is recorded once at the end.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from ..io import read_jsonl, write_json

log = logging.getLogger(__name__)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _post(url: str, payload: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _vm(pid: int, key: str) -> float | None:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024.0  # MB
    except OSError:
        return None
    return None


def find_server(llama_cpp: str = "", server_bin: str | None = None) -> str:
    cands = [server_bin] if server_bin else []
    if llama_cpp:
        cands += [str(Path(llama_cpp) / "build" / "bin" / "llama-server"), str(Path(llama_cpp) / "llama-server")]
    cands.append(shutil.which("llama-server") or "")
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("llama-server not found; pass --llama-cpp or --server-bin")


def infer_cpu(gguf: str, examples_file: str, out_file: str, llama_cpp: str = "", threads: int | None = None,
              ctx: int = 8192, limit: int | None = None, max_new_tokens: int = 512, server_bin: str | None = None,
              model_cfg: dict | None = None, cache_prompt: bool = True, cache_ram_mib: int | None = 0) -> dict:
    from .common import load_tokenizer
    from .data import render_prompt

    assert model_cfg is not None, "model_cfg is needed to render prompts with the model's chat template"
    tok = load_tokenizer(model_cfg)
    examples = read_jsonl(examples_file)
    if limit:
        examples = examples[:limit]
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.open() if line.strip()}
    todo = [e for e in examples if e["case_id"] not in done]
    threads = threads or os.cpu_count() or 4
    port = _free_port()
    binary = find_server(llama_cpp, server_bin)
    cmd = [binary, "-m", gguf, "-t", str(threads), "-c", str(ctx), "--port", str(port), "--host", "127.0.0.1",
           "-np", "1", "--no-warmup"]
    if cache_ram_mib is not None:
        # llama-server keeps up to 8 GB of past prompts in host RAM by default; cases here are
        # independent (only the system-prompt prefix is shared, which the slot's KV reuses anyway)
        cmd += ["--cache-ram", str(cache_ram_mib)]
    log.info("starting %s", " ".join(cmd))
    t_load = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out.with_suffix(".server.log"), "w"))
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(1800):
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {proc.returncode}; see {out.with_suffix('.server.log')}")
        load_s = time.time() - t_load
        peak = 0.0
        with out.open("a") as f:
            for i, e in enumerate(todo):
                prompt = render_prompt(tok, e["messages"], model_cfg)
                ids = tok(prompt, add_special_tokens=False)["input_ids"]
                t0 = time.time()
                r = _post(base + "/completion", {"prompt": ids, "n_predict": max_new_tokens, "temperature": 0.0,
                                                 "top_k": 1, "cache_prompt": cache_prompt, "seed": 0})
                dt = time.time() - t0
                tm = r.get("timings", {})
                row = {"case_id": e["case_id"], "output": r.get("content", ""), "prompt_tokens": len(ids),
                       "prompt_tokens_evaluated": tm.get("prompt_n"), "output_tokens": r.get("tokens_predicted"),
                       "seconds": dt, "prompt_ms": tm.get("prompt_ms"), "predicted_ms": tm.get("predicted_ms"),
                       "backend": "llama.cpp", "gguf": Path(gguf).name, "threads": threads}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                peak = max(peak, _vm(proc.pid, "VmHWM") or 0.0)
                if i % 10 == 0:
                    log.info("%d/%d %.1fs prompt=%d out=%s", i + 1, len(todo), dt, len(ids), r.get("tokens_predicted"))
        peak = max(peak, _vm(proc.pid, "VmHWM") or 0.0)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    info = {"gguf": Path(gguf).name, "gguf_bytes": Path(gguf).stat().st_size, "threads": threads, "ctx": ctx,
            "load_seconds": load_s, "peak_rss_mb": peak, "cache_ram_mib": cache_ram_mib, "n_new": len(todo),
            "cpu": _cpu_name(),
            "host": socket.gethostname()}
    write_json(out.with_suffix(".run.json"), info)
    return info


def _cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"
