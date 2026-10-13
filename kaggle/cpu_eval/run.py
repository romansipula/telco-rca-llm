"""Kaggle CPU kernel: GGUF export and/or llama.cpp CPU inference + cost measurement.

Tasks (``SPEC["task"]``):
  export   merge a LoRA adapter (from an attached GPU-kernel output) into its base model on CPU,
           convert to GGUF and quantise (Q8_0, Q4_K_M); GGUFs go to /kaggle/working/gguf
  infer    run one GGUF (from an attached export-kernel output, or downloaded from the HF Hub) on
           the test prompts with llama-server; writes predictions with tokens/seconds and peak RSS

``SPEC`` below is replaced by scripts/kaggle_run.sh at push time.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SPEC = {
    "repo": "https://github.com/romansipula/telco-rca-llm.git",
    "ref": "main",
    "task": "infer",
    "model": "gemma4_e4b",
    "adapter_glob": "gemma4_e4b/sft_hybrid/adapter",
    "quants": ["Q8_0", "Q4_K_M"],
    "gguf_glob": "*Q4_K_M.gguf",
    "hf_gguf": None,  # e.g. ["google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "gemma-4-26B_q4_0-it.gguf"]
    "mode": "hybrid",
    "examples_glob": None,  # e.g. "hybrid_test.jsonl" from an attached dataset with the GPU-run prompts
    "limit": None,
    "budget_hours": 11.0,
    "max_new_tokens": 512,
    "tag": "",
}  # @@SPEC@@

W = Path("/kaggle/working")
TMP = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path("/tmp/kaggle_tmp")
REPO = TMP / "repo"
LCPP = TMP / "llama.cpp"
LOG = W / "logs"
T0 = time.time()


def sh(cmd, log=None, check=True):
    print("+", cmd, flush=True)
    with open(log or LOG / "setup.log", "a") as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def find_input(pattern):
    hits = sorted(Path("/kaggle/input").rglob(pattern)) if Path("/kaggle/input").exists() else []
    if not hits:
        raise SystemExit(f"nothing matching {pattern} under /kaggle/input")
    return hits[0]


def setup():
    LOG.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    sh("lscpu; free -g; df -h /kaggle/working /kaggle/tmp", LOG / "machine.txt", check=False)
    if not REPO.exists():
        sh(f"git clone --depth 50 --branch {SPEC['ref']} {SPEC['repo']} {REPO}")
    sh(f"cd {REPO} && git log -1 --format='%H %s' > {LOG / 'commit.txt'}")
    extra = "'peft==0.21.2' 'accelerate==1.15.0'" if SPEC["task"] == "export" else ""
    sh(f"{sys.executable} -m pip install -q -e '{REPO}' 'transformers==5.19.0' {extra} sentencepiece protobuf "
       "huggingface_hub")
    if not LCPP.exists():
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp.git {LCPP}")
    sh(f"cd {LCPP} && git log -1 --format='%H %cd' > {LOG / 'llama_cpp_commit.txt'}")
    targets = "llama-quantize" if SPEC["task"] == "export" else "llama-server"
    sh(f"cd {LCPP} && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "
       f"&& cmake --build build -j{os.cpu_count()} --target {targets}", LOG / "llama_build.log")
    if SPEC["task"] == "export":
        sh(f"{sys.executable} -m pip install -q -e {LCPP}/gguf-py mistral-common", check=False)
    os.environ["TELCO_RCA_CACHE"] = str(TMP / "cache")
    os.environ["HF_HOME"] = str(TMP / "hf")


def export():
    adapter = find_input(SPEC["adapter_glob"])
    print("adapter:", adapter)
    out = TMP / "gguf"
    env = "TELCO_RCA_DROP_F16=1 "
    sh(f"cd {REPO} && {env}telco-rca export-gguf --model-config configs/models/{SPEC['model']}.yaml "
       f"--adapter {adapter} --out-dir {out} --llama-cpp {LCPP} --quants {','.join(SPEC['quants'])}",
       LOG / "export.log")
    dst = W / "gguf"
    dst.mkdir(exist_ok=True)
    for f in out.glob("*.gguf"):
        f.rename(dst / f.name)
    (dst / "export_info.json").write_text((out / "export_info.json").read_text())
    (dst / "adapter_source.txt").write_text(str(adapter))


def infer():
    if SPEC.get("examples_glob"):
        # byte-identical prompts to the GPU runs (built and hash-checked once, then shared)
        examples = find_input(SPEC["examples_glob"])
    else:
        prep = TMP / "prepared"
        sh(f"cd {REPO} && telco-rca prepare --out {prep} --stats-out {W / 'data_stats.json'} "
           f"--check-hash {REPO / 'results' / 'data_stats.json'}")
        sft = TMP / "sft"
        sh(f"cd {REPO} && telco-rca build-sft --prepared {prep} --mode {SPEC['mode']} --out {sft} "
           f"--gb-probs {REPO / 'results' / 'baseline' / 'gb_probs.jsonl'}")
        examples = sft / f"{SPEC['mode']}_test.jsonl"
    if SPEC.get("hf_gguf"):
        from huggingface_hub import hf_hub_download

        gguf = Path(hf_hub_download(SPEC["hf_gguf"][0], SPEC["hf_gguf"][1], cache_dir=str(TMP / "hf")))
    else:
        gguf = find_input(SPEC["gguf_glob"])
    tag = SPEC.get("tag") or gguf.stem
    out = W / tag / "predictions.jsonl"
    lim = f" --limit {SPEC['limit']}" if SPEC.get("limit") else ""
    # stop cleanly before the Kaggle 12 h wall: predictions are appended line by line
    timeout = int(SPEC["budget_hours"] * 3600 - (time.time() - T0))
    cmd = (f"cd {REPO} && timeout {timeout} telco-rca infer-cpu --gguf {gguf} --model-config "
           f"configs/models/{SPEC['model']}.yaml --examples {examples} --out {out} "
           f"--llama-cpp {LCPP} --max-new-tokens {SPEC['max_new_tokens']}{lim}")
    rc = sh(cmd, LOG / "infer.log", check=False)
    (W / tag / "status.json").write_text(json.dumps({"rc": rc, "elapsed_s": time.time() - T0, "spec": SPEC}))


def main():
    setup()
    if SPEC["task"] == "export":
        export()
    else:
        infer()
    (W / "done.json").write_text(json.dumps({"elapsed_s": time.time() - T0, "spec": SPEC}))


if __name__ == "__main__":
    main()
