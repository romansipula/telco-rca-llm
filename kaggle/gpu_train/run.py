"""Kaggle GPU kernel (2x T4): zero-shot + QLoRA SFT + fp16 evaluation for one or more models.

Installs this repository from GitHub, rebuilds the prepared data from the Hugging Face dataset,
checks its content hash against the committed one, then runs a small job graph on the two GPUs.
Everything it produces (predictions, train logs, LoRA adapters, job log) lands in /kaggle/working.

``SPEC`` below is replaced by scripts/kaggle_run.sh at push time.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SPEC = {
    "repo": "https://github.com/romansipula/telco-rca-llm.git",
    "ref": "main",
    "models": ["gemma4_e4b"],
    "jobs": ["train_evidence", "train_hybrid", "zs_evidence", "sft_evidence_eval", "sft_hybrid_eval", "zs_raw"],
    "train_max_steps": None,
    "infer_limit": None,
    "raw_limit": None,
    "budget_hours": 10.5,
    "max_new_tokens": 512,
}  # @@SPEC@@

W = Path("/kaggle/working")
TMP = Path("/kaggle/tmp") if Path("/kaggle").exists() else Path("/tmp/kaggle_tmp")
REPO = TMP / "repo"
LOG = W / "logs"
T0 = time.time()


def sh(cmd, log=None, check=True, env=None):
    print("+", cmd, flush=True)
    with open(log or LOG / "setup.log", "a") as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}; see {log or LOG / 'setup.log'}")
    return r.returncode


def setup():
    LOG.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    sh("nvidia-smi", LOG / "nvidia-smi.txt", check=False)
    if not REPO.exists():
        sh(f"git clone --depth 50 --branch {SPEC['ref']} {SPEC['repo']} {REPO}")
    sh(f"cd {REPO} && git log -1 --format='%H %s' > {LOG / 'commit.txt'}")
    sh(f"{sys.executable} -m pip install -q -e '{REPO}[train]' 'transformers==5.19.0' 'peft==0.21.2' "
       "'accelerate==1.15.0' 'bitsandbytes>=0.46' 'lightgbm>=4.0' rouge-score")
    # Kaggle ships torchao 0.10, which recent peft refuses to import alongside; it is not needed here
    sh(f"{sys.executable} -m pip uninstall -y -q torchao", check=False)
    sh(f"{sys.executable} -m pip freeze > {LOG / 'pip-freeze.txt'}", check=False)
    os.environ["TELCO_RCA_CACHE"] = str(TMP / "cache")
    os.environ["HF_HOME"] = str(TMP / "hf")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    prep = TMP / "prepared"
    sh(f"cd {REPO} && telco-rca prepare --out {prep} --stats-out {W / 'data_stats.json'} "
       f"--check-hash {REPO / 'results' / 'data_stats.json'}")
    sft = TMP / "sft"
    for mode in ("evidence", "raw", "hybrid"):
        sh(f"cd {REPO} && telco-rca build-sft --prepared {prep} --mode {mode} --out {sft} "
           f"--gb-probs {REPO / 'results' / 'baseline' / 'gb_probs.jsonl'}")
    return prep, sft


def model_cfg_path(name):
    return REPO / "configs" / "models" / f"{name}.yaml"


def jobs_for(model, sft):
    import yaml

    cfg = yaml.safe_load(model_cfg_path(model).read_text())
    ng = 2 if cfg.get("infer_multi_gpu") else 1
    tg = 2 if cfg.get("train_multi_gpu") else 1
    mc = model_cfg_path(model)
    out = W / model
    ms = f" --max-steps {SPEC['train_max_steps']}" if SPEC.get("train_max_steps") else ""
    lim = f" --limit {SPEC['infer_limit']}" if SPEC.get("infer_limit") else ""
    rlim = f" --limit {SPEC['raw_limit']}" if SPEC.get("raw_limit") else lim
    mnt = f" --max-new-tokens {SPEC['max_new_tokens']}"
    J = {
        "train_evidence": dict(gpus=tg, deps=[], cmd=f"telco-rca train --model-config {mc} --train-file "
                               f"{sft}/evidence_train.jsonl --output-dir {out}/sft_evidence{ms}"),
        "train_hybrid": dict(gpus=tg, deps=[], cmd=f"telco-rca train --model-config {mc} --train-file "
                             f"{sft}/hybrid_train.jsonl --output-dir {out}/sft_hybrid{ms}"),
        "zs_evidence": dict(gpus=ng, deps=[], cmd=f"telco-rca infer-hf --model-config {mc} --examples "
                            f"{sft}/evidence_test.jsonl --out {out}/zs_evidence/predictions.jsonl --batch-size 8{lim}{mnt}"),
        "zs_raw": dict(gpus=ng, deps=[], cmd=f"telco-rca infer-hf --model-config {mc} --examples "
                       f"{sft}/raw_test.jsonl --out {out}/zs_raw/predictions.jsonl --batch-size 4{rlim}{mnt}"),
        "sft_evidence_eval": dict(gpus=ng, deps=["train_evidence"], cmd=f"telco-rca infer-hf --model-config {mc} "
                                  f"--examples {sft}/evidence_test.jsonl --adapter {out}/sft_evidence/adapter "
                                  f"--out {out}/sft_evidence/predictions.jsonl --batch-size 8{lim}{mnt}"),
        "sft_hybrid_eval": dict(gpus=ng, deps=["train_hybrid"], cmd=f"telco-rca infer-hf --model-config {mc} "
                                f"--examples {sft}/hybrid_test.jsonl --adapter {out}/sft_hybrid/adapter "
                                f"--out {out}/sft_hybrid/predictions.jsonl --batch-size 8{lim}{mnt}"),
    }
    # dependencies outside this run's job list are assumed satisfied (adapters seeded from inputs)
    return [(f"{model}/{j}", dict(J[j], deps=[f"{model}/{d}" for d in J[j]["deps"] if d in SPEC["jobs"]]))
            for j in SPEC["jobs"]]


def seed_from_inputs():
    """Resume: copy earlier partial predictions (attached as a Kaggle dataset/kernel output laid out
    as <model>/<system>/predictions.jsonl) into /kaggle/working; infer-hf skips finished cases."""
    root = Path("/kaggle/input")
    if not root.exists():
        return
    systems = {"zs_evidence", "zs_raw", "sft_evidence", "sft_hybrid"}
    for a in root.rglob("adapter_config.json"):
        parts = a.parent.relative_to(root).parts
        model = next((p for p in parts if p in SPEC["models"]), None)
        if model and len(parts) >= 2 and parts[-2] in systems:
            dst = W / model / parts[-2] / "adapter"
            if not dst.exists():
                shutil.copytree(a.parent, dst)
                print("seeded adapter", dst, flush=True)
    for f in root.rglob("predictions.jsonl"):
        parts = f.relative_to(root).parts
        model = next((p for p in parts if p in SPEC["models"]), None)
        system = parts[-2] if len(parts) >= 2 else None
        if system not in systems:
            continue
        if model is None and len(SPEC["models"]) == 1:
            model = SPEC["models"][0]  # dataset uploads drop the top-level folder
        if model is None:
            continue
        dst = W / model / system / "predictions.jsonl"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
            print("seeded", dst, flush=True)


def run_jobs(jobs):
    import torch

    n_gpu = torch.cuda.device_count()
    free = list(range(n_gpu))
    pending = list(jobs)
    running = {}
    status = {}
    budget = SPEC["budget_hours"] * 3600
    while pending or running:
        for name, (proc, gpus, t0, logf) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                status[name] = {"rc": rc, "seconds": time.time() - t0, "gpus": gpus}
                free += gpus
                del running[name]
                print(f"[{time.time() - T0:7.0f}s] done {name} rc={rc} in {status[name]['seconds']:.0f}s", flush=True)
            elif time.time() - T0 > budget + 1800:
                proc.kill()
        launched = True
        while launched:
            launched = False
            for i, (name, j) in enumerate(pending):
                if any(status.get(d, {}).get("rc", 0) != 0 for d in j["deps"] if d in status):
                    status[name] = {"rc": "skipped (dependency failed)"}
                    pending.pop(i)
                    launched = True
                    break
                if any(d not in status for d in j["deps"]):
                    continue
                need = min(j["gpus"], n_gpu)
                if len(free) < need:
                    continue
                if time.time() - T0 > budget:
                    status[name] = {"rc": "skipped (budget)"}
                    pending.pop(i)
                    launched = True
                    break
                gpus, free = free[:need], free[need:]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
                logf = LOG / (name.replace("/", "__") + ".log")
                proc = subprocess.Popen(f"cd {REPO} && {j['cmd']}", shell=True, env=env,
                                        stdout=open(logf, "w"), stderr=subprocess.STDOUT)
                running[name] = (proc, gpus, time.time(), logf)
                print(f"[{time.time() - T0:7.0f}s] start {name} on GPU {gpus}", flush=True)
                pending.pop(i)
                launched = True
                break
        (W / "jobs.json").write_text(json.dumps({"spec": SPEC, "status": status, "running": list(running),
                                                 "elapsed_s": time.time() - T0}, indent=1))
        time.sleep(15)
    return status


def main():
    prep, sft = setup()
    seed_from_inputs()
    jobs = []
    for m in SPEC["models"]:
        jobs += jobs_for(m, sft)
    # two training jobs first so both GPUs are busy, then evaluations
    status = run_jobs(jobs)
    for p in W.rglob("ckpt"):
        shutil.rmtree(p, ignore_errors=True)
    shutil.copy(prep / "meta.json", W / "meta.json")
    # keep the exact test prompts for the CPU runs (not committed: kaggle_fetch skips sft/)
    (W / "sft").mkdir(exist_ok=True)
    for f in sft.glob("*_test.jsonl"):
        shutil.copy(f, W / "sft" / f.name)
    (W / "jobs.json").write_text(json.dumps({"spec": SPEC, "status": status, "elapsed_s": time.time() - T0},
                                            indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
