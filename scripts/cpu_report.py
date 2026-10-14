"""CPU deployment table: GPU fp16 vs GGUF quantisations on the same cases, plus cost per case.

Reads every ``predictions.jsonl`` that has a sibling ``predictions.run.json`` (written by
``telco-rca infer-cpu``) and the GPU reference predictions given with --gpu-ref. Writes
results/cpu_report.json and results/cpu_report.md.

    python scripts/cpu_report.py --gpu-ref gemma4=<gemma fp16 predictions> --gpu-ref ministral3=<...> \
        --cpu results/cpu_vm_* results/cpu_kaggle_*
(each run is matched to the reference whose key occurs in its GGUF file name)
"""
import argparse
from pathlib import Path

import numpy as np

from telco_rca.io import read_json, read_jsonl, write_json
from telco_rca.metrics import set_f1
from telco_rca.pipeline import load_prepared
from telco_rca.schema import parse_output


def score(preds, by_id, schema, ids):
    P = {p["case_id"]: p for p in preds}
    acc, valid, kf1 = [], [], []
    for i in ids:
        po = parse_output(P[i]["output"], schema)
        acc.append(po.root_cause == by_id[i]["root_cause"])
        valid.append(po.schema_ok)
        kf1.append(set_f1(po.affected_kpis, by_id[i]["affected_kpis"]))
    return {"accuracy": float(np.mean(acc)), "schema_valid": float(np.mean(valid)), "kpi_f1": float(np.mean(kf1))}


def agreement(a, b, schema, ids):
    A = {p["case_id"]: parse_output(p["output"], schema).root_cause for p in a}
    B = {p["case_id"]: parse_output(p["output"], schema).root_cause for p in b}
    return float(np.mean([A[i] == B[i] for i in ids]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-ref", action="append", required=True, help="key=predictions.jsonl")
    ap.add_argument("--cpu", nargs="+", required=True, help="dirs searched for predictions.run.json")
    ap.add_argument("--prepared", default="data_cache/prepared/telecomts")
    ap.add_argument("--out", default="results/cpu_report")
    a = ap.parse_args()
    rows, meta = load_prepared(a.prepared)
    by_id = {r["case_id"]: r for r in rows}
    schema = meta["schema"]
    refs = {k: read_jsonl(v) for k, v in (x.split("=", 1) for x in a.gpu_ref)}
    runs = []
    for d in a.cpu:
        for rj in sorted(Path(d).rglob("predictions.run.json")):
            if "raw_download" in rj.parts:
                continue
            preds = read_jsonl(rj.with_name("predictions.jsonl"))
            info = read_json(rj)
            key = next((k for k in refs if k in info["gguf"]), None)
            if key is None:
                continue
            gpu = refs[key]
            ids = sorted({p["case_id"] for p in preds} & {p["case_id"] for p in gpu})
            if not ids:
                continue
            sec = np.array([p["seconds"] for p in preds])
            runs.append({
                "run": str(rj.parent), "cache_ram_mib": info.get("cache_ram_mib", "llama.cpp default (8192)"),
                "gguf": info["gguf"], "gguf_gb": info["gguf_bytes"] / 1e9, "cpu_model": info["cpu"],
                "threads": info["threads"], "n": len(ids),
                "gpu_fp16_on_same_cases": score(gpu, by_id, schema, ids),
                "cpu": score(preds, by_id, schema, ids),
                "root_cause_agreement_with_gpu": agreement(gpu, preds, schema, ids),
                "prompt_tokens_mean": float(np.mean([p["prompt_tokens"] for p in preds])),
                "output_tokens_mean": float(np.mean([p["output_tokens"] or 0 for p in preds])),
                "seconds_mean": float(sec.mean()), "seconds_median": float(np.median(sec)),
                "seconds_p95": float(np.quantile(sec, 0.95)), "peak_rss_gb": info["peak_rss_mb"] / 1024,
                "load_seconds": info["load_seconds"],
                "cases_per_hour": 3600 / float(sec.mean()),
            })
    write_json(a.out + ".json", runs)
    head = ["run", "GGUF", "size GB", "CPU", "thr", "n", "acc GPU fp16", "acc CPU", "Δacc", "agree", "schema ok",
            "prompt tok", "out tok", "s/case (mean)", "p95 s", "peak RAM GB", "cases/h"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        g, c = r["gpu_fp16_on_same_cases"]["accuracy"], r["cpu"]["accuracy"]
        lines.append("| " + " | ".join(str(x) for x in [
            r["run"].removeprefix("results/"), r["gguf"], f"{r['gguf_gb']:.2f}", r["cpu_model"],
            r["threads"], r["n"], f"{g:.3f}", f"{c:.3f}", f"{c - g:+.3f}", f"{r['root_cause_agreement_with_gpu']:.3f}",
            f"{r['cpu']['schema_valid']:.3f}", f"{r['prompt_tokens_mean']:.0f}", f"{r['output_tokens_mean']:.0f}",
            f"{r['seconds_mean']:.1f}", f"{r['seconds_p95']:.1f}", f"{r['peak_rss_gb']:.2f}",
            f"{r['cases_per_hour']:.0f}"]) + " |")
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
