"""Command line entry point: ``telco-rca <command> ...``.

Commands (each reads YAML config and/or files written by earlier commands):

  prepare        adapter -> split -> evidence -> data_cache/.../cases.jsonl + meta.json
  baseline       gradient boosting on evidence features (+ OOF probs, + GB+template system)
  build-sft      chat-format SFT/eval files for a prompt mode (evidence | raw | hybrid)
  train          QLoRA SFT of a HF causal LM (GPU; fp16)
  infer-hf       batched generation with transformers (GPU or CPU), optional LoRA adapter
  export-gguf    merge adapter, convert to GGUF and quantise (llama.cpp)
  infer-cpu      run a GGUF with llama.cpp's llama-server, record tokens / seconds / peak RAM
  evaluate       metrics for a predictions file (+ confusion matrix plot)
  review-sheet   30-case blind review CSV for a human engineer
  report         collect metrics.json files into markdown tables
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .io import read_json, read_jsonl, write_json, write_jsonl

log = logging.getLogger("telco_rca")


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_prepare(args):
    from .pipeline import data_stats, load_prepared, prepare

    cfg = _cfg(args)
    out = prepare(cfg, resplit=args.resplit, out_dir=args.out)
    rows, _ = load_prepared(out)
    stats = data_stats(rows)
    if args.stats_out:
        write_json(args.stats_out, stats)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "groups"} if isinstance(v, dict) else v)
                      for k, v in stats.items() if k != "case_hashes"}, indent=1))
    if args.check_hash:
        from .pipeline import compare_hashes

        diff = compare_hashes(rows, read_json(args.check_hash))
        report = {"n_cases": len(rows), "n_differing": len(diff), "differing_case_ids": diff}
        if args.stats_out:
            write_json(Path(args.stats_out).with_name("hash_check.json"), report)
        print(f"prepared data vs committed: {len(diff)} of {len(rows)} cases differ")
        if len(diff) > args.max_diff_frac * len(rows):
            raise SystemExit(f"{len(diff)} cases differ from the committed prepared data: {diff[:5]}")


def cmd_baseline(args):
    import joblib

    from .baseline import class_templates, run_baseline, template_output
    from .data.split import random_window_split
    from .metrics import evaluate_predictions, plot_confusion
    from .pipeline import load_prepared

    cfg = _cfg(args)
    rows, meta = load_prepared(args.prepared or cfg["prepared_dir"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run_baseline(rows, meta, seed=cfg.get("seed", 42))
    write_jsonl(out / "gb_probs.jsonl", res["predictions"])
    if args.save_model:
        joblib.dump(res["model"], out / "gb_model.joblib")
    test = [r for r in rows if r["split"] == "test"]
    preds = {p["case_id"]: p for p in res["predictions"] if p["split"] == "test"}
    gb_rows = [{"case_id": r["case_id"], "output": json.dumps({"root_cause": preds[r["case_id"]]["pred"]})}
               for r in test]
    m = evaluate_predictions(gb_rows, test, meta, text_metrics=False)
    m["system"] = "gradient_boosting"
    m["params"] = res["params"]
    m["cv"] = res["cv"]
    write_json(out / "gb" / "metrics.json", m)
    write_jsonl(out / "gb" / "predictions.jsonl", gb_rows)
    plot_confusion(m["root_cause"]["confusion_matrix"], str(out / "gb" / "confusion.png"),
                   f"Gradient boosting on evidence features (acc {m['root_cause']['accuracy']:.3f})")
    # full-JSON non-LLM system
    tmpl = class_templates(rows)
    tp_rows = [{"case_id": r["case_id"], "output": template_output(r, preds[r["case_id"]]["pred"], tmpl)}
               for r in test]
    write_jsonl(out / "gb_template" / "predictions.jsonl", tp_rows)
    m2 = evaluate_predictions(tp_rows, test, meta, text_metrics=not args.no_text, bert=not args.no_bert)
    m2["system"] = "gb_plus_ticket_template"
    write_json(out / "gb_template" / "metrics.json", m2)
    # oracle: the SFT targets themselves (upper bound of text metrics for this target design)
    or_rows = [{"case_id": r["case_id"], "output": json.dumps(r["target"])} for r in test]
    write_jsonl(out / "oracle_targets" / "predictions.jsonl", or_rows)
    m3 = evaluate_predictions(or_rows, test, meta, text_metrics=not args.no_text, bert=not args.no_bert)
    m3["system"] = "oracle_targets"
    write_json(out / "oracle_targets" / "metrics.json", m3)
    # majority-class reference
    from collections import Counter

    maj = Counter(r["root_cause"] for r in rows if r["split"] == "train").most_common(1)[0][0]
    mj = evaluate_predictions([{"case_id": r["case_id"], "output": json.dumps({"root_cause": maj})} for r in test],
                              test, meta, text_metrics=False)
    mj["system"] = f"majority_class ({maj})"
    write_json(out / "majority" / "metrics.json", mj)
    # leaky window-level random split, as in TelecomTS run.py (for comparison only)
    if args.random_split:
        from .data.base import Case

        fake = [Case(r["case_id"], r["group_id"], {}, root_cause=r["root_cause"]) for r in rows]
        man = random_window_split(fake, 0.2, cfg.get("seed", 42))
        test_ids = set(man["case_ids"]["test"])
        rr = [dict(r, split="test" if r["case_id"] in test_ids else "train") for r in rows]
        res_r = run_baseline(rr, meta, seed=cfg.get("seed", 42), grid={k: [v] for k, v in res["params"].items()})
        tr = [r for r in rr if r["split"] == "test"]
        pr = {p["case_id"]: p for p in res_r["predictions"] if p["split"] == "test"}
        mr = evaluate_predictions([{"case_id": r["case_id"], "output": json.dumps({"root_cause": pr[r["case_id"]]["pred"]})}
                                   for r in tr], tr, meta, text_metrics=False)
        mr["system"] = "gradient_boosting_on_leaky_random_window_split"
        mr["note"] = "window-level random 80/20 split; overlapping windows of one recording on both sides"
        write_json(out / "gb_random_split" / "metrics.json", mr)
    print(json.dumps({"gb_acc": m["root_cause"]["accuracy"], "gb_macro_f1": m["root_cause"]["macro_f1"],
                      "params": res["params"]}, indent=1))


def cmd_build_sft(args):
    from .llm.data import build_examples
    from .pipeline import load_prepared

    cfg = _cfg(args)
    rows, meta = load_prepared(args.prepared or cfg["prepared_dir"])
    probs = read_jsonl(args.gb_probs) if args.gb_probs else None
    for split in ("train", "test"):
        ex = build_examples(rows, meta, args.mode, split, probs)
        write_jsonl(Path(args.out) / f"{args.mode}_{split}.jsonl", ex)
        print(split, len(ex))


def cmd_train(args):
    from .llm.train import train

    model_cfg = load_config(args.model_config, args.set)
    train(model_cfg, args.train_file, args.output_dir, eval_file=args.eval_file, max_steps=args.max_steps)


def cmd_infer_hf(args):
    from .llm.hf_infer import infer

    model_cfg = load_config(args.model_config, args.set)
    infer(model_cfg, args.examples, args.out, adapter=args.adapter, limit=args.limit,
          batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)


def cmd_export_gguf(args):
    from .llm.export import export_gguf

    model_cfg = load_config(args.model_config, args.set)
    export_gguf(model_cfg, args.adapter, args.out_dir, args.llama_cpp, quants=args.quants.split(","))


def cmd_infer_cpu(args):
    from .llm.cpu_infer import infer_cpu

    model_cfg = load_config(args.model_config, args.set)
    infer_cpu(args.gguf, args.examples, args.out, args.llama_cpp, threads=args.threads, ctx=args.ctx,
              limit=args.limit, max_new_tokens=args.max_new_tokens, server_bin=args.server_bin, model_cfg=model_cfg)


def cmd_evaluate(args):
    from .metrics import cost_summary, evaluate_predictions, plot_confusion
    from .pipeline import load_prepared

    rows, meta = load_prepared(args.prepared)
    preds = read_jsonl(args.pred)
    ids = {p["case_id"] for p in preds}
    cases = [r for r in rows if r["case_id"] in ids]
    m = evaluate_predictions(preds, cases, meta, text_metrics=not args.no_text, bert=not args.no_bert,
                             bert_device=args.device)
    m["system"] = args.name or Path(args.pred).parent.name
    m["cost"] = cost_summary(preds)
    out = Path(args.out or Path(args.pred).with_name("metrics.json"))
    write_json(out, m)
    plot_confusion(m["root_cause"]["confusion_matrix"], str(out.with_name("confusion.png")),
                   f"{m['system']} (acc {m['root_cause']['accuracy']:.3f}, n={m['n']})")
    print(json.dumps({k: m[k] for k in ("n", "json_valid_rate", "affected_kpis_set_f1")} |
                     {"acc": m["root_cause"]["accuracy"], "macro_f1": m["root_cause"]["macro_f1"]}, indent=1))


def cmd_review_sheet(args):
    from .pipeline import load_prepared
    from .review import review_sheet

    rows, meta = load_prepared(args.prepared)
    review_sheet(rows, meta, args.pred_a, args.pred_b, args.out_dir, n=args.n, seed=args.seed)


def cmd_report(args):
    from .report import report

    report(args.results, args.out)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="telco-rca", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", default="configs/telecomts.yaml")
        p.add_argument("--set", action="append", default=[], help="override config key, e.g. evidence.z_threshold=2.5")
        return p

    p = common(sub.add_parser("prepare"))
    p.add_argument("--resplit", action="store_true")
    p.add_argument("--out")
    p.add_argument("--stats-out")
    p.add_argument("--check-hash", help="committed data_stats.json to compare per-case hashes with")
    p.add_argument("--max-diff-frac", type=float, default=0.03,
                   help="fail if more cases than this differ (float/SIMD differences across CPUs)")
    p.set_defaults(fn=cmd_prepare)

    p = common(sub.add_parser("baseline"))
    p.add_argument("--prepared")
    p.add_argument("--out", required=True)
    p.add_argument("--random-split", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-bert", action="store_true")
    p.set_defaults(fn=cmd_baseline)

    p = common(sub.add_parser("build-sft"))
    p.add_argument("--prepared")
    p.add_argument("--mode", choices=["evidence", "raw", "hybrid"], required=True)
    p.add_argument("--gb-probs")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_build_sft)

    p = common(sub.add_parser("train"), config=False)
    p.add_argument("--model-config", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int)
    p.set_defaults(fn=cmd_train)

    p = common(sub.add_parser("infer-hf"), config=False)
    p.add_argument("--model-config", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.set_defaults(fn=cmd_infer_hf)

    p = common(sub.add_parser("export-gguf"), config=False)
    p.add_argument("--model-config", required=True)
    p.add_argument("--adapter")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llama-cpp", required=True, help="path to a llama.cpp checkout (with build/bin)")
    p.add_argument("--quants", default="Q8_0,Q4_K_M")
    p.set_defaults(fn=cmd_export_gguf)

    p = common(sub.add_parser("infer-cpu"), config=False)
    p.add_argument("--gguf", required=True)
    p.add_argument("--model-config", required=True, help="for the chat template / tokenizer")
    p.add_argument("--examples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--llama-cpp", default="")
    p.add_argument("--server-bin")
    p.add_argument("--threads", type=int)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.set_defaults(fn=cmd_infer_cpu)

    p = common(sub.add_parser("evaluate"), config=False)
    p.add_argument("--prepared", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--out")
    p.add_argument("--name")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--no-bert", action="store_true")
    p.add_argument("--device")
    p.set_defaults(fn=cmd_evaluate)

    p = common(sub.add_parser("review-sheet"), config=False)
    p.add_argument("--prepared", required=True)
    p.add_argument("--pred-a", required=True)
    p.add_argument("--pred-b")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_review_sheet)

    p = common(sub.add_parser("report"), config=False)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/summary.md")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
