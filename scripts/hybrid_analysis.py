"""How much does the hybrid LLM follow the gradient-boosting probabilities in its prompt?

    python scripts/hybrid_analysis.py results/<run>/<model>/sft_hybrid/predictions.jsonl [...]
Writes results/hybrid_analysis.json.
"""
import json
import sys

from telco_rca.io import read_jsonl, write_json
from telco_rca.pipeline import load_prepared
from telco_rca.schema import parse_output


def main():
    rows, meta = load_prepared("data_cache/prepared/telecomts")
    by = {r["case_id"]: r for r in rows}
    gb = {p["case_id"]: p for p in read_jsonl("results/baseline/gb_probs.jsonl")}
    out = {}
    for f in sys.argv[1:]:
        P = read_jsonl(f)
        s = {"n": len(P), "agree_with_gb_top1": 0, "within_gb_top3": 0, "llm_right_gb_wrong": 0,
             "llm_wrong_gb_right": 0, "both_right": 0, "both_wrong": 0}
        for p in P:
            rc = parse_output(p["output"], meta["schema"]).root_cause
            g, t = gb[p["case_id"]], by[p["case_id"]]["root_cause"]
            s["agree_with_gb_top1"] += rc == g["pred"]
            s["within_gb_top3"] += rc in [x[0] for x in g["top3"]]
            s["llm_right_gb_wrong"] += rc == t and g["pred"] != t
            s["llm_wrong_gb_right"] += rc != t and g["pred"] == t
            s["both_right"] += rc == t and g["pred"] == t
            s["both_wrong"] += rc != t and g["pred"] != t
        s["agree_with_gb_top1"] /= len(P)
        s["within_gb_top3"] /= len(P)
        out[f] = s
    write_json("results/hybrid_analysis.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
