"""Facts about the prepared TelecomTS data that the README relies on -> results/data_audit.json.

    python scripts/data_audit.py [--prepared data_cache/prepared/telecomts]
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from telco_rca.io import write_json
from telco_rca.pipeline import load_prepared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="data_cache/prepared/telecomts")
    ap.add_argument("--out", default="results/data_audit.json")
    a = ap.parse_args()
    rows, meta = load_prepared(a.prepared)
    tickets, affected = defaultdict(set), defaultdict(set)
    for r in rows:
        tickets[r["root_cause"]].add(json.dumps(r["ticket"], sort_keys=True))
        affected[r["root_cause"]].add(tuple(sorted(r["affected_kpis"])))
    # window overlap inside a recording (start times 3.2 s apart at 12.8 s windows -> 75 % overlap)
    by_g = defaultdict(list)
    for r in rows:
        by_g[r["group_id"]].append(datetime.strptime(r["context"]["start_time"], "%Y-%m-%d %H:%M:%S.%f"))
    gaps = Counter()
    for ts in by_g.values():
        ts.sort()
        for x, y in zip(ts, ts[1:]):
            gaps[round((y - x).total_seconds(), 1)] += 1
    seg_full = sum(1 for r in rows if r["segment"] == [0, r["length"] - 1])
    n_dev = [len(r["deviating"]) for r in rows]
    out = {
        "n_cases": len(rows),
        "n_recordings": len(by_g),
        "cases_per_recording": {g: len(v) for g, v in sorted(by_g.items())},
        "unique_tickets_per_class": {k: len(v) for k, v in sorted(tickets.items())},
        "unique_affected_kpi_sets_per_class": {k: len(v) for k, v in sorted(affected.items())},
        "most_common_start_time_gaps_s": dict(gaps.most_common(5)),
        "window_seconds": rows[0]["length"] * 0.1,
        "segment_covers_whole_window": seg_full,
        "baseline_source": dict(Counter(r["baseline_source"] for r in rows)),
        "deviating_kpis_per_case": {"mean": sum(n_dev) / len(n_dev), "zero": sum(1 for x in n_dev if x == 0)},
        "jamming_recordings": sorted({r["group_id"] for r in rows if r["root_cause"] == "Jamming"}),
        "labels_only_in_jamming_recordings": sorted(
            {r["root_cause"] for r in rows if r["group_id"].startswith("jammer")}),
    }
    write_json(a.out, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
