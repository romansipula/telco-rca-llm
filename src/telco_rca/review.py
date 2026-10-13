"""Blind review sheet for a human telecom engineer.

``review_sheet.csv`` shows the evidence block and two anonymised outputs per case (system order
randomised per row); ``review_key.csv`` (keep away from the reviewer) maps rows back to case,
ground truth and system.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from .io import read_jsonl
from .schema import parse_output

RATING_COLS = ["root_cause_plausible_Y_N", "explanation_faithful_1to5", "resolution_actionable_1to5"]


def _fields(po):
    return [po.root_cause or (po.obj or {}).get("root_cause", "") or "<invalid JSON>",
            ", ".join(po.affected_kpis), po.explanation, po.resolution]


def review_sheet(rows: list[dict], meta: dict, pred_a: str, pred_b: str | None, out_dir: str, n: int = 30,
                 seed: int = 7) -> None:
    rng = random.Random(seed)
    schema = meta["schema"]
    A = {p["case_id"]: p for p in read_jsonl(pred_a)}
    B = {p["case_id"]: p for p in read_jsonl(pred_b)} if pred_b else {}
    test = [r for r in rows if r["split"] == "test" and r["case_id"] in A and (not B or r["case_id"] in B)]
    by_cls = defaultdict(list)
    for r in sorted(test, key=lambda r: r["case_id"]):
        by_cls[r["root_cause"]].append(r)
    for v in by_cls.values():
        rng.shuffle(v)
    picked, classes = [], sorted(by_cls)
    while len(picked) < n and any(by_cls.values()):
        for c in classes:
            if by_cls[c] and len(picked) < n:
                picked.append(by_cls[c].pop())
    rng.shuffle(picked)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    systems = [("A", Path(pred_a).parent.name, A)] + ([("B", Path(pred_b).parent.name, B)] if B else [])
    head = ["item", "evidence"]
    for i in range(1, len(systems) + 1):
        head += [f"out{i}_root_cause", f"out{i}_affected_kpis", f"out{i}_explanation", f"out{i}_resolution"]
        head += [f"out{i}_{c}" for c in RATING_COLS]
    if len(systems) == 2:
        head += ["preferred_1_2_tie"]
    head += ["comments"]
    key_head = ["item", "case_id", "true_root_cause", "reference_resolution"] + \
        [f"out{i}_system" for i in range(1, len(systems) + 1)]
    with open(out / "review_sheet.csv", "w", newline="") as f, open(out / "review_key.csv", "w", newline="") as fk:
        w, wk = csv.writer(f), csv.writer(fk)
        w.writerow(head)
        wk.writerow(key_head)
        for i, r in enumerate(picked, 1):
            order = systems[:]
            rng.shuffle(order)
            line = [i, r["evidence_text"]]
            for _, _, preds in order:
                line += _fields(parse_output(preds[r["case_id"]]["output"], schema)) + [""] * len(RATING_COLS)
            if len(systems) == 2:
                line.append("")
            line.append("")
            w.writerow(line)
            wk.writerow([i, r["case_id"], r["root_cause"], r["reference_resolution"]] + [s[1] for s in order])
