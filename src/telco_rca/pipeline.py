"""Data preparation: adapter -> split -> evidence -> prompts/targets -> ``cases.jsonl``.

``cases.jsonl`` is the only interface between data and models: training, HF inference, llama.cpp
inference, the classical baseline and evaluation all read it. One line per case.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .data import get_adapter
from .data.split import apply_manifest, grouped_stratified_split, load_manifest, save_manifest
from .evidence import EvidenceConfig, extract, feature_names, render, to_features
from .io import read_json, read_jsonl, write_json, write_jsonl
from .prompts import build_target, kpi_glossary, raw_series_text, system_prompt
from .schema import build_schema

log = logging.getLogger(__name__)


def evidence_config(cfg: dict) -> EvidenceConfig:
    return EvidenceConfig(**cfg.get("evidence", {}))


def prepare(cfg: dict, resplit: bool = False, out_dir: str | None = None) -> Path:
    ds = cfg["dataset"]
    adapter = get_adapter(ds["adapter"], **ds.get("kwargs", {}))
    cases = adapter.load_cases()
    log.info("loaded %d cases from %s", len(cases), adapter.name)

    sp = cfg["split"]
    man_path = Path(sp["manifest"])
    if resplit or not man_path.exists():
        manifest = grouped_stratified_split(cases, sp.get("test_fraction", 0.25), sp.get("seed", 42))
        save_manifest(manifest, man_path)
        log.info("wrote split manifest %s", man_path)
    manifest = load_manifest(man_path)
    splits = apply_manifest(cases, manifest)

    ecfg = evidence_config(cfg)
    specs = adapter.kpis()
    units = {k.name: k.unit for k in specs}
    num, cat = adapter.numeric_kpis(), adapter.categorical_kpis()
    rows = []
    for split_name, items in splits.items():
        for c in items:
            ref = adapter.reference_profile(c)
            ev = extract(c, ref, ecfg, units)
            row = {
                "case_id": c.case_id,
                "group_id": c.group_id,
                "split": split_name,
                "root_cause": c.root_cause,
                "affected_kpis": c.affected_kpis,
                "ticket": asdict(c.ticket) if c.ticket else None,
                "reference_diagnosis": c.ticket.diagnosis_text() if c.ticket else "",
                "reference_resolution": c.ticket.resolution if c.ticket else "",
                "context": c.context,
                "segment": list(c.segment) if c.segment else None,
                "length": c.length,
                "baseline_source": ev.kpis[0].baseline_source if ev.kpis else "none",
                "evidence_text": render(ev, ecfg),
                "raw_text": raw_series_text(c),
                "deviating": [k.name for k in ev.deviating()],
                "evidence": ev.to_dict(),
                "features": [None if f != f else f for f in to_features(ev, num, cat).tolist()],
                "target": build_target(c, ev) if c.ticket and c.root_cause else None,
            }
            rows.append(row)
    rows.sort(key=lambda r: (r["split"], r["case_id"]))
    out = Path(out_dir or cfg["prepared_dir"])
    write_jsonl(out / "cases.jsonl", rows)
    labels = adapter.label_set()
    meta = {
        "adapter": adapter.name,
        "labels": labels,
        "kpis": [asdict(k) for k in specs],
        "numeric_kpis": num,
        "categorical_kpis": cat,
        "feature_names": feature_names(num, cat),
        "system_prompt": system_prompt(labels, specs),
        "kpi_glossary": kpi_glossary(specs),
        "schema": build_schema(labels, [k.name for k in specs]),
        "evidence_config": asdict(ecfg),
        "split_manifest": str(man_path),
    }
    write_json(out / "meta.json", meta)
    log.info("wrote %d prepared cases to %s", len(rows), out)
    return out


def load_prepared(prepared_dir: str | Path) -> tuple[list[dict], dict]:
    p = Path(prepared_dir)
    return read_jsonl(p / "cases.jsonl"), read_json(p / "meta.json")


def prepared_hash(rows: list[dict]) -> str:
    """Content hash of what the models see (split, evidence, raw text, target); used to check that a
    Kaggle kernel rebuilt exactly the same prepared data as the committed results."""
    import hashlib
    import json

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["case_id"]):
        h.update(json.dumps([r["case_id"], r["split"], r["evidence_text"], r["raw_text"], r["target"]],
                            sort_keys=True).encode())
    return h.hexdigest()


def data_stats(rows: list[dict]) -> dict:
    out: dict = {"prepared_sha256": prepared_hash(rows)}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        out[split] = {
            "n_cases": len(rs),
            "groups": sorted({r["group_id"] for r in rs}),
            "class_counts": dict(sorted(Counter(r["root_cause"] for r in rs).items())),
            "baseline_source": dict(Counter(r["baseline_source"] for r in rs)),
            "segment_covers_full_window": sum(
                1 for r in rs if r["segment"] and r["segment"][0] == 0 and r["segment"][1] == r["length"] - 1),
        }
    return out
