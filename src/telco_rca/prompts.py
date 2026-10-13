"""Prompt formatting and SFT target construction.

Three input modes share one system prompt and one output contract:

* ``evidence``: the deterministic evidence block (see :mod:`telco_rca.evidence`);
* ``raw``: the raw KPI series (rounded) plus the detector's segment, i.e. what the TelecomTS
  paper gave frontier LLMs;
* ``hybrid``: the evidence block plus the classical baseline's top-3 class probabilities.
"""
from __future__ import annotations

import numpy as np

from .data.base import Case, KPISpec
from .evidence import Evidence, fmt_num
from .schema import dump_target

SYSTEM_TEMPLATE = """You are a senior 5G radio network troubleshooting engineer. An upstream detector has \
already flagged an anomaly in one cell's KPIs. Diagnose it.

Reply with ONE JSON object and nothing else, with exactly these keys:
- "root_cause": exactly one of: {labels}
- "affected_kpis": list of the KPI names this fault affects, chosen from: {kpis}
- "explanation": 2-4 sentences: what failed, and which KPIs in this window show it, citing the evidence numbers
- "resolution": the recommended resolution, written like a troubleshooting ticket (1-2 sentences)"""


def system_prompt(labels: list[str], kpis: list[KPISpec]) -> str:
    return SYSTEM_TEMPLATE.format(
        labels="; ".join(f'"{lab}"' for lab in labels),
        kpis=", ".join(k.name for k in kpis),
    )


def kpi_glossary(kpis: list[KPISpec]) -> str:
    return "KPIs: " + "; ".join(
        f"{k.name} = {k.description}{f' [{k.unit}]' if k.unit else ''}" for k in kpis if k.description
    )


def user_prompt_evidence(evidence_text: str) -> str:
    return f"EVIDENCE (deterministic statistics of the detected anomaly):\n{evidence_text}"


def user_prompt_hybrid(evidence_text: str, top3: list[tuple[str, float]]) -> str:
    probs = ", ".join(f"{lab} {p:.2f}" for lab, p in top3)
    return (
        f"EVIDENCE (deterministic statistics of the detected anomaly):\n{evidence_text}\n"
        f"CLASSICAL MODEL (gradient boosting on the same evidence) top-3 root-cause probabilities: {probs}"
    )


def _rle(vals: list[str]) -> str:
    out, prev, n = [], None, 0
    for v in vals:
        if v == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{prev}x{n}")
            prev, n = v, 1
    if prev is not None:
        out.append(f"{prev}x{n}")
    return " ".join(out)


def raw_series_text(case: Case, sig: int = 3) -> str:
    s, e = case.segment if case.segment is not None else (0, case.length - 1)
    lines = [
        f"WINDOW: {case.length} steps, one sample every {fmt_num(case.sampling_seconds)} s.",
        f"ANOMALOUS SEGMENT (from detector): steps {s}-{e}.",
        "RAW KPI SERIES (comma separated, step 0 first):",
    ]
    for name, x in case.numeric.items():
        lines.append(f"{name}: " + ",".join(fmt_num(float(v), sig) for v in np.asarray(x)))
    for name, v in case.categorical.items():
        lines.append(f"{name} (run-length): {_rle(list(v))}")
    return "\n".join(lines)


def user_prompt_raw(raw_text: str) -> str:
    return raw_text


def observations(ev: Evidence, affected: list[str] | None, max_items: int = 4) -> str:
    """Case-specific sentence for the target explanation, built only from the evidence."""
    dev = ev.deviating()
    pick = [k for k in dev if affected is None or k.name in affected] or dev
    pick = pick[:max_items]
    s, e = ev.segment
    span = f"steps {s}-{e} ({fmt_num((e - s + 1) * ev.sampling_seconds)} s)"
    if not pick:
        return f"In this window ({span}) no KPI leaves the |z|>={fmt_num(ev.z_threshold)} baseline band."
    parts = []
    for k in pick:
        z = f"{k.z:+.1f}" if abs(k.z) < 99 else ("+99+" if k.z > 0 else "-99+")
        parts.append(f"{k.name} {k.direction} ({fmt_num(k.base_median)} -> {fmt_num(k.seg_median)}, z={z}, {k.shape})")
    return f"In this window ({span}): " + "; ".join(parts) + "."


def build_target(case: Case, ev: Evidence) -> dict:
    t = case.ticket
    assert t is not None and case.root_cause is not None, "targets need labels"
    explanation = " ".join(x for x in (t.issue, t.root_cause, observations(ev, case.affected_kpis)) if x)
    return {
        "root_cause": case.root_cause,
        "affected_kpis": list(case.affected_kpis),
        "explanation": explanation,
        "resolution": t.resolution,
    }


def target_text(case: Case, ev: Evidence) -> str:
    return dump_target(build_target(case, ev))
