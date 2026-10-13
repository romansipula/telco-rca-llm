"""Deterministic evidence extraction (no LLM).

Given an already-detected anomalous window, compare each KPI's anomalous segment against a
baseline and describe the deviation with a handful of robust statistics:

* baseline: the window's own non-anomalous steps when there are enough of them ("local"),
  otherwise a reference profile of normal behaviour for the same context ("reference");
* robust z of the segment median vs baseline median (scale = 1.4826*MAD with fallbacks);
* relative change, peak z, fraction of segment points beyond the threshold;
* shape: step / ramp / spikes / oscillating / higher-variance / shift, from a linear fit and
  exceedance pattern of the z-scored segment;
* reaction lag: steps from the segment start until the KPI first leaves the baseline band;
* categorical channels: share of each category during vs before.

The result is rendered as a compact text block for the LLM and as a flat feature vector for the
classical baseline, so both see exactly the same evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .data.base import Case, ReferenceProfile, robust_scale


@dataclass
class EvidenceConfig:
    z_threshold: float = 3.0
    min_local_baseline: int = 16  # min. normal steps inside the window to use them as baseline
    prefer_local: bool = True
    max_listed: int = 16
    decimals: int = 3  # significant digits in rendered numbers
    use_segment: bool = True  # False: treat the whole window as anomalous (no localisation)


@dataclass
class KPIEvidence:
    name: str
    unit: str
    baseline_source: str  # local | reference | none
    base_median: float
    base_scale: float
    seg_median: float
    z: float
    z_ref: float | None
    z_loc: float | None
    rel_change: float  # percent
    peak_z: float
    frac_exceed: float
    slope_z: float  # fitted change over the segment, in baseline-scale units
    r2: float
    log_var_ratio: float
    osc_rate: float
    zero_frac_delta: float
    lag: int | None  # steps after segment start when |z| first exceeds the threshold
    shape: str
    score: float

    @property
    def direction(self) -> str:
        return "up" if self.z > 0 else "down"


@dataclass
class CategoricalEvidence:
    name: str
    seg_share: dict[str, float]
    base_share: dict[str, float]
    max_shift: float  # largest absolute share change (0..1)
    shifted_value: str


@dataclass
class Evidence:
    case_id: str
    length: int
    sampling_seconds: float
    segment: tuple[int, int]
    segment_given: bool
    baseline_note: str
    kpis: list[KPIEvidence]
    categorical: list[CategoricalEvidence] = field(default_factory=list)
    z_threshold: float = 3.0

    def ranked(self) -> list[KPIEvidence]:
        return sorted(self.kpis, key=lambda k: -k.score)

    def deviating(self) -> list[KPIEvidence]:
        return [k for k in self.ranked() if k.score >= self.z_threshold]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segment"] = list(self.segment)
        return d


REF_NOTES = {
    "history": "same cell's unflagged history before the window",
    "normal_recording": "normal-operation profile of a comparable cell (no own history available)",
}


def evidence_from_dict(d: dict) -> Evidence:
    d = dict(d)
    d["kpis"] = [KPIEvidence(**k) for k in d["kpis"]]
    d["categorical"] = [CategoricalEvidence(**c) for c in d.get("categorical", [])]
    d["segment"] = tuple(d["segment"])
    return Evidence(**d)


def _robust_z(value: float, center: float, scale: float) -> float:
    return float(np.clip((value - center) / scale, -999.0, 999.0))


def _shape(zs: np.ndarray, thr: float, var_ratio: float, osc: float) -> tuple[str, float, float]:
    n = len(zs)
    if n >= 3:
        t = np.arange(n, dtype=float)
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        fit = A @ coef
        ss_res = float(np.sum((zs - fit) ** 2))
        ss_tot = float(np.sum((zs - zs.mean()) ** 2)) or 1e-12
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        slope_total = float(coef[0] * (n - 1))
    else:
        r2, slope_total = 0.0, 0.0
    med = float(np.median(zs))
    peak = float(zs[np.argmax(np.abs(zs))]) if n else 0.0
    frac = float(np.mean(np.abs(zs) > thr)) if n else 0.0
    if n >= 8 and r2 > 0.5 and abs(slope_total) > max(thr, 0.5 * abs(peak)):
        shape = "ramp up" if slope_total > 0 else "ramp down"
    elif abs(med) >= thr and frac >= 0.6:
        shape = "step up" if med > 0 else "step down"
    elif abs(peak) >= 2 * thr and frac < 0.3:
        shape = "spikes up" if peak > 0 else "spikes down"
    elif osc > 0.35 and var_ratio > 2.0:
        shape = "oscillating"
    elif var_ratio > 3.0:
        shape = "higher variance"
    elif abs(med) >= thr or frac >= 0.3:
        shape = "shift up" if med > 0 else "shift down"
    else:
        shape = "stable"
    return shape, slope_total, r2


def extract(
    case: Case,
    reference: ReferenceProfile | None = None,
    cfg: EvidenceConfig | None = None,
    units: dict[str, str] | None = None,
) -> Evidence:
    cfg = cfg or EvidenceConfig()
    units = units or {}
    T = case.length
    if cfg.use_segment and case.segment is not None:
        s, e = case.segment
        s, e = max(0, int(s)), min(T - 1, int(e))
        segment_given = True
    else:
        s, e = 0, T - 1
        segment_given = False
    in_seg = np.zeros(T, dtype=bool)
    in_seg[s : e + 1] = True
    n_local = int((~in_seg).sum())
    use_local = cfg.prefer_local and n_local >= cfg.min_local_baseline
    if use_local:
        note = f"in-window normal steps ({n_local} steps outside the segment)"
    elif reference is not None:
        note = REF_NOTES.get(reference.source, "reference profile of normal operation")
        if n_local:
            note += f" (only {n_local} normal steps inside the window)"
    else:
        note = "none available; values compared with the window itself"

    kpis = []
    for name, x in case.numeric.items():
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
        seg = x[s : e + 1]
        local = x[~in_seg]
        z_loc = z_ref = None
        loc_med = loc_scale = None
        if n_local >= max(4, cfg.min_local_baseline // 2):
            loc_med = float(np.median(local))
            loc_scale = robust_scale(local, loc_med)
            z_loc = _robust_z(float(np.median(seg)), loc_med, loc_scale)
        ref_med = ref_scale = None
        if reference is not None and name in reference.median:
            ref_med, ref_scale = reference.median[name], reference.scale[name]
            z_ref = _robust_z(float(np.median(seg)), ref_med, ref_scale)
        if use_local and loc_med is not None:
            src, bm, bs, base_vals = "local", loc_med, loc_scale, local
        elif ref_med is not None:
            src, bm, bs, base_vals = "reference", ref_med, ref_scale, None
        else:
            bm = float(np.median(x))
            src, bs, base_vals = "none", robust_scale(x, bm), x
        zs = np.clip((seg - bm) / bs, -999, 999)
        seg_med = float(np.median(seg))
        z = _robust_z(seg_med, bm, bs)
        seg_sd = float(np.std(seg))
        base_sd = float(np.std(base_vals)) if base_vals is not None and len(base_vals) > 1 else bs
        var_ratio = (seg_sd + 1e-9) / (max(base_sd, bs * 0.5) + 1e-9)
        detr = seg - np.median(seg)
        osc = float(np.mean(np.diff(np.sign(detr[detr != 0])) != 0)) if np.sum(detr != 0) > 2 else 0.0
        shape, slope_total, r2 = _shape(zs, cfg.z_threshold, var_ratio, osc)
        exceed = np.abs(zs) > cfg.z_threshold
        lag = None
        for i in range(len(zs) - 1):
            if exceed[i] and exceed[i + 1]:
                lag = i
                break
        if lag is None and len(zs) == 1 and exceed[0]:
            lag = 0
        zf_seg = float(np.mean(seg == 0))
        if base_vals is not None:
            zf_base = float(np.mean(base_vals == 0))
        else:
            zf_base = reference.zero_frac.get(name, 0.0) if reference is not None else 0.0
        rel = (seg_med - bm) / max(abs(bm), bs) * 100.0
        score = max(abs(z), 0.5 * abs(slope_total) * (r2 > 0.5))
        kpis.append(KPIEvidence(
            name=name, unit=units.get(name, ""), baseline_source=src, base_median=bm, base_scale=bs,
            seg_median=seg_med, z=z, z_ref=z_ref, z_loc=z_loc, rel_change=float(rel),
            peak_z=float(zs[np.argmax(np.abs(zs))]) if len(zs) else 0.0, frac_exceed=float(exceed.mean()),
            slope_z=slope_total, r2=r2, log_var_ratio=float(np.log(var_ratio)), osc_rate=osc,
            zero_frac_delta=zf_seg - zf_base, lag=lag, shape=shape, score=float(min(score, 999.0))))

    cats = []
    for name, vals in case.categorical.items():
        arr = np.asarray(vals, dtype=str)
        seg_v = arr[s : e + 1]
        seg_share = _share(seg_v)
        if use_local:
            base_share = _share(arr[~in_seg])
        elif reference is not None and name in reference.categorical_share:
            base_share = reference.categorical_share[name]
        else:
            base_share = _share(arr)
        keys = sorted(set(seg_share) | set(base_share))
        shifts = {k: seg_share.get(k, 0.0) - base_share.get(k, 0.0) for k in keys}
        # deterministic tie-break (with two categories the shifts are always +x / -x):
        # largest |shift|, then the category that grew, then name
        kmax = max(keys, key=lambda k: (round(abs(shifts[k]), 9), shifts[k] > 0, k)) if keys else ""
        cats.append(CategoricalEvidence(name, seg_share, base_share, abs(shifts.get(kmax, 0.0)), kmax))

    return Evidence(case.case_id, T, case.sampling_seconds, (s, e), segment_given, note, kpis, cats,
                    cfg.z_threshold)


def _share(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {}
    u, c = np.unique(arr, return_counts=True)
    return {str(a): float(b) / len(arr) for a, b in zip(u, c)}


def fmt_num(x: float, sig: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 10 ** sig:
        return f"{x:,.0f}"
    if ax >= 1e-3:
        s = f"{x:.{sig}g}"
        return s if "e" not in s else f"{x:.4f}"
    return f"{x:.2e}"


def _fmt_pct(p: float) -> str:
    if abs(p) >= 1000:
        return ">+999%" if p > 0 else "<-999%"
    return f"{p:+.0f}%"


def _fmt_z(z: float) -> str:
    if abs(z) >= 99:
        return "+99+" if z > 0 else "-99+"
    return f"{z:+.1f}"


def render(ev: Evidence, cfg: EvidenceConfig | None = None, context: dict | None = None) -> str:
    """Compact, model-readable text block."""
    cfg = cfg or EvidenceConfig()
    s, e = ev.segment
    n = e - s + 1
    dt = ev.sampling_seconds
    lines = []
    ctx = ""
    if context:
        ctx = " Context: " + ", ".join(f"{k}={v}" for k, v in context.items() if v not in (None, ""))
    lines.append(f"WINDOW: {ev.length} steps x {fmt_num(dt)} s = {fmt_num(ev.length * dt)} s.{ctx}")
    if ev.segment_given:
        lines.append(f"ANOMALOUS SEGMENT: steps {s}-{e} ({n} steps = {fmt_num(n * dt)} s, "
                     f"{round(100 * n / ev.length)}% of window; starts "
                     f"{'at window start' if s == 0 else f'at step {s}'}, "
                     f"{'still active at window end' if e == ev.length - 1 else f'ends at step {e}'}).")
    else:
        lines.append("ANOMALOUS SEGMENT: not localised; whole window treated as anomalous.")
    lines.append(f"BASELINE: {ev.baseline_note}.")
    dev = ev.deviating()[: cfg.max_listed]
    lines.append(f"DEVIATING KPIs (robust z of segment median vs baseline, |z|>={fmt_num(ev.z_threshold)}; "
                 "strongest first):")
    if not dev:
        lines.append("- none above threshold")
    for k in dev:
        unit = f" {k.unit}" if k.unit and k.unit not in ("ratio", "index") else ""
        extra = []
        if k.baseline_source == "local" and k.z_ref is not None:
            extra.append(f"vs reference z={_fmt_z(k.z_ref)}")
        if k.lag is not None:
            extra.append(f"reacts +{k.lag} steps")
        extra.append(f"{round(100 * k.frac_exceed)}% of segment beyond band")
        lines.append(
            f"- {k.name}: {k.direction} z={_fmt_z(k.z)}, {fmt_num(k.base_median)} -> {fmt_num(k.seg_median)}{unit} "
            f"({_fmt_pct(k.rel_change)}), {k.shape}; " + ", ".join(extra))
    stable = [k for k in ev.kpis if k.score < ev.z_threshold]
    if stable:
        parts = []
        for k in sorted(stable, key=lambda k: -k.score):
            tag = f"{k.name}({_fmt_z(k.z)}"
            if k.shape not in ("stable",):
                tag += f", {k.shape}"
            parts.append(tag + ")")
        lines.append("WITHIN BASELINE: " + ", ".join(parts))
    for c in ev.categorical:
        if c.max_shift >= 0.10:
            lines.append(f"CATEGORICAL {c.name}: '{c.shifted_value}' share "
                         f"{round(100 * c.base_share.get(c.shifted_value, 0))}% -> "
                         f"{round(100 * c.seg_share.get(c.shifted_value, 0))}%")
    return "\n".join(lines)


FEATURES_PER_KPI = ["z", "z_ref", "z_loc", "rel_change", "peak_z", "frac_exceed", "slope_z", "r2",
                    "log_var_ratio", "osc_rate", "zero_frac_delta", "lag"]


def feature_names(numeric_kpis: list[str], categorical_kpis: list[str]) -> list[str]:
    names = [f"{k}__{f}" for k in numeric_kpis for f in FEATURES_PER_KPI]
    names += [f"{k}__cat_shift" for k in categorical_kpis]
    names += ["seg_start_frac", "seg_len_frac", "baseline_local", "n_deviating"]
    return names


def to_features(ev: Evidence, numeric_kpis: list[str], categorical_kpis: list[str]) -> np.ndarray:
    by = {k.name: k for k in ev.kpis}
    vals: list[float] = []
    for name in numeric_kpis:
        k = by.get(name)
        for f in FEATURES_PER_KPI:
            v = getattr(k, f) if k is not None else None
            vals.append(np.nan if v is None else float(np.clip(v, -999, 999)))
    cby = {c.name: c for c in ev.categorical}
    for name in categorical_kpis:
        c = cby.get(name)
        vals.append(c.max_shift if c else np.nan)
    s, e = ev.segment
    vals += [s / ev.length, (e - s + 1) / ev.length,
             float(any(k.baseline_source == "local" for k in ev.kpis)), float(len(ev.deviating()))]
    return np.asarray(vals, dtype=float)
