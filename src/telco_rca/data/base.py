"""Dataset-agnostic data model.

Every data source is converted into a list of :class:`Case` objects by an adapter. Everything
downstream (split, evidence extraction, prompts, training, evaluation) only sees ``Case`` and
``KPISpec``; nothing downstream knows which dataset a case came from.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KPISpec:
    """Static description of one KPI channel."""

    name: str
    unit: str = ""
    description: str = ""
    # "numeric" or "categorical"
    kind: str = "numeric"


@dataclass
class Ticket:
    """Reference troubleshooting ticket (the dataset's ground-truth text)."""

    issue: str = ""
    symptoms: str = ""
    root_cause: str = ""
    resolution: str = ""

    def diagnosis_text(self) -> str:
        """Reference text the generated ``explanation`` is compared against."""
        return " ".join(s for s in (self.issue, self.symptoms, self.root_cause) if s).strip()


@dataclass
class Case:
    """One already-detected anomaly: a multivariate KPI window plus labels and metadata.

    ``numeric`` maps KPI name -> float array of length T. ``categorical`` maps KPI name -> list of
    str of length T. ``segment`` is the anomalous interval [start, end] (inclusive step indices)
    reported by the upstream detector, or None if the detector only flags the whole window.
    """

    case_id: str
    group_id: str  # recording / cell / incident id; never crosses splits
    numeric: dict[str, np.ndarray]
    categorical: dict[str, list[str]] = field(default_factory=dict)
    sampling_seconds: float = 1.0
    segment: tuple[int, int] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    # labels (may be missing for unlabeled inference)
    root_cause: str | None = None
    affected_kpis: list[str] = field(default_factory=list)
    ticket: Ticket | None = None
    source: str = ""

    @property
    def length(self) -> int:
        return len(next(iter(self.numeric.values())))


@dataclass
class ReferenceProfile:
    """Robust per-KPI statistics of normal behaviour for a given context (e.g. one cell).

    Used as the baseline when the anomalous window has too little normal data of its own.
    """

    median: dict[str, float]
    scale: dict[str, float]  # robust scale (1.4826*MAD, with fallbacks)
    q05: dict[str, float]
    q95: dict[str, float]
    zero_frac: dict[str, float]
    n: int = 0
    categorical_share: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = "reference"  # free-text tag shown in the evidence block


class DatasetAdapter(abc.ABC):
    """Interface each data source implements."""

    name: str = "base"

    @abc.abstractmethod
    def kpis(self) -> list[KPISpec]:
        """All KPI channels this source provides (numeric and categorical)."""

    @abc.abstractmethod
    def label_set(self) -> list[str]:
        """Closed set of root-cause labels."""

    @abc.abstractmethod
    def load_cases(self) -> list[Case]:
        """Load all anomalous cases (already-detected anomalies only)."""

    def reference_profile(self, case: Case) -> ReferenceProfile | None:
        """Normal-behaviour profile for the case's context, or None."""
        return None

    def numeric_kpis(self) -> list[str]:
        return [k.name for k in self.kpis() if k.kind == "numeric"]

    def categorical_kpis(self) -> list[str]:
        return [k.name for k in self.kpis() if k.kind == "categorical"]


def robust_profile(
    numeric: dict[str, np.ndarray], categorical: dict[str, list[str]] | None = None
) -> ReferenceProfile:
    """Build a :class:`ReferenceProfile` from long stretches of normal data."""
    med, scale, q05, q95, zf = {}, {}, {}, {}, {}
    n = 0
    for k, x in numeric.items():
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = max(n, len(x))
        if len(x) == 0:
            med[k], scale[k], q05[k], q95[k], zf[k] = 0.0, 1.0, 0.0, 0.0, 0.0
            continue
        m = float(np.median(x))
        med[k] = m
        scale[k] = robust_scale(x, m)
        q05[k] = float(np.quantile(x, 0.05))
        q95[k] = float(np.quantile(x, 0.95))
        zf[k] = float(np.mean(x == 0))
    cat_share: dict[str, dict[str, float]] = {}
    for k, vals in (categorical or {}).items():
        vals = list(vals)
        if not vals:
            continue
        uniq, cnt = np.unique(np.asarray(vals, dtype=str), return_counts=True)
        cat_share[k] = {str(u): float(c) / len(vals) for u, c in zip(uniq, cnt)}
    return ReferenceProfile(med, scale, q05, q95, zf, n, cat_share)


def robust_scale(x: np.ndarray, center: float | None = None) -> float:
    """1.4826*MAD with fallbacks (IQR, std, relative floor) so it is never zero."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 1.0
    c = float(np.median(x)) if center is None else center
    mad = 1.4826 * float(np.median(np.abs(x - c)))
    if mad <= 0:
        q75, q25 = np.quantile(x, [0.75, 0.25])
        mad = float(q75 - q25) / 1.349
    if mad <= 0:
        mad = float(np.std(x))
    floor = max(1e-6, 0.01 * abs(c))
    return max(mad, floor)
