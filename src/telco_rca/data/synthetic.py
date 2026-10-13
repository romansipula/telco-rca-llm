"""Tiny synthetic data source with known ground truth.

Used by tests and CI, and as a minimal worked example of an adapter (see ADAPTING.md). Each case
is a stationary multivariate KPI window with one injected fault whose onset, duration, shape and
affected KPIs are known exactly.
"""
from __future__ import annotations

import numpy as np

from .base import Case, DatasetAdapter, KPISpec, ReferenceProfile, Ticket, robust_profile

SPECS = [
    KPISpec("RSRP", "dBm", "reference signal received power"),
    KPISpec("SINR", "dB", "signal to interference and noise ratio"),
    KPISpec("BLER", "ratio", "block error rate"),
    KPISpec("PRB_Util", "%", "PRB utilisation"),
    KPISpec("Throughput", "Mbps", "user throughput"),
    KPISpec("Buffer", "kB", "buffer occupancy"),
]

BASE = {"RSRP": (-95.0, 1.0), "SINR": (20.0, 1.0), "BLER": (0.05, 0.005), "PRB_Util": (40.0, 3.0),
        "Throughput": (50.0, 3.0), "Buffer": (100.0, 10.0)}

# fault -> list of (kpi, kind, magnitude in baseline-sigma units)
FAULTS = {
    "Interference": [("SINR", "step", -10), ("BLER", "step", +12), ("Throughput", "step", -6)],
    "Congestion": [("PRB_Util", "ramp", +15), ("Buffer", "ramp", +20), ("Throughput", "ramp", -5)],
    "Hardware Fault": [("RSRP", "step", -15), ("SINR", "step", -6), ("BLER", "step", +8)],
}

TICKETS = {
    "Interference": Ticket(
        "Link degradation from external interference.", "SINR drops, BLER rises, throughput falls.",
        "Uplink interference source in the serving band.",
        "Locate and remove the interference source; retune affected carriers."),
    "Congestion": Ticket(
        "Cell congestion building up.", "PRB utilisation and buffer grow, throughput falls.",
        "Traffic demand exceeding cell capacity.",
        "Rebalance load to neighbour cells and adjust scheduler priorities."),
    "Hardware Fault": Ticket(
        "Radio hardware degradation.", "RSRP drops sharply with SINR and BLER degradation.",
        "Faulty antenna or RF chain.", "Inspect antenna and RF path; replace the faulty unit."),
}


def make_series(rng: np.random.Generator, T: int, fault: str, start: int, end: int) -> dict[str, np.ndarray]:
    out = {}
    for k, (mu, sd) in BASE.items():
        out[k] = mu + sd * rng.standard_normal(T)
    for kpi, kind, mag in FAULTS[fault]:
        sd = BASE[kpi][1]
        n = end - start + 1
        if kind == "step":
            out[kpi][start : end + 1] += mag * sd
        else:
            out[kpi][start : end + 1] += np.linspace(0, mag * sd, n)
    return out


class SyntheticAdapter(DatasetAdapter):
    name = "synthetic"

    def __init__(self, n_groups: int = 12, cases_per_group: int = 6, length: int = 128, seed: int = 0):
        self.n_groups, self.cases_per_group, self.length, self.seed = n_groups, cases_per_group, length, seed

    def kpis(self) -> list[KPISpec]:
        return list(SPECS)

    def label_set(self) -> list[str]:
        return list(FAULTS)

    def load_cases(self) -> list[Case]:
        rng = np.random.default_rng(self.seed)
        labels = self.label_set()
        cases = []
        for g in range(self.n_groups):
            for i in range(self.cases_per_group):
                fault = labels[(g + i) % len(labels)]
                start = int(rng.integers(32, 64))
                end = int(min(self.length - 1, start + rng.integers(32, 64)))
                cases.append(Case(
                    case_id=f"g{g:02d}-c{i:02d}", group_id=f"g{g:02d}",
                    numeric=make_series(rng, self.length, fault, start, end),
                    sampling_seconds=60.0, segment=(start, end), context={"cell": f"cell{g}"},
                    root_cause=fault, affected_kpis=[k for k, _, _ in FAULTS[fault]],
                    ticket=TICKETS[fault], source=self.name))
        return cases

    def reference_profile(self, case: Case) -> ReferenceProfile | None:
        rng = np.random.default_rng(1234)
        return robust_profile({k: mu + sd * rng.standard_normal(5000) for k, (mu, sd) in BASE.items()})
