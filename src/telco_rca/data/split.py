"""Group-aware, label-stratified train/test split.

Windows from the same group (recording, cell, incident) never cross splits. Among all assignments
of groups to the test side (exhaustive for <=16 groups, seeded random search otherwise) we pick
the one that (1) keeps every label present in train and, where possible, in test, (2) hits the
target test fraction, and (3) minimises the Jensen-Shannon divergence between the train and test
label distributions. Ties are broken by a seeded permutation, so the split is fully determined by
(data, seed, test_fraction).
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .base import Case


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def grouped_stratified_split(
    cases: list[Case], test_fraction: float = 0.25, seed: int = 42, max_random: int = 50000
) -> dict:
    labels = sorted({c.root_cause for c in cases})
    li = {lab: i for i, lab in enumerate(labels)}
    groups = sorted({c.group_id for c in cases})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(groups)))
    groups = [groups[i] for i in order]
    gcount = {g: np.zeros(len(labels)) for g in groups}
    for c in cases:
        gcount[c.group_id][li[c.root_cause]] += 1
    total = sum(gcount.values())
    n_total = total.sum()

    def candidates():
        if len(groups) <= 16:
            for r in range(1, len(groups)):
                yield from itertools.combinations(range(len(groups)), r)
        else:
            for _ in range(max_random):
                k = int(rng.integers(1, len(groups)))
                yield tuple(sorted(rng.choice(len(groups), size=k, replace=False)))

    best, best_score = None, None
    for test_idx in candidates():
        test = sum((gcount[groups[i]] for i in test_idx), np.zeros(len(labels)))
        train = total - test
        if np.any(train == 0):
            continue  # every label must be learnable
        missing_test = int(np.sum(test == 0))
        frac = test.sum() / n_total
        score = (missing_test, round(abs(frac - test_fraction), 2), _js(train, test))
        if best_score is None or score < best_score:
            best, best_score = test_idx, score
    if best is None:
        raise ValueError("no valid grouped split: some label occurs in a single group only")
    test_groups = sorted(groups[i] for i in best)
    train_groups = sorted(set(groups) - set(test_groups))
    return make_manifest(cases, train_groups, test_groups, seed, test_fraction)


def make_manifest(cases, train_groups, test_groups, seed, test_fraction) -> dict:
    tg = set(test_groups)
    split_ids: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        split_ids["test" if c.group_id in tg else "train"].append(c.case_id)
    return {
        "method": "grouped_stratified",
        "seed": seed,
        "test_fraction_target": test_fraction,
        "train_groups": sorted(train_groups),
        "test_groups": sorted(test_groups),
        "case_ids": {k: sorted(v) for k, v in split_ids.items()},
        "stats": split_stats(cases, split_ids),
    }


def random_window_split(cases: list[Case], test_fraction: float = 0.2, seed: int = 42) -> dict:
    """Window-level random split, i.e. what TelecomTS' own run.py does (leaky; for comparison only)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    n_test = int(round(test_fraction * len(cases)))
    test = {cases[i].case_id for i in idx[:n_test]}
    split_ids = {"train": sorted(c.case_id for c in cases if c.case_id not in test), "test": sorted(test)}
    return {"method": "random_window", "seed": seed, "test_fraction_target": test_fraction,
            "case_ids": split_ids, "stats": split_stats(cases, split_ids)}


def split_stats(cases: list[Case], split_ids: dict[str, list[str]]) -> dict:
    by_id = {c.case_id: c for c in cases}
    out = {}
    for name, ids in split_ids.items():
        cnt = Counter(by_id[i].root_cause for i in ids)
        out[name] = {"n": len(ids), "groups": len({by_id[i].group_id for i in ids}),
                     "labels": dict(sorted(cnt.items()))}
    return out


def save_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1) + "\n")


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_manifest(cases: list[Case], manifest: dict) -> dict[str, list[Case]]:
    by_id = {c.case_id: c for c in cases}
    out = {}
    for name, ids in manifest["case_ids"].items():
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} case ids from manifest not found in data, e.g. {missing[:3]}")
        out[name] = [by_id[i] for i in ids]
    return out
