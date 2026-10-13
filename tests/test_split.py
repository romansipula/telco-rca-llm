import json
from collections import Counter
from pathlib import Path

import pytest

from telco_rca.data.split import apply_manifest, grouped_stratified_split, random_window_split
from telco_rca.data.synthetic import SyntheticAdapter

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "splits" / "telecomts_grouped_seed42.json"


def _groups_of(manifest, split):
    # TelecomTS case ids are "<group>@<start_time>"
    return {cid.split("@")[0] for cid in manifest["case_ids"][split]}


def test_synthetic_grouped_split_has_no_group_leakage():
    cases = SyntheticAdapter(n_groups=10, cases_per_group=6).load_cases()
    man = grouped_stratified_split(cases, test_fraction=0.3, seed=1)
    parts = apply_manifest(cases, man)
    g_tr = {c.group_id for c in parts["train"]}
    g_te = {c.group_id for c in parts["test"]}
    assert g_tr and g_te and not (g_tr & g_te)
    assert len(parts["train"]) + len(parts["test"]) == len(cases)
    # every label learnable and present in test
    assert set(Counter(c.root_cause for c in parts["train"])) == set(Counter(c.root_cause for c in cases))
    assert set(Counter(c.root_cause for c in parts["test"])) == set(Counter(c.root_cause for c in cases))


def test_split_is_deterministic():
    cases = SyntheticAdapter(n_groups=9, cases_per_group=5).load_cases()
    a = grouped_stratified_split(cases, 0.25, seed=42)
    b = grouped_stratified_split(cases, 0.25, seed=42)
    assert a["test_groups"] == b["test_groups"]


def test_random_window_split_does_leak_groups():
    # documents why the official window-level split is not used
    cases = SyntheticAdapter(n_groups=6, cases_per_group=10).load_cases()
    man = random_window_split(cases, 0.2, seed=0)
    parts = apply_manifest(cases, man)
    assert {c.group_id for c in parts["train"]} & {c.group_id for c in parts["test"]}


@pytest.mark.skipif(not MANIFEST.exists(), reason="TelecomTS manifest not generated")
def test_committed_telecomts_manifest_has_no_recording_leakage():
    man = json.loads(MANIFEST.read_text())
    tr, te = _groups_of(man, "train"), _groups_of(man, "test")
    assert not (tr & te)
    assert tr == set(man["train_groups"]) and te == set(man["test_groups"])
    assert not set(man["case_ids"]["train"]) & set(man["case_ids"]["test"])
    # all 11 root causes present on both sides
    assert len(man["stats"]["train"]["labels"]) == 11
    assert len(man["stats"]["test"]["labels"]) == 11
