import numpy as np

from telco_rca.data.base import Case, robust_profile
from telco_rca.evidence import EvidenceConfig, extract, feature_names, render, to_features


def _case(T=128, seg=(64, 127), seed=0):
    rng = np.random.default_rng(seed)
    x = {
        "flat": 10 + rng.standard_normal(T),
        "step_down": 20 + rng.standard_normal(T),
        "ramp_up": 5 + rng.standard_normal(T),
        "spike": 0 + 0.1 * rng.standard_normal(T),
        "late": 50 + rng.standard_normal(T),
    }
    s, e = seg
    x["step_down"][s : e + 1] -= 10
    x["ramp_up"][s : e + 1] += np.linspace(0, 40, e - s + 1)
    x["spike"][s + 5] += 30
    x["late"][s + 20 : e + 1] += 15  # reacts 20 steps after segment start
    cat = {"proto": ["TCP"] * s + ["UDP"] * (e - s + 1) + ["TCP"] * (T - e - 1)}
    return Case("c1", "g1", x, cat, sampling_seconds=0.1, segment=seg)


def _by(ev):
    return {k.name: k for k in ev.kpis}


def test_known_deviations_shapes_and_onsets():
    ev = extract(_case())
    k = _by(ev)
    dev = [d.name for d in ev.deviating()]
    assert set(dev) == {"step_down", "ramp_up", "late"}
    assert "flat" not in dev and "spike" not in dev  # one spike does not move the median
    assert k["step_down"].shape == "step down" and k["step_down"].z < -5
    assert k["ramp_up"].shape == "ramp up" and k["ramp_up"].z > 3
    assert k["spike"].shape == "spikes up"
    assert k["step_down"].lag == 0
    assert 18 <= k["late"].lag <= 21
    assert abs(k["step_down"].base_median - 20) < 1 and abs(k["step_down"].seg_median - 10) < 1
    assert all(x.baseline_source == "local" for x in ev.kpis)
    assert ev.categorical[0].shifted_value in ("UDP", "TCP") and ev.categorical[0].max_shift > 0.9


def test_ranking_strongest_first():
    ev = extract(_case())
    scores = [k.score for k in ev.ranked()]
    assert scores == sorted(scores, reverse=True)
    # medians: ramp ~+20 sigma, late step +15 sigma on 2/3 of the segment, step -10 sigma
    assert [k.name for k in ev.ranked()[:3]] == ["ramp_up", "late", "step_down"]


def test_reference_baseline_used_when_segment_covers_window():
    c = _case(seg=(0, 127))
    rng = np.random.default_rng(5)
    ref = robust_profile({"flat": 10 + rng.standard_normal(3000), "step_down": 10 + rng.standard_normal(3000),
                          "ramp_up": 5 + rng.standard_normal(3000), "spike": 0.1 * rng.standard_normal(3000),
                          "late": 50 + rng.standard_normal(3000)})
    ev = extract(c, ref)
    k = _by(ev)
    assert all(x.baseline_source == "reference" for x in ev.kpis)
    assert "step_down" not in [d.name for d in ev.deviating()]  # whole window 10 below its own start, = ref
    assert k["ramp_up"].z > 3


def test_no_segment_mode_and_render_and_features():
    c = _case()
    cfg = EvidenceConfig(use_segment=False)
    ev = extract(c, None, cfg)
    assert ev.segment == (0, 127) and not ev.segment_given
    txt = render(extract(c))
    assert "ANOMALOUS SEGMENT: steps 64-127" in txt and "step_down: down" in txt
    names = feature_names(list(c.numeric), list(c.categorical))
    f = to_features(extract(c), list(c.numeric), list(c.categorical))
    assert len(names) == len(f)
