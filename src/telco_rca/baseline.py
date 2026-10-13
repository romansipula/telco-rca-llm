"""Classical baseline: gradient boosting (LightGBM) on the evidence feature vector (root cause only).

Also produces
* out-of-fold (group k-fold) class probabilities on the training split, used as the hybrid
  model's extra evidence during SFT, so the LLM never sees over-confident in-sample probabilities;
* a "GB + ticket template" system: GB's root cause, the class's affected-KPI list and resolution
  looked up from the training tickets, and an explanation assembled from the evidence. It is the
  non-LLM reference for the whole JSON output.
"""
from __future__ import annotations

import itertools
import json
import logging
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)

DEFAULT_GRID = {
    "learning_rate": [0.05],
    "n_estimators": [300],
    "num_leaves": [7, 15],
    "colsample_bytree": [0.5, 1.0],
    "min_child_samples": [10],
}


def _matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[np.nan if v is None else v for v in r["features"]] for r in rows], dtype=float)


def _fit(X, y, params, seed):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(random_state=seed, class_weight="balanced", subsample=0.8, subsample_freq=1,
                             verbose=-1, n_jobs=4, deterministic=True, force_col_wise=True, **params)
    clf.fit(X, y)
    return clf


def _proba(clf, X, labels):
    P = np.zeros((len(X), len(labels)))
    li = {lab: i for i, lab in enumerate(labels)}
    for j, c in enumerate(clf.classes_):
        P[:, li[c]] = clf.predict_proba(X)[:, j]
    return P


def group_oof(X, y, groups, labels, params, seed, n_splits):
    P = np.zeros((len(X), len(labels)))
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = _fit(X[tr], y[tr], params, seed)
        P[va] = _proba(clf, X[va], labels)
    return P


def run_baseline(rows: list[dict], meta: dict, seed: int = 42, grid: dict | None = None,
                 n_splits: int | None = None) -> dict:
    labels = meta["labels"]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    Xtr, Xte = _matrix(train), _matrix(test)
    ytr = np.asarray([r["root_cause"] for r in train])
    gtr = np.asarray([r["group_id"] for r in train])
    k = n_splits or min(5, len(set(gtr)))
    grid = grid or DEFAULT_GRID
    keys = sorted(grid)
    results = []
    for combo in itertools.product(*(grid[k_] for k_ in keys)):
        params = dict(zip(keys, combo))
        P = group_oof(Xtr, ytr, gtr, labels, params, seed, k)
        pred = np.asarray(labels)[P.argmax(1)]
        mf1 = f1_score(ytr, pred, average="macro")
        results.append({"params": params, "cv_macro_f1": float(mf1), "cv_accuracy": float(np.mean(pred == ytr))})
        log.info("grid %s -> cv macro-F1 %.3f", params, mf1)
    best = max(results, key=lambda r: r["cv_macro_f1"])
    params = best["params"]
    oof = group_oof(Xtr, ytr, gtr, labels, params, seed, k)
    clf = _fit(Xtr, ytr, params, seed)
    Pte = _proba(clf, Xte, labels)
    out_rows = []
    for split_rows, P, src in ((train, oof, "oof"), (test, Pte, "fit_on_train")):
        for r, p in zip(split_rows, P):
            order = np.argsort(-p)
            out_rows.append({
                "case_id": r["case_id"], "split": r["split"], "prob_source": src,
                "probs": {labels[i]: float(p[i]) for i in range(len(labels))},
                "top3": [[labels[i], float(p[i])] for i in order[:3]],
                "pred": labels[int(order[0])],
            })
    return {"params": params, "cv": results, "n_splits": k, "predictions": out_rows, "model": clf}


def class_templates(rows: list[dict]) -> dict[str, dict]:
    """Most frequent affected-KPI list and ticket text per class, from the TRAIN split only."""
    out = {}
    for lab in sorted({r["root_cause"] for r in rows if r["split"] == "train"}):
        rs = [r for r in rows if r["split"] == "train" and r["root_cause"] == lab]
        aff = Counter(tuple(r["affected_kpis"]) for r in rs).most_common(1)[0][0]
        tk = Counter(json.dumps(r["ticket"], sort_keys=True) for r in rs).most_common(1)[0][0]
        out[lab] = {"affected_kpis": list(aff), "ticket": json.loads(tk)}
    return out


def template_output(row: dict, label: str, templates: dict) -> str:
    """Full JSON answer from a predicted class + evidence, no LLM involved."""
    from .evidence import evidence_from_dict
    from .prompts import observations

    t = templates[label]
    tk = t["ticket"]
    # observation sentence from this case's evidence, filtered by the PREDICTED class's KPI list
    obs = observations(evidence_from_dict(row["evidence"]), t["affected_kpis"])
    expl = " ".join(x for x in (tk.get("issue", ""), tk.get("root_cause", ""), obs) if x)
    return json.dumps({"root_cause": label, "affected_kpis": t["affected_kpis"], "explanation": expl,
                       "resolution": tk.get("resolution", "")})
