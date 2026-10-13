"""Evaluation: classification, KPI-set F1, text similarity, JSON validity, grounding, cost."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .schema import parse_output


def set_f1(pred: list[str], gold: list[str]) -> float:
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def classification(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    yp = [p if p is not None else "__invalid__" for p in y_pred]
    present = [lab for lab in labels if lab in set(y_true)]
    acc = accuracy_score(y_true, yp)
    # macro-F1 over the classes present in the gold labels; invalid outputs count as errors
    mf1 = f1_score(y_true, yp, labels=present, average="macro", zero_division=0)
    per_class = f1_score(y_true, yp, labels=present, average=None, zero_division=0)
    cm = confusion_matrix(y_true, yp, labels=labels + ["__invalid__"])
    return {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "per_class_f1": {lab: float(f) for lab, f in zip(present, per_class)},
        "confusion_matrix": {"labels": labels + ["__invalid__"], "matrix": cm.tolist()},
        "n": len(y_true),
    }


def rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [sc.score(r, p)["rougeL"].fmeasure if p else 0.0 for p, r in zip(preds, refs)]
    return float(np.mean(vals)) if vals else 0.0


_BERT_CACHE: dict = {}


def bertscore(preds: list[str], refs: list[str], model_type: str = "roberta-large", device: str | None = None
              ) -> dict:
    """BERTScore F1 with a non-Chinese encoder (roberta-large by default); raw and baseline-rescaled."""
    from bert_score import score

    preds = [p if p else "." for p in preds]
    out = {}
    for rescale in (False, True):
        _, _, f = score(preds, refs, model_type=model_type, lang="en", rescale_with_baseline=rescale,
                        device=device, batch_size=32, verbose=False)
        out["f1_rescaled" if rescale else "f1"] = float(f.mean())
    out["model"] = model_type
    return out


def mentioned_kpis(text: str, kpis: list[str]) -> list[str]:
    found = []
    for k in kpis:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])", text):
            found.append(k)
    return found


def evaluate_predictions(preds: list[dict], cases: list[dict], meta: dict, text_metrics: bool = True,
                         bert: bool = True, bert_device: str | None = None) -> dict:
    """``preds``: rows with ``case_id`` and ``output`` (raw model text). ``cases``: prepared rows."""
    by_id = {c["case_id"]: c for c in cases}
    schema = meta["schema"]
    labels = meta["labels"]
    kpi_names = [k["name"] for k in meta["kpis"]]
    y_true, y_pred, f1s, expl, resol, ref_d, ref_r = [], [], [], [], [], [], []
    json_ok = schema_ok = 0
    grounded, mentions, halluc = [], 0, 0
    for p in preds:
        c = by_id[p["case_id"]]
        po = parse_output(p.get("output", ""), schema)
        json_ok += po.json_ok
        schema_ok += po.schema_ok
        y_true.append(c["root_cause"])
        y_pred.append(po.root_cause)
        f1s.append(set_f1(po.affected_kpis, c["affected_kpis"]))
        expl.append(po.explanation)
        resol.append(po.resolution)
        ref_d.append(c["reference_diagnosis"])
        ref_r.append(c["reference_resolution"])
        m = mentioned_kpis(po.explanation, kpi_names)
        if m:
            ok = [k for k in m if k in c["deviating"]]
            grounded.append(len(ok) / len(m))
            mentions += len(m)
            halluc += len(m) - len(ok)
    n = len(preds)
    res = {
        "n": n,
        "json_valid_rate": json_ok / n if n else 0.0,
        "schema_valid_rate": schema_ok / n if n else 0.0,
        "root_cause": classification(y_true, y_pred, labels),
        "affected_kpis_set_f1": float(np.mean(f1s)) if f1s else 0.0,
        "explanation_kpi_grounding": {
            "mean_fraction_of_mentioned_kpis_that_deviate": float(np.mean(grounded)) if grounded else None,
            "cases_mentioning_kpis": len(grounded),
            "mentions": mentions,
            "mentions_not_in_deviating_list": halluc,
        },
    }
    # accuracy without the Jamming class, for comparability with TelecomTS Table 4 (10 synthetic types)
    keep = [i for i, t in enumerate(y_true) if t != "Jamming"]
    if keep and len(keep) < n:
        res["root_cause_excl_jamming"] = {
            "accuracy": float(np.mean([y_true[i] == y_pred[i] for i in keep])), "n": len(keep)}
    if text_metrics:
        res["explanation_rougeL"] = rouge_l(expl, ref_d)
        res["resolution_rougeL"] = rouge_l(resol, ref_r)
        if bert:
            res["explanation_bertscore"] = bertscore(expl, ref_d, device=bert_device)
            res["resolution_bertscore"] = bertscore(resol, ref_r, device=bert_device)
    res["pred_distribution"] = dict(Counter(str(x) for x in y_pred).most_common())
    return res


def cost_summary(preds: list[dict]) -> dict:
    keys = ["prompt_tokens", "output_tokens", "seconds"]
    out = {}
    for k in keys:
        vals = [p[k] for p in preds if p.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                      "p95": float(np.quantile(vals, 0.95)), "total": float(np.sum(vals))}
    return out


def plot_confusion(cm: dict, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = cm["labels"]
    M = np.asarray(cm["matrix"], dtype=float)
    # drop the invalid column/row if unused
    if M[:, -1].sum() == 0 and M[-1].sum() == 0:
        labels, M = labels[:-1], M[:-1, :-1]
    rows = M.sum(1, keepdims=True)
    N = np.divide(M, rows, out=np.zeros_like(M), where=rows > 0)
    short = [re.sub(r"\s*\((.*?)\)", lambda m: f" ({m.group(1)[:6]})", lab)[:24] for lab in labels]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if N[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(short)), short, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
