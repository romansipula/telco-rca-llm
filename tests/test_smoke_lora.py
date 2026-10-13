"""End-to-end CPU smoke test: synthetic adapter -> prepare -> SFT data -> tiny LoRA -> generate -> evaluate."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def test_tiny_lora_end_to_end(tmp_path):
    from telco_rca.io import read_jsonl, write_jsonl
    from telco_rca.llm.data import build_examples
    from telco_rca.llm.hf_infer import infer
    from telco_rca.llm.train import train
    from telco_rca.metrics import evaluate_predictions
    from telco_rca.pipeline import load_prepared, prepare

    cfg = {"dataset": {"adapter": "synthetic", "kwargs": {"n_groups": 6, "cases_per_group": 3}},
           "split": {"manifest": str(tmp_path / "split.json"), "test_fraction": 0.34, "seed": 0},
           "evidence": {}, "prepared_dir": str(tmp_path / "prep")}
    prepare(cfg)
    rows, meta = load_prepared(tmp_path / "prep")
    assert {r["split"] for r in rows} == {"train", "test"}
    tr = build_examples(rows, meta, "evidence", "train")
    te = build_examples(rows, meta, "evidence", "test")
    write_jsonl(tmp_path / "tr.jsonl", tr)
    write_jsonl(tmp_path / "te.jsonl", te)
    mcfg = {"hf_id": "__tiny__", "compute_dtype": "float32",
            "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
            "train": {"epochs": 1, "batch_size": 2, "grad_accum": 1, "lr": 1e-3, "max_seq_len": 4096,
                      "logging_steps": 1}, "seed": 0}
    info = train(mcfg, str(tmp_path / "tr.jsonl"), str(tmp_path / "run"), max_steps=3)
    assert info["n_examples"] == len(tr) and info["fp16_probe"]["finite"]
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()
    infer(mcfg, str(tmp_path / "te.jsonl"), str(tmp_path / "pred.jsonl"), adapter=str(tmp_path / "run" / "adapter"),
          limit=2, batch_size=2, max_new_tokens=8)
    preds = read_jsonl(tmp_path / "pred.jsonl")
    assert len(preds) == 2 and all("output" in p for p in preds)
    m = evaluate_predictions(preds, [r for r in rows if r["split"] == "test"], meta, text_metrics=False)
    assert m["n"] == 2 and 0.0 <= m["json_valid_rate"] <= 1.0
    # the SFT target itself must satisfy the strict schema
    oracle = [{"case_id": e["case_id"], "output": e["target"]} for e in te]
    mo = evaluate_predictions(oracle, [r for r in rows if r["split"] == "test"], meta, text_metrics=False)
    assert mo["schema_valid_rate"] == 1.0 and mo["root_cause"]["accuracy"] == 1.0
    assert json.loads(te[0]["target"])["root_cause"] in meta["labels"]
