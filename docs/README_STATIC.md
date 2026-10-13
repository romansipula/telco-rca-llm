## Architecture

```
 detected anomaly (KPI window + detector interval)
        │
        ▼
 ┌──────────────────────────┐   per KPI: baseline (in-window normal steps → same cell's unflagged
 │ evidence extraction      │   history → comparable normal cell), robust z of segment median,
 │ (deterministic, numpy)   │   relative change, shape (step/ramp/spikes/oscillating/variance),
 └──────────┬───────────────┘   reaction lag, % of segment outside the band, categorical shifts
            │ text block (~450 tokens)          │ feature vector (198 features)
            ▼                                   ▼
 ┌──────────────────────────┐       ┌──────────────────────────┐
 │ small LLM (QLoRA SFT)    │◄──────│ LightGBM (root cause)    │  hybrid: top-3 class probabilities
 │ Gemma 4 E4B / Ministral 3│ top-3 └──────────────────────────┘  go into the prompt
 └──────────┬───────────────┘
            ▼
 strict JSON {root_cause, affected_kpis, explanation, resolution}  ── jsonschema-validated
            │
            ▼
 GGUF (Q8_0 / Q4_K_M) → llama.cpp llama-server, CPU only
```

* `src/telco_rca/data/`: the adapter interface (`base.py`), the TelecomTS adapter, a synthetic adapter
  (tests and worked example), and the grouped split.
* `src/telco_rca/evidence.py`: evidence extraction, the text rendering and the feature vector.
* `src/telco_rca/prompts.py`, `schema.py`: prompts for the three input modes; SFT target; output
  contract and parser.
* `src/telco_rca/baseline.py`: LightGBM baseline, out-of-fold probabilities, the GB+ticket-template
  system.
* `src/telco_rca/llm/`: QLoRA training, HF batched inference, GGUF export, and llama.cpp CPU
  inference with cost measurement.
* `src/telco_rca/metrics.py`, `review.py`, `report.py`: metrics, the blind review sheet, and summary
  tables.
* `configs/`: data/evidence YAML, one YAML per model, and the committed split manifest.
* `kaggle/gpu_train/run.py`: GPU kernel (installs this repo from GitHub; job graph on 2×T4).
  `kaggle/cpu_eval/run.py`: CPU kernel (GGUF export or llama.cpp inference).
  `scripts/kaggle_run.sh` / `kaggle_fetch.sh`: push, poll (≤ every 10 min), and download into
  `results/<run_id>/`.
* `tests/`: no recording leakage across splits, evidence on synthetic series with known answers,
  JSON schema, a determinism regression test, and a tiny CPU LoRA end-to-end test.
  `.github/workflows/ci.yml` runs them on CPU.

## Reproduce

```bash
pip install -e ".[eval,dev]"                       # + torch (CPU wheel is enough for everything below)
telco-rca prepare --stats-out results/data_stats.json   # downloads TelecomTS from HF, evidence, split check
telco-rca baseline --out results/baseline --random-split --save-model
python scripts/data_audit.py
pytest -q

# GPU runs on Kaggle (needs KAGGLE_API_TOKEN); each line pushes, polls ≤ every 10 min, downloads to results/<id>/
scripts/kaggle_run.sh gpu telco-rca-gpu-gemma4     gpu_gemma4     '{"models":["gemma4_e4b"]}'
scripts/kaggle_run.sh gpu telco-rca-gpu-ministral3 gpu_ministral3 '{"models":["ministral3_3b"]}'
# GGUF export of the chosen adapter (CPU kernel), then CPU inference (Kaggle CPU and this machine)
scripts/kaggle_run.sh cpu telco-rca-export-<m> export_<m> '{"task":"export","model":"<m>","adapter_glob":"<m>/sft_hybrid/adapter"}' <user>/telco-rca-gpu-<m>
DATASETS=<user>/telco-rca-prompts scripts/kaggle_run.sh cpu telco-rca-cpu-<m>-q4 cpu_kaggle_<m>_q4 \
  '{"task":"infer","model":"<m>","gguf_glob":"*Q4_K_M.gguf","mode":"hybrid","examples_glob":"hybrid_test.jsonl"}' <user>/telco-rca-export-<m>
telco-rca infer-cpu --gguf <file>.gguf --model-config configs/models/<m>.yaml --examples data_cache/sft/hybrid_test.jsonl \
  --out results/cpu_vm_<m>_q4/predictions.jsonl --llama-cpp ~/llama.cpp
scripts/evaluate_all.sh && python scripts/cpu_report.py --gpu-ref <gpu predictions> --cpu results/cpu_*
```

Without a Kaggle token, the same commands work on any CUDA machine (see ADAPTING.md §4). The
Kaggle kernels are only thin job runners around the CLI.
