# Adapting telco-rca to your own KPI data

This project is a proxy for an operator-internal system. TelecomTS is wired in through one adapter
(`src/telco_rca/data/telecomts.py`). Everything after the adapter is dataset-agnostic and has no
dependency on Kaggle. This guide covers how to plug in proprietary data, how to label it, and how to
train and serve on your own hardware.

```
your KPI store ──► Adapter ──► Case ──► evidence (deterministic) ──► prompt ──► LLM ──► strict JSON
                     │                       │                                         │
                     └─ reference profile ───┘          GB baseline (same features) ───┘ (hybrid)
```

## 1. Write an adapter

Implement `telco_rca.data.base.DatasetAdapter`. `src/telco_rca/data/synthetic.py` is a complete
example of about 100 lines.

```python
from telco_rca.data.base import Case, DatasetAdapter, KPISpec, Ticket, robust_profile

class MyOSSAdapter(DatasetAdapter):
    name = "my_oss"

    def kpis(self):            # every channel you will feed, with unit and a one-line meaning
        return [KPISpec("DL_PRB_Util", "%", "downlink PRB utilisation"),
                KPISpec("RRC_SR", "%", "RRC setup success rate"),
                KPISpec("Avg_CQI", "", "average CQI"), ...]

    def label_set(self):       # closed list of root causes (what engineers actually close tickets with)
        return ["Transport congestion", "Antenna/RET fault", "Uplink interference", ...]

    def load_cases(self):      # one Case per detected incident
        for inc in my_incidents():
            yield Case(
                case_id=inc.ticket_id,
                group_id=inc.cell_id,                 # or site / cluster: the leakage unit, see below
                numeric={k: np.array(...) for k in kpi_names},   # same length T for all KPIs
                categorical={},                       # optional str channels
                sampling_seconds=900,                 # 15-min PM counters
                segment=(start_idx, end_idx),         # detector's interval inside the window, or None
                context={"cell": inc.cell_id, "vendor": inc.vendor, "band": inc.band},
                root_cause=inc.closed_root_cause,
                affected_kpis=inc.affected_kpis,
                ticket=Ticket(issue=..., symptoms=..., root_cause=..., resolution=...),
                source=self.name)

    def reference_profile(self, case):   # normal behaviour of THIS cell before the incident
        hist = load_history(case.context["cell"], end=incident_start, days=14, exclude_alarms=True)
        return robust_profile(hist.numeric, hist.categorical)
```

Then register it (`telco_rca.data.ADAPTERS["my_oss"] = MyOSSAdapter`, or add a branch in
`get_adapter`). Copy `configs/telecomts.yaml` and change `dataset.adapter`, the split manifest path
and `prepared_dir`. After that, run the same commands as for TelecomTS:

```bash
telco-rca prepare --config configs/my_oss.yaml --resplit --stats-out results/my_oss/data_stats.json
telco-rca baseline --config configs/my_oss.yaml --out results/my_oss/baseline
telco-rca build-sft --config configs/my_oss.yaml --mode hybrid --gb-probs results/my_oss/baseline/gb_probs.jsonl --out data_cache/my_oss/sft
```

### Choices that matter for real operator KPIs (1-15 min aggregates)

* **Window and segment.** With 15-min PM data, use roughly 2-7 days of history as the window (192-672
  steps) and pass the detector's interval as `segment`. The evidence block is built from segment-vs-
  baseline statistics, so it is scale-free and does not care whether one step is 100 ms or 15 min.
  Only `min_local_baseline` (steps) and the reference history length need retuning.
* **Reference profile.** Seasonality dominates hourly and daily KPIs. Build the profile from the same
  hour-of-week over the previous 2-4 weeks (e.g. median and MAD per hour-of-week slot), and subtract
  the slot median before extraction, or pass per-slot profiles. TelecomTS (12.8 s windows) has no
  seasonality, so this code path is the part that most needs work for real data.
* **group_id = the leakage unit.** Use at least the cell. Better is the site or the cluster when one
  physical fault shows up on neighbours. Use time blocks as well if you will evaluate on future
  incidents (train on months 1-9, test on 10-12). The split function guarantees that no group
  crosses splits. Do not use a random incident-level split: on TelecomTS that inflated the GB
  baseline from 0.744 to 0.960 accuracy (`results/baseline/gb_random_split/metrics.json`).
* **Counters vs KPIs.** Feed ratios and rates (success rates, utilisation, BLER, CQI, throughput per
  user) rather than raw cumulative counters. Add the denominators as their own KPIs (attempts, active
  users) so the model can tell "low success" apart from "no traffic".
* **Categorical channels** (vendor alarms, config-change flags, neighbour-relation changes) go into
  `categorical`. The evidence block reports share shifts between baseline and segment.

## 2. Labels you need

Per incident:

| field | required | notes |
|---|---|---|
| KPI window + detector interval | yes | what the upstream anomaly detector already has |
| `root_cause` | yes | from a closed list of 8-20 classes; map free-text closure codes to it |
| `group_id` | yes | cell / site / cluster |
| `affected_kpis` | recommended | KPIs that an engineer considers symptomatic of this incident (not of the class) |
| ticket `issue`, `root_cause`, `resolution` | recommended | from the closed trouble ticket; rewrite into 1-2 sentences each |
| `symptoms` | optional | used only as a metric reference |

## 3. Annotation guide for ~200 incidents

Goal: about 200 labelled incidents with at least 10 per class, spread over as many cells and sites
as possible.

1. **Sample.** Take closed tickets that link to a detector alarm. Stratify by root cause and by
   vendor/region. Cap each site at 5 incidents.
2. **Fix the label set first.** Two engineers map the last 6-12 months of closure codes to a short
   list of root causes. Merge rare codes into "Other (specify)", and drop incidents that stay
   ambiguous after review.
3. **Per incident the annotator gets** the rendered evidence block (`telco-rca prepare` output), KPI
   plots of the window and the original ticket, and fills in:
   * `root_cause` (from the list);
   * `affected_kpis`: KPIs that visibly deviate *because of this fault*. Do not list every KPI the
     class could touch. This is what makes the KPI F1 metric informative, unlike TelecomTS where the
     list is a class constant;
   * `explanation` (2-4 sentences): what failed and which KPIs show it, with numbers taken from the
     evidence block. No information that is not visible in the data or the ticket;
   * `resolution` (1-2 sentences): what was actually done, in ticket style ("RET re-aligned on sector 2;
     VSWR alarm cleared"), not a generic playbook;
   * a confidence of 1-3, and a flag "evidence block insufficient" with a free-text reason. These flags
     tell you which KPIs to add next.
4. **Double-annotate 20%.** Report Cohen's kappa on `root_cause` and the Jaccard overlap on
   `affected_kpis`. If kappa < 0.6, refine the label definitions and redo that batch.
5. **Freeze a test set** of ≥50 incidents from sites never seen in training before you look at any
   model output.
6. Expect about 10-15 minutes per incident, i.e. about 40-50 engineer-hours for 200 incidents
   including double annotation.

200 examples are enough for QLoRA to learn the output format and the label vocabulary. They are not
enough to make the LLM a better classifier than gradient boosting on the same evidence. That is why
the hybrid mode (GB probabilities in the prompt) and the GB+template system are part of the pipeline.

## 4. Train on your own GPU (no Kaggle)

Any CUDA GPU with ≥16 GB works for the 3-4B models with QLoRA. Everything is configured in YAML.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124      # match your CUDA
pip install -e ".[train,eval]" "transformers==5.19.0" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes>=0.46"
telco-rca train   --model-config configs/models/ministral3_3b.yaml \
                  --train-file data_cache/my_oss/sft/hybrid_train.jsonl --output-dir runs/my_oss/ministral_hybrid
telco-rca infer-hf --model-config configs/models/ministral3_3b.yaml --adapter runs/my_oss/ministral_hybrid/adapter \
                  --examples data_cache/my_oss/sft/hybrid_test.jsonl --out runs/my_oss/ministral_hybrid/predictions.jsonl
telco-rca evaluate --prepared data_cache/prepared/my_oss --pred runs/my_oss/ministral_hybrid/predictions.jsonl
```

* With bf16-capable GPUs (A100, L4, RTX 30xx+), set `compute_dtype: bfloat16` in the model YAML. The
  fp16 settings exist only because of the T4.
* `kaggle/gpu_train/run.py` is just a job runner around these same commands. You can run it on any
  2-GPU box after editing the paths at the top.

## 5. CPU-only inference on your own servers

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF && cmake --build build -j --target llama-server llama-quantize
pip install -e gguf-py
cd .. && telco-rca export-gguf --model-config configs/models/ministral3_3b.yaml \
      --adapter runs/my_oss/ministral_hybrid/adapter --out-dir models/ --llama-cpp ./llama.cpp --quants Q8_0,Q4_K_M
telco-rca infer-cpu --gguf models/ministral3-3b-rca-Q4_K_M.gguf --model-config configs/models/ministral3_3b.yaml \
      --examples data_cache/my_oss/sft/hybrid_test.jsonl --out runs/cpu/predictions.jsonl --llama-cpp ./llama.cpp --threads 16
```

* Only the tokenizer files are fetched from the Hugging Face Hub, to render the chat template. To run
  air-gapped, copy them once (`huggingface-cli download <id> --include "tokenizer*" "*.jinja"`) and set
  `tokenizer_id:` in the YAML to the local directory.
* For batch processing, run one `llama-server` per NUMA node with `-np 1 -t <cores per node>`. Prompts
  share the system-prompt prefix, which `cache_prompt` reuses.
* The runtime needs the GB model too (`gb_model.joblib`, written by `telco-rca baseline --save-model`)
  to fill the hybrid prompt. It is a few MB and has no GPU dependency.
* Validate every answer with `telco_rca.schema.parse_output`. If an answer is not schema-valid, fall
  back to the GB+template answer (`telco_rca.baseline.template_output`) so a ticket is always produced.

## 6. Where 3GPP and vendor-manual retrieval plugs in

The prompt has two slots: the evidence block and (optional) GB probabilities. Retrieval fits in as a
third slot, inserted in `telco_rca.prompts`:

1. **Query** = top deviating KPIs + shapes + the GB top-3 candidates (all already in the evidence
   object). Example: "UL_SNR step down, BLER up, RSRP stable; candidates: uplink interference, RET fault".
2. **Corpus.** Index 3GPP TS 28.552 / 32.425 (counter definitions), TS 38.300 / 38.331 (procedures),
   vendor counter reference manuals, and your own closed-ticket resolutions. Chunk them per counter or
   per procedure. Use an embedding model of non-Chinese origin to match the model policy here (for
   example IBM `granite-embedding-*`, `nomic-embed-text`, or a BM25 baseline, which is surprisingly
   strong on counter names).
3. **Prompt.** Add 2-4 short passages under a `REFERENCE` heading. Keep the evidence block first, so
   the model grounds on data and uses the manuals for terminology and resolution steps.
4. **Train with retrieval in the prompt** (same SFT recipe). Otherwise the model learns to ignore the
   new section.
5. **Measure** with the same metrics, plus how many cited counters exist in the retrieved passages.
   The `explanation_kpi_grounding` metric shows how to check for cited-but-not-deviating KPIs.

The most valuable corpus is usually the operator's own resolved tickets (retrieve the five most
similar past incidents by evidence features). The vendor manuals come second.
