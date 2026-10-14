#!/usr/bin/env bash
# Model RAM on this VM: 10 cases per GGUF with llama-server's host prompt cache disabled
# (--cache-ram 0). The main CPU runs used llama.cpp's default 8 GB prompt cache, which dominates
# their peak RSS.
set -euo pipefail
cd "$(dirname "$0")/.."
G=${GGUF_DIR:-/home/user/gguf}
L=${LLAMA_CPP:-/home/user/llama.cpp}
mem() {  # name gguf model-config
  telco-rca infer-cpu --gguf "$2" --model-config "$3" --examples data_cache/sft/hybrid_test_sub100.jsonl \
    --out "results/cpu_vm_memory/$1/predictions.jsonl" --llama-cpp "$L" --threads 4 --limit 10 --cache-ram 0
}
mem gemma4_q4_k_m "$G/gemma4_hybrid/gguf/gemma4-e4b-rca-Q4_K_M.gguf" configs/models/gemma4_e4b.yaml
mem gemma4_q8_0 "$G/gemma4_hybrid/gguf/gemma4-e4b-rca-Q8_0.gguf" configs/models/gemma4_e4b.yaml
mem ministral3_q4_k_m "$G/ministral3_hybrid/ministral3-3b-rca-Q4_K_M.gguf" configs/models/ministral3_3b.yaml
mem ministral3_q8_0 "$G/ministral3_hybrid/ministral3-3b-rca-Q8_0.gguf" configs/models/ministral3_3b.yaml
