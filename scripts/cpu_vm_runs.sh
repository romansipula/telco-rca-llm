#!/usr/bin/env bash
# CPU-only runs on this VM, strictly one at a time (timings must not share the CPU with anything else),
# on the fixed 100-case subset results/cpu_subset_ids.json (the Kaggle CPU kernels cover all 308 cases).
#   GGUF_DIR=/path/to/ggufs LLAMA_CPP=/path/to/llama.cpp scripts/cpu_vm_runs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
G=${GGUF_DIR:-/home/user/gguf}
L=${LLAMA_CPP:-/home/user/llama.cpp}
T=${THREADS:-4}
run() {  # name gguf model-config examples
  telco-rca infer-cpu --gguf "$2" --model-config "$3" --examples "$4" --out "results/$1/predictions.jsonl" \
    --llama-cpp "$L" --threads "$T"
}
run cpu_vm_gemma4_hybrid_q4_k_m_sub100 "$G/gemma4_hybrid/gguf/gemma4-e4b-rca-Q4_K_M.gguf" configs/models/gemma4_e4b.yaml data_cache/sft/hybrid_test_sub100.jsonl
run cpu_vm_gemma4_hybrid_q8_0_sub100 "$G/gemma4_hybrid/gguf/gemma4-e4b-rca-Q8_0.gguf" configs/models/gemma4_e4b.yaml data_cache/sft/hybrid_test_sub100.jsonl
run cpu_vm_ministral3_hybrid_q4_k_m_sub100 "$G/ministral3_hybrid/ministral3-3b-rca-Q4_K_M.gguf" configs/models/ministral3_3b.yaml data_cache/sft/hybrid_test_sub100.jsonl
