#!/usr/bin/env bash
# Recompute metrics.json (+ confusion.png) for every predictions.jsonl under results/, then the summary.
#   scripts/evaluate_all.sh [prepared_dir]
# BERTScore uses roberta-large (downloaded once, ~1.4 GB); set NO_BERT=1 to skip it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREP=${1:-$ROOT/data_cache/prepared/telecomts}
EXTRA=${NO_BERT:+--no-bert}
cd "$ROOT"
find results -name predictions.jsonl -not -path "*/raw_download/*" | sort | while read -r p; do
  d=$(dirname "$p")
  case "$d" in results/baseline/gb|results/baseline/majority) continue ;; esac  # label-only systems, written by `baseline`
  if [ -s "$p" ] && { [ ! -f "$d/metrics.json" ] || [ "$p" -nt "$d/metrics.json" ] || [ -n "${FORCE:-}" ]; }; then
    echo "== $d"
    telco-rca evaluate --prepared "$PREP" --pred "$p" --name "${d#results/}" $EXTRA
  fi
done
telco-rca report --results results --out results/summary.md > /dev/null
echo "wrote results/summary.md"
