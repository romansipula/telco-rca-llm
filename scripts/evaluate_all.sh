#!/usr/bin/env bash
# Recompute metrics.json (+ confusion.png) for every system in results/systems.json, then the summary.
#   scripts/evaluate_all.sh [prepared_dir]
# BERTScore uses roberta-large (downloaded once, ~1.4 GB); NO_BERT=1 skips it. Existing metrics are
# reused unless the predictions are newer or FORCE=1.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREP=${1:-$ROOT/data_cache/prepared/telecomts}
EXTRA=${NO_BERT:+--no-bert}
cd "$ROOT"
python3 - <<'PY' | while IFS=$'\t' read -r name pred; do
import json
m = json.load(open("results/systems.json"))
for k, v in m.items():
    if not k.startswith("_") and v.get("pred") and not v.get("label_only"):
        print(f"{k}\t{v['pred']}")
PY
  d=$(dirname "$pred")
  if [ -s "$pred" ] && { [ ! -f "$d/metrics.json" ] || [ "$pred" -nt "$d/metrics.json" ] || [ -n "${FORCE:-}" ]; }; then
    echo "== $name"
    telco-rca evaluate --prepared "$PREP" --pred "$pred" --name "$name" $EXTRA
  fi
done
telco-rca report --results results --out results/summary.md > /dev/null
echo "wrote results/summary.md"
