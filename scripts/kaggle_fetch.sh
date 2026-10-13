#!/usr/bin/env bash
# Poll a pushed Kaggle kernel until it finishes, then download its output.
#   scripts/kaggle_fetch.sh <user/slug> <run_id>
# Large binaries (adapters, GGUF) stay in results/<run_id>/raw_download (git-ignored); small result
# files (predictions, logs, metrics, train logs) are copied to results/<run_id>/ for committing.
set -euo pipefail
KERNEL=$1; RUN_ID=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
POLL=${POLL:-600}
OUT="$ROOT/results/$RUN_ID"
mkdir -p "$OUT/raw_download"
while true; do
  S=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  echo "$(date -u +%T) $S"
  case "$S" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep "$POLL"
done
kaggle kernels output "$KERNEL" -p "$OUT/raw_download" -o > "$OUT/download.log" 2>&1 || true
echo "$S" > "$OUT/kernel_status.txt"
# copy small files (<= 5 MB) preserving layout
( cd "$OUT/raw_download" && find . -type f -size -5M \
    ! -path "*/adapter/*" ! -path "./sft/*" ! -name "*.gguf" ! -name "*.safetensors" -print0 \
  | while IFS= read -r -d '' f; do mkdir -p "$OUT/$(dirname "$f")"; cp "$f" "$OUT/$f"; done )
echo "outputs in $OUT"
