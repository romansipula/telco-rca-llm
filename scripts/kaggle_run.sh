#!/usr/bin/env bash
# Push a Kaggle kernel, poll it, download its outputs into results/<run_id>/.
#
#   scripts/kaggle_run.sh <gpu|cpu> <kernel-slug> <run_id> '<json spec overrides>' [kernel_source ...]
#
# Examples
#   scripts/kaggle_run.sh gpu telco-rca-gpu-gemma gpu_gemma4 '{"models":["gemma4_e4b"]}'
#   scripts/kaggle_run.sh cpu telco-rca-export-gemma export_gemma4 \
#       '{"task":"export","model":"gemma4_e4b","adapter_glob":"gemma4_e4b/sft_hybrid/adapter"}' \
#       <user>/telco-rca-gpu-gemma
#
# Needs the kaggle CLI and KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json).
# Environment:
#   REF=<git branch/commit the kernel installs>   (default: current branch, must be pushed)
#   POLL=<seconds between status checks>          (default 600, i.e. at most every 10 min)
#   NO_WAIT=1                                     push only, do not poll/download
set -euo pipefail

KIND=$1; SLUG=$2; RUN_ID=$3; OVERRIDES=${4:-'{}'}; shift 4 || true
SOURCES=("$@")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REF=${REF:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}
REPO_URL=${REPO_URL:-$(git -C "$ROOT" remote get-url origin | sed -E 's#^git@github.com:#https://github.com/#; s#^https?://[^@]*@#https://#; s#/git/#/#; s#^http://127.0.0.1:[0-9]+/#https://github.com/#')}
POLL=${POLL:-600}
USER_NAME=$(python3 -c "from kaggle.api.kaggle_api_extended import KaggleApi as A; a=A(); a.authenticate(); print(a.get_config_value('username'))" 2>/dev/null | tail -1)
OUT="$ROOT/results/$RUN_ID"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

case $KIND in
  gpu) SRC="$ROOT/kaggle/gpu_train/run.py"; GPU=true; SHAPE='"machine_shape": "NvidiaTeslaT4",' ;;
  cpu) SRC="$ROOT/kaggle/cpu_eval/run.py"; GPU=false; SHAPE='' ;;
  *) echo "kind must be gpu or cpu"; exit 2 ;;
esac

# bake the spec into the script
python3 - "$SRC" "$WORK/run.py" "$OVERRIDES" "$REPO_URL" "$REF" <<'PY'
import ast, json, re, sys
src, dst, ov, repo, ref = sys.argv[1:]
code = open(src).read()
m = re.search(r"^SPEC = (\{.*?\})  # @@SPEC@@$", code, re.S | re.M)
spec = ast.literal_eval(m.group(1))
spec.update({"repo": repo, "ref": ref})
spec.update(json.loads(ov))
code = code[:m.start()] + "SPEC = " + repr(spec) + "  # @@SPEC@@" + code[m.end():]
open(dst, "w").write(code)
print(json.dumps(spec))
PY

SRC_JSON=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${SOURCES[@]+"${SOURCES[@]}"}")
cat > "$WORK/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/$SLUG",
  "title": "$SLUG",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $GPU,
  $SHAPE
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": $SRC_JSON
}
EOF

echo "pushing $USER_NAME/$SLUG (ref $REF)"
kaggle kernels push -p "$WORK"
mkdir -p "$OUT"
echo "{\"kernel\": \"$USER_NAME/$SLUG\", \"kind\": \"$KIND\", \"ref\": \"$REF\", \"commit\": \"$(git -C "$ROOT" rev-parse HEAD)\", \"pushed_at\": \"$(date -u +%FT%TZ)\", \"overrides\": $OVERRIDES}" > "$OUT/kaggle_run.json"
[ -n "${NO_WAIT:-}" ] && exit 0

"$ROOT/scripts/kaggle_fetch.sh" "$USER_NAME/$SLUG" "$RUN_ID"
