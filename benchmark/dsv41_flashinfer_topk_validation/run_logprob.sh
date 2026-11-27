#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash from the PR tree (the accuracy runs' recipe plus
# --disable-radix-cache) and collect greedy logprobs on the RULER2 64K prompts twice:
# pass 1 over PASS1_PER_TASK rows per subtask, pass 2 over the first PASS2_PER_TASK.
#   bash run_logprob.sh <label> <gpus> <port> [VAR=VALUE ...] [-- extra server args ...]
set -uo pipefail

LABEL=${1:?label}
GPUS=${2:?gpus}
PORT=${3:?port}
shift 3
ENV_ARGS=()
while [[ $# -gt 0 && $1 != -- ]]; do
  ENV_ARGS+=("$1")
  shift
done
[[ ${1:-} == -- ]] && shift

ROOT=/hai-workspace/dsv41-fi-topk
OUT=$ROOT/${LOGPROB_DIR:-logprob}/$LABEL
DATA=/root/.cache/sgl_eval/ruler2/data-models-DeepSeek-V4.1-Flash-1017_65536_n20
MODEL=/data/models/DeepSeek-V4.1-Flash-1017
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"

unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
export PYTHONPATH=$ROOT/sglang/python
export SGLANG_CACHE_DIR=$ROOT/cache/sglang
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/sglang/flashinfer
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python

{
  echo "label=$LABEL gpus=$GPUS port=$PORT"
  echo "env=${ENV_ARGS[*]:-}"
  echo "server_args=--disable-radix-cache $*"
  echo "pass1_per_task=${PASS1_PER_TASK:-20} pass2_per_task=${PASS2_PER_TASK:-5} max_new_tokens=32 collect_args=${COLLECT_ARGS:-}"
  date -u +%FT%TZ
} > "$OUT/run.txt"

env CUDA_VISIBLE_DEVICES="$GPUS" "${ENV_ARGS[@]}" python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --trust-remote-code \
  --tp 4 --ep-size 4 --max-running-requests 256 \
  --reasoning-parser auto --tool-call-parser auto --disable-radix-cache \
  --host 127.0.0.1 --port "$PORT" "$@" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

until curl -sf "$URL/health" > /dev/null; do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited before becoming healthy" | tee -a "$OUT/run.txt"
    exit 1
  fi
  sleep 10
done
echo "healthy $(date -u +%FT%TZ)" >> "$OUT/run.txt"

python3 "$ROOT/tools/logprob_equivalence.py" collect --url "$URL" --data-dir "$DATA" \
  --per-task "${PASS1_PER_TASK:-20}" ${COLLECT_ARGS:-} --out "$OUT/pass1.jsonl" > "$OUT/pass1.log" 2>&1
echo "pass1 exit=$? $(date -u +%FT%TZ)" >> "$OUT/run.txt"
python3 "$ROOT/tools/logprob_equivalence.py" collect --url "$URL" --data-dir "$DATA" \
  --per-task "${PASS2_PER_TASK:-5}" ${COLLECT_ARGS:-} --out "$OUT/pass2.jsonl" > "$OUT/pass2.log" 2>&1
echo "pass2 exit=$? $(date -u +%FT%TZ)" >> "$OUT/run.txt"
