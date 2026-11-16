#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash from the branch tree with the B300 High-Throughput
# cookbook recipe (TP4 + EP4) and run GSM8K and RULER2 64K with sgl-eval.
#   bash run_e2e.sh <name> <gpus> <port> [VAR=VALUE ...] [-- extra server args ...]
set -uo pipefail

NAME=${1:?name}
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
OUT=$ROOT/e2e/$NAME
MODEL=/data/models/DeepSeek-V4.1-Flash-1017
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
EVAL=$ROOT/sgl-eval-venv/bin/sgl-eval
URL=http://127.0.0.1:$PORT
mkdir -p "$OUT"

unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
export PYTHONPATH=$ROOT/sglang/python
export SGLANG_CACHE_DIR=$ROOT/cache/sglang
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/sglang/flashinfer
# The tree under test carries the Rust TreeCore sources but no built extension, and the
# image sets SGLANG_RUST_BUILD_MODE=never, so pin the Python TreeCore (r1-r3 fell back
# to it while the tree had no rust/ sources).
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python

{
  echo "name=$NAME gpus=$GPUS port=$PORT"
  echo "env=${ENV_ARGS[*]:-}"
  echo "server_args=$*"
  echo "tree_core=$SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND rust_build_mode=${SGLANG_RUST_BUILD_MODE:-unset}"
  date -u +%FT%TZ
} > "$OUT/run.txt"

env CUDA_VISIBLE_DEVICES="$GPUS" "${ENV_ARGS[@]}" python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED" --trust-remote-code \
  --tp 4 --ep-size 4 --max-running-requests 256 \
  --reasoning-parser auto --tool-call-parser auto \
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

# The RULER2 generators spawn `python -m sgl_eval...`, so the eval venv comes first.
EVAL_PATH=$ROOT/sgl-eval-venv/bin:$PATH
for eval_name in ${E2E_EVALS:-gsm8k ruler2}; do
  case $eval_name in
    gsm8k)
      env PATH="$EVAL_PATH" "$EVAL" run gsm8k --base-url "$URL/v1" \
        --load-preset-from-model-id "$SERVED" --out-dir "$OUT" > "$OUT/gsm8k.log" 2>&1
      ;;
    ruler2)
      env PATH="$EVAL_PATH" "$EVAL" run ruler2 --base-url "$URL/v1" \
        --load-preset-from-model-id "$SERVED" --ruler2-seq-len 65536 \
        --ruler2-dataset-size 20 --ruler2-tokenizer "$MODEL" \
        --out-dir "$OUT" > "$OUT/ruler2.log" 2>&1
      ;;
    *)
      echo "unknown eval $eval_name" >> "$OUT/run.txt"
      exit 1
      ;;
  esac
  echo "$eval_name exit=$? $(date -u +%FT%TZ)" >> "$OUT/run.txt"
done
