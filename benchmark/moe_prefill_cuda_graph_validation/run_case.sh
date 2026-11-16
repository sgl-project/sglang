#!/usr/bin/env bash
# Run one registered e2e test class on four devbox GPUs.
#   bash run_case.sh <gpus, e.g. 0,1,2,3> <run name> <test file> <TestClass | all>
# TREE selects the checkout under ROOT (default: sglang). Each run gets a fresh
# SGLANG_CACHE_DIR, so FlashInfer autotuning starts cold, unless CACHE_NAME names a
# shared one (kept between runs); JIT caches are always shared.
set -uo pipefail

GPUS=${1:?gpus}
RUN=${2:?run name}
FILE=${3:?test file}
CLASS=${4:?test class}
ROOT=${ROOT:-/hai-workspace/moe-prefill-cg}
TREE=${TREE:-sglang}
LOG_DIR=$ROOT/logs/$RUN
mkdir -p "$LOG_DIR"

cd "$ROOT/$TREE"
export PYTHONPATH="$ROOT/$TREE/python"
if [[ -n "${CACHE_NAME:-}" ]]; then
  export SGLANG_CACHE_DIR="$ROOT/cache/shared/$CACHE_NAME"
else
  export SGLANG_CACHE_DIR="$ROOT/cache/runs/$RUN"
  rm -rf "$SGLANG_CACHE_DIR"
fi
export FLASHINFER_WORKSPACE_BASE="$ROOT/cache/flashinfer-ws"
export CUTE_DSL_CACHE_DIR="$ROOT/cache/cute-dsl"
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONUNBUFFERED=1
mkdir -p "$SGLANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CUTE_DSL_CACHE_DIR"

{
  echo "run=$RUN tree=$TREE gpus=$GPUS file=$FILE class=$CLASS start=$(date -u +%FT%TZ)"
  env | grep -E '^(SGLANG_|FLASHINFER_|CUTE_DSL|CUDA_VISIBLE|PYTHONPATH)' | sort
  if [[ "$CLASS" == all ]]; then
    python3 -u "$FILE"
  else
    python3 -u "$FILE" "$CLASS"
  fi
  echo "exit=$? end=$(date -u +%FT%TZ)"
} > "$LOG_DIR/test.log" 2>&1
tail -n 3 "$LOG_DIR/test.log"
