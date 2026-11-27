#!/usr/bin/env bash
# Populate a fresh devbox's JIT caches before a benchmark: one throwaway pass of every
# variant over every shape (results discarded), so no kernel compiles inside a timed run.
#   bash prewarm_jit.sh <scratch dir>
set -euo pipefail
SCRATCH=${1:?scratch dir}
ROOT=/hai-workspace/dsv41-fi-topk
mkdir -p "$SCRATCH"
export CUDA_VISIBLE_DEVICES=${BENCH_GPU:-0}
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
for variant in sgl-kernel-base sgl-kernel sgl-kernel-mode2 flashinfer flashinfer-small flashinfer-small-det; do
  tree=sglang
  [[ $variant == sgl-kernel-base ]] && tree=sglang-base
  env_args=(PYTHONPATH="$ROOT/$tree/python" SGLANG_CACHE_DIR="$ROOT/cache/$tree"
    FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer")
  case $variant in
    sgl-kernel-mode2) env_args+=(SGLANG_TOPK_V2_MODE=2) ;;
    flashinfer-small) env_args+=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small) ;;
    flashinfer-small-det)
      env_args+=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1) ;;
  esac
  echo "== prewarm $variant $(date -u +%FT%TZ)"
  env "${env_args[@]}" python3 "$ROOT/tools/bench_topk.py" --variant "$variant" --tree "$ROOT/$tree" \
    --rounds 1 --out "$SCRATCH/prewarm-$variant.jsonl" > "$SCRATCH/prewarm-$variant.log" 2>&1
done
echo "== prewarm done $(date -u +%FT%TZ)"
