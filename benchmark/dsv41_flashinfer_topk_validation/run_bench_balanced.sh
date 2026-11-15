#!/usr/bin/env bash
# Four runs of every variant on one GPU, one process per variant, after one discarded
# warm-up process. The run orders form a Latin square (each variant once in every
# slot) and are two opposite-order pairs, so slot and linear drift effects cancel.
#   bash run_bench_balanced.sh <result dir> [extra bench_topk.py args...]
set -euo pipefail

OUT=${1:?result dir}
shift
ROOT=/hai-workspace/dsv41-fi-topk
mkdir -p "$OUT" "$OUT/warmup"
export CUDA_VISIBLE_DEVICES=${BENCH_GPU:-0}
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2

run_variant() {
  local out=$1 variant=$2 tree=sglang
  shift 2
  [[ $variant == sgl-kernel-base ]] && tree=sglang-base
  local -a env_args=(PYTHONPATH="$ROOT/$tree/python" SGLANG_CACHE_DIR="$ROOT/cache/$tree"
    FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer")
  [[ $variant == flashinfer-small ]] && env_args+=(SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small)
  echo "== $(basename "$out") $variant ($tree) $(date -u +%FT%TZ)"
  env "${env_args[@]}" python3 "$ROOT/tools/bench_topk.py" --variant "$variant" \
    --tree "$ROOT/$tree" --out "$out" "$@"
}

run_variant "$OUT/warmup/run0-sgl-kernel-base.jsonl" sgl-kernel-base "$@"
ORDERS=(
  "sgl-kernel-base sgl-kernel flashinfer flashinfer-small"
  "flashinfer-small flashinfer sgl-kernel sgl-kernel-base"
  "sgl-kernel sgl-kernel-base flashinfer-small flashinfer"
  "flashinfer flashinfer-small sgl-kernel-base sgl-kernel"
)
for run in 1 2 3 4; do
  for variant in ${ORDERS[$((run - 1))]}; do
    run_variant "$OUT/run$run-$variant.jsonl" "$variant" "$@"
  done
done
echo "== done $(date -u +%FT%TZ)"
