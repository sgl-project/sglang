#!/usr/bin/env bash
# Record the sgl-kernel launches, outputs and top-k validity of the base tree twice (A/A)
# and the branch tree once on GPU 0, then compare.
set -uo pipefail
ROOT=/hai-workspace/dsv41-fi-topk
cd "$ROOT"
mkdir -p trace
unset FLASHINFER_TOPK_ALGO SGLANG_DSA_FUSE_TOPK SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC \
  SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK SGLANG_TOPK_V2_MODE SGLANG_OPT_USE_TOPK_V2
for step in "sgl-kernel-base sglang-base sgl-kernel-base" "sgl-kernel sglang sgl-kernel" \
  "sgl-kernel-base sglang-base sgl-kernel-base-again"; do
  read -r variant tree out <<< "$step"
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/$tree/python" SGLANG_CACHE_DIR="$ROOT/cache/$tree" \
    FLASHINFER_WORKSPACE_BASE="$ROOT/cache/$tree/flashinfer" \
    python3 tools/trace_sgl_kernel.py record --variant "$variant" --tree "$ROOT/$tree" \
    --out "trace/$out.jsonl" > "trace/$out.log" 2>&1
  status=$?
  echo "$(date -u +%FT%TZ) $out exit=$status"
done
python3 tools/trace_sgl_kernel.py compare trace/sgl-kernel-base.jsonl trace/sgl-kernel.jsonl \
  --repeat trace/sgl-kernel-base-again.jsonl > trace/compare.md 2>&1
echo "$(date -u +%FT%TZ) trace finished"
