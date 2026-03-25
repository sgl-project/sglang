#!/bin/bash
# Profile ONLY inference (skip warmup) using cudaProfilerApi.
#
# Uses nsys --capture-range=cudaProfilerApi so profiling is precisely
# controlled by cudaProfilerStart/Stop in the Python script.
# No guessing on --delay.
#
# Usage:
#   bash zimage_256_256/profile_inference_only.sh
#
# After completion, check for transpose kernels:
#   grep -i transpose <output>_cuda_gpu_kern_sum.csv

set -euo pipefail

export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1

NSYS_OUT_DIR=/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/zimage_bench/nsys
NSYS_OUT="$NSYS_OUT_DIR/fp8_inference_only_256"
mkdir -p "$NSYS_OUT_DIR"
mkdir -p "$NSYS_OUT_DIR/csv"

echo "============================================================"
echo "nsys profile with cudaProfilerApi capture range"
echo "  Only GPU activity between cudaProfilerStart/Stop is recorded"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
  nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda \
    --cuda-memory-usage=false \
    --force-overwrite=true \
    -o "$NSYS_OUT" \
  python zimage_256_256/profile_inference_only.py

echo ""
echo "============================================================"
echo "Generating stats..."
echo "============================================================"

nsys stats "$NSYS_OUT.nsys-rep" \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output "$NSYS_OUT_DIR/csv/nsys_fp8_inference_only" \
  --force-overwrite true \
  2>/dev/null

CSV="$NSYS_OUT_DIR/csv/nsys_fp8_inference_only_cuda_gpu_kern_sum.csv"

echo ""
echo "── transpose_fp32 in inference-only profile ──"
if grep -i "transpose" "$CSV"; then
    echo ""
    echo "^^^ transpose kernels FOUND during inference"
    echo "    The 139.9ms is NOT just warmup — sfb transpose happens at runtime too."
else
    echo "✅ NO transpose kernels during inference"
    echo "   All 139.9ms was from warmup. No code change needed for sfb."
fi

echo ""
echo "── Top 15 kernels by time ──"
head -16 "$CSV"

echo ""
echo "Full CSV: $CSV"
