#!/bin/bash
# Profile ONLY inference (skip warmup) for FP8+DeepGemm on 256x256.
#
# Strategy: run sglang generate twice.
#   Run 1: no nsys, measure warmup+inference wall time → extract warmup duration
#   Run 2: nsys with --delay=(warmup duration + margin), capture inference only
#
# Usage:
#   bash zimage_256_256/profile_inference_only.sh

set -euo pipefail

export MODEL=/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export PROMPT="A beautiful sunset over the ocean with golden clouds"
export IMAGE_SIZE=256

NSYS_OUT_DIR=/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/zimage_bench/nsys
NSYS_OUT="$NSYS_OUT_DIR/fp8_inference_only_256"
mkdir -p "$NSYS_OUT_DIR/csv"

# ── Run 1: Dry run to measure warmup duration ────────────────────────────
echo "============================================================"
echo "Run 1: Dry run to measure warmup duration"
echo "============================================================"

# sglang generate with --warmup outputs a log line like:
#   "Warmed-up request processed in X.XX seconds (with warmup excluded)"
# The total wall time minus X.XX = warmup duration.

START_SEC=$(date +%s)

CUDA_VISIBLE_DEVICES=0 sglang generate \
  --model-path "$MODEL" \
  --transformer-path "$MODEL/transformer-FP8-block128" \
  --prompt "$PROMPT" \
  --height "$IMAGE_SIZE" --width "$IMAGE_SIZE" \
  --warmup \
  --text-encoder-precisions bf16 \
  --seed 42 2>&1 | tee /tmp/sglang_dryrun.log

END_SEC=$(date +%s)
TOTAL_SEC=$((END_SEC - START_SEC))

# Extract inference time from log
INFERENCE_SEC=$(grep -oP 'processed in \K[0-9.]+' /tmp/sglang_dryrun.log | tail -1)
if [ -z "$INFERENCE_SEC" ]; then
    echo "WARNING: Could not extract inference time from log. Using default."
    INFERENCE_SEC=2
fi

# Warmup duration = total - inference (with some margin)
WARMUP_SEC=$(python3 -c "import math; print(math.ceil($TOTAL_SEC - $INFERENCE_SEC))")
# Add 5s margin to be safe
DELAY_SEC=$((WARMUP_SEC + 5))
# Capture duration = inference time * 3 (generous)
DURATION_SEC=$(python3 -c "import math; print(max(30, math.ceil($INFERENCE_SEC * 3)))")

echo ""
echo "  Total wall time:  ${TOTAL_SEC}s"
echo "  Inference time:   ${INFERENCE_SEC}s"
echo "  Warmup duration:  ~${WARMUP_SEC}s"
echo "  nsys --delay:     ${DELAY_SEC}s"
echo "  nsys --duration:  ${DURATION_SEC}s"

# ── Run 2: nsys profile with calculated delay ─────────────────────────────
echo ""
echo "============================================================"
echo "Run 2: nsys profile (delay=${DELAY_SEC}s, duration=${DURATION_SEC}s)"
echo "  Captures only inference, skips warmup"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 \
  nsys profile \
    --trace=cuda \
    --cuda-memory-usage=false \
    --force-overwrite=true \
    --delay="$DELAY_SEC" \
    --duration="$DURATION_SEC" \
    -o "$NSYS_OUT" \
  sglang generate \
    --model-path "$MODEL" \
    --transformer-path "$MODEL/transformer-FP8-block128" \
    --prompt "$PROMPT" \
    --height "$IMAGE_SIZE" --width "$IMAGE_SIZE" \
    --warmup \
    --text-encoder-precisions bf16 \
    --seed 42

# ── Analysis ──────────────────────────────────────────────────────────────
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
    echo "    The 139.9ms is NOT just warmup — transpose happens at runtime too."
else
    echo ""
    echo "✅ NO transpose kernels during inference"
    echo "   All 139.9ms was from warmup. No code change needed."
fi

echo ""
echo "── Top 15 kernels by time ──"
head -16 "$CSV"

echo ""
echo "Full CSV: $CSV"
