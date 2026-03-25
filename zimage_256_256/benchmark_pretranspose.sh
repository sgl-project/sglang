#!/bin/bash
# Benchmark script for DeepGemm Pre-Transpose Bs optimization
# Run on GPU server after applying the code changes
#
# Compares:
# 1. BF16 baseline (text encoder bf16)
# 2. FP8+DeepGemm with pre-transpose (the new optimization)
#
# Usage:
#   bash zimage_256_256/benchmark_pretranspose.sh

set -euo pipefail

export MODEL=/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PROMPT="A beautiful sunset over the ocean with golden clouds"
export IMAGE_SIZE=256

export BENCH_OUT_DIR=/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/zimage_bench/pretranspose_${IMAGE_SIZE}_${IMAGE_SIZE}
export OUT_DIR=/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/outputs/pretranspose_${IMAGE_SIZE}_${IMAGE_SIZE}

mkdir -p "$BENCH_OUT_DIR"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Step 1: BF16 baseline (text encoder bf16)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 sglang generate \
  --model-path "$MODEL" \
  --prompt "$PROMPT" \
  --height "$IMAGE_SIZE" --width "$IMAGE_SIZE" \
  --warmup \
  --text-encoder-precisions bf16 \
  --perf-dump-path "$BENCH_OUT_DIR/bf16_baseline.json" \
  --save-output \
  --output-path "$OUT_DIR" \
  --output-file-name "bf16_baseline_${IMAGE_SIZE}x${IMAGE_SIZE}.png" \
  --seed 42

echo ""
echo "============================================================"
echo "Step 2: FP8+DeepGemm with pre-transpose Bs"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 SGLANG_ENABLE_JIT_DEEPGEMM=1 sglang generate \
  --model-path "$MODEL" \
  --transformer-path "$MODEL/transformer-FP8-block128" \
  --prompt "$PROMPT" \
  --height "$IMAGE_SIZE" --width "$IMAGE_SIZE" \
  --warmup \
  --text-encoder-precisions bf16 \
  --perf-dump-path "$BENCH_OUT_DIR/fp8_deepgemm_pretranspose.json" \
  --save-output \
  --output-path "$OUT_DIR" \
  --output-file-name "fp8_deepgemm_pretranspose_${IMAGE_SIZE}x${IMAGE_SIZE}.png" \
  --seed 42

echo ""
echo "============================================================"
echo "Step 3: nsys profile for FP8+DeepGemm (verify transpose eliminated)"
echo "============================================================"
NSYS_OUT="$BENCH_OUT_DIR/nsys_fp8_pretranspose"
CUDA_VISIBLE_DEVICES=0 SGLANG_ENABLE_JIT_DEEPGEMM=1 nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "$NSYS_OUT" \
  sglang generate \
    --model-path "$MODEL" \
    --transformer-path "$MODEL/transformer-FP8-block128" \
    --prompt "$PROMPT" \
    --height "$IMAGE_SIZE" --width "$IMAGE_SIZE" \
    --warmup \
    --text-encoder-precisions bf16 \
    --seed 42

echo ""
echo "============================================================"
echo "Results Summary"
echo "============================================================"
echo "BF16 baseline:     $BENCH_OUT_DIR/bf16_baseline.json"
echo "FP8 pre-transpose: $BENCH_OUT_DIR/fp8_deepgemm_pretranspose.json"
echo "nsys profile:      ${NSYS_OUT}.nsys-rep"
echo ""
echo "To analyze nsys profile:"
echo "  nsys stats ${NSYS_OUT}.nsys-rep"
echo "  # Look for 'transpose_fp32' kernels — should be eliminated for weight scales"
