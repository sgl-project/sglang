#!/bin/bash
# =============================================================================
# FP8 Debug & Verification Script
# =============================================================================
# 在 GPU 集群上按顺序执行以下步骤：
#
# Step 1: 诊断 scale 加载状态（修复前 or 修复后）
# Step 2: 运行修复后的 FP8 基准测试
# Step 3: 图像质量验证
# Step 4: nsys GPU kernel 分析
# =============================================================================

set -e

MODEL="/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo"
FP8_DIR="$MODEL/transformer-FP8-block128"
FP8_NOFFN_DIR="$MODEL/transformer-FP8-block128-no-ffn"
OUT_DIR="/data/home/rhyshen/rhyshen/workspace/sglang/zimage_256_256"
PROMPT="A beautiful sunset over the ocean with golden clouds"

echo "============================================================"
echo "Step 1: Diagnostic — Check FP8 Scale Loading"
echo "============================================================"
echo ""
echo "Run: python $OUT_DIR/debug_fp8_scales.py"
echo ""
echo "Expected:"
echo "  - Part A: checkpoint should have w1.weight_scale_inv and w3.weight_scale_inv"
echo "  - Part C: BEFORE fix -> scales NOT MAPPED; AFTER fix -> scales MAPPED"
echo "  - Part B: BEFORE fix -> DEFAULT_INIT=YES; AFTER fix -> status=OK"
echo ""

# Uncomment to run:
# python $OUT_DIR/debug_fp8_scales.py

echo "============================================================"
echo "Step 2: FP8 Benchmark (after fix)"
echo "============================================================"
echo ""
echo "Run FP8+FFN benchmark:"
echo "  sglang generate \\"
echo "    --model-path $MODEL \\"
echo "    --transformer-path $FP8_DIR \\"
echo "    --text-encoder-precisions bf16 \\"
echo "    --prompt \"$PROMPT\" \\"
echo "    --height 256 --width 256 --warmup --save-output \\"
echo "    --perf-dump-path $OUT_DIR/zimage_bench/baseline_1gpu_bf16te_fp8dit_fixed.json"
echo ""
echo "Run FP8-noFFN benchmark:"
echo "  sglang generate \\"
echo "    --model-path $MODEL \\"
echo "    --transformer-path $FP8_NOFFN_DIR \\"
echo "    --text-encoder-precisions bf16 \\"
echo "    --prompt \"$PROMPT\" \\"
echo "    --height 256 --width 256 --warmup --save-output \\"
echo "    --perf-dump-path $OUT_DIR/zimage_bench/baseline_1gpu_bf16te_fp8dit_noffn_fixed.json"
echo ""

echo "============================================================"
echo "Step 3: Image Quality Verification (fixed seed)"
echo "============================================================"
echo ""
echo "FP8+FFN quality test (seed=42):"
echo "  sglang generate \\"
echo "    --model-path $MODEL \\"
echo "    --transformer-path $FP8_DIR \\"
echo "    --text-encoder-precisions bf16 \\"
echo "    --prompt \"$PROMPT\" \\"
echo "    --height 256 --width 256 --seed 42 --save-output \\"
echo "    --output-path $OUT_DIR/outputs/fp8_fixed_seed42.png"
echo ""
echo "BF16 baseline quality test (seed=42):"
echo "  sglang generate \\"
echo "    --model-path $MODEL \\"
echo "    --text-encoder-precisions bf16 \\"
echo "    --prompt \"$PROMPT\" \\"
echo "    --height 256 --width 256 --seed 42 --save-output \\"
echo "    --output-path $OUT_DIR/outputs/bf16_baseline_seed42.png"
echo ""
echo "Compare the two images visually."
echo ""

echo "============================================================"
echo "Step 4: nsys GPU Kernel Analysis"
echo "============================================================"
echo ""
echo "BF16 baseline nsys:"
echo "  nsys profile -o $OUT_DIR/logs/nsys_bf16_baseline \\"
echo "    --trace=cuda,nvtx --force-overwrite true \\"
echo "    sglang generate \\"
echo "      --model-path $MODEL \\"
echo "      --text-encoder-precisions bf16 \\"
echo "      --prompt \"$PROMPT\" \\"
echo "      --height 256 --width 256 --warmup"
echo ""
echo "FP8 fixed nsys:"
echo "  nsys profile -o $OUT_DIR/logs/nsys_fp8_fixed \\"
echo "    --trace=cuda,nvtx --force-overwrite true \\"
echo "    sglang generate \\"
echo "      --model-path $MODEL \\"
echo "      --transformer-path $FP8_DIR \\"
echo "      --text-encoder-precisions bf16 \\"
echo "      --prompt \"$PROMPT\" \\"
echo "      --height 256 --width 256 --warmup"
echo ""

echo "============================================================"
echo "Success Criteria"
echo "============================================================"
echo "  [ ] FP8 E2E < 420ms (target: 375-415ms)"
echo "  [ ] FP8+FFN images are correct (not noise/corruption)"
echo "  [ ] DeepGemm replaces nvjet GEMM (not adds on top)"
echo "  [ ] No scale parameters show DEFAULT_INIT in diagnostic"
echo "============================================================"
