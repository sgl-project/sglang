#!/bin/bash
# ============================================================
# 导出 nsys kernel 统计为 CSV，用于离线分析
# 在 GPU 服务器上运行：bash export_nsys_stats.sh
# ============================================================

NSYS_DIR="/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/zimage_bench/nsys"
OUT_DIR="/mnt/geminihzceph/rhyshen/profiles/zimage_256_256/debug"
mkdir -p "$OUT_DIR"

# BF16 baseline
echo "=== Exporting BF16 baseline ==="
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUT_DIR/nsys_bf16_kernels" \
    "${NSYS_DIR}/zimage_1gpu_256x256_te16.nsys-rep"

# FP8 fixed
echo "=== Exporting FP8 fixed ==="
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUT_DIR/nsys_fp8_kernels" \
    "${NSYS_DIR}/zimage_1gpu_256x256_fp8.nsys-rep"

echo ""
echo "=== Output files ==="
ls -la "$OUT_DIR"/nsys_*.csv 2>/dev/null || ls -la "$OUT_DIR"/nsys_*_cuda_gpu_kern_sum.csv 2>/dev/null

echo ""
echo "Done. Copy results to local for analysis."
