#!/bin/bash
# One-time environment setup for the MiniMax-M3 AgentX reproduction (ROCm 7.2.4 container, gfx950).
# Usage: bash setup_env.sh   (idempotent; set SGLANG_DIR/AITER_DIR/MODELS_DIR/AIPERF_DIR to relocate)
set -euo pipefail
: "${SGLANG_DIR:=/sgl-workspace/sglang}" "${AITER_DIR:=/sgl-workspace/aiter}" "${MODELS_DIR:=/scratch/models}"
: "${AIPERF_DIR:=/scratch/aiperf-sa}" "${AIPERF_VENV:=/scratch/aiperf-sa-venv}"
HERE=$(cd "$(dirname "$0")" && pwd)

[ -d "$SGLANG_DIR/.git" ] || git clone -b M3-perf https://github.com/kevin-mii/sglang "$SGLANG_DIR"
pip install -q -e "$SGLANG_DIR/python"

if [ ! -d "$AITER_DIR/.git" ]; then
  git clone https://github.com/ROCm/aiter "$AITER_DIR" && git -C "$AITER_DIR" checkout 4ad99832
fi
# FlyDSL XCD-swizzle fix (workgroup remap was not a bijection when the count is not a multiple of 8)
if [ -f "$HERE/aiter_flydsl_xcd_swizzle_fix.patch" ] && ! git -C "$AITER_DIR" apply --check -R "$HERE/aiter_flydsl_xcd_swizzle_fix.patch" 2>/dev/null; then
  git -C "$AITER_DIR" apply "$HERE/aiter_flydsl_xcd_swizzle_fix.patch"
fi
pip install -q -e "$AITER_DIR"

# tuned fp4 fused-MoE rows for the M3 shape (aiter reads /tmp/aiter_configs/tuned_fmoe.csv or AITER_CONFIG_FMOE)
mkdir -p /tmp/aiter_configs && cp "$HERE/tuned_fmoe_m3_gfx950.csv" /tmp/aiter_configs/tuned_fmoe.csv

mkdir -p "$MODELS_DIR"
[ -f "$MODELS_DIR/MiniMax-M3-MXFP4/config.json" ] || huggingface-cli download amd/MiniMax-M3-MXFP4 --local-dir "$MODELS_DIR/MiniMax-M3-MXFP4"
[ -f "$MODELS_DIR/MiniMax-M3-EAGLE3-GQA/config.json" ] || huggingface-cli download Inferact/MiniMax-M3-EAGLE3-GQA --local-dir "$MODELS_DIR/MiniMax-M3-EAGLE3-GQA"

# SemiAnalysis AIPerf fork (stock aiperf 0.12.0 rejects --trace-idle-gap-cap-seconds)
if [ ! -x "$AIPERF_VENV/bin/aiperf" ]; then
  python3.12 -m venv "$AIPERF_VENV"
  [ -d "$AIPERF_DIR/.git" ] || git clone https://github.com/SemiAnalysisAI/aiperf "$AIPERF_DIR"
  git -C "$AIPERF_DIR" checkout b7b16cf8
  "$AIPERF_VENV/bin/pip" install -q -e "$AIPERF_DIR"
fi
echo "setup complete"
