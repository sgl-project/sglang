#!/bin/bash
# One-time environment setup for the MiniMax-M3 AgentX reproduction (ROCm 7.2.4 container, gfx950). Idempotent.
# Usage: bash setup_env.sh          [M3_WORK=/scratch] [SKIP_MODELS=1] [DRY_RUN=1]
set -euo pipefail
source "$(dirname "$0")/env.sh"
run() { echo "+ $*"; [ -n "${DRY_RUN:-}" ] || "$@"; }
echo "sglang=$SGLANG_DIR aiter=$AITER_DIR models=$MODELS_DIR aiperf=$AIPERF_DIR venv=$AIPERF_VENV work=$M3_WORK"
command -v rocm-smi >/dev/null || echo "WARNING: rocm-smi not found; is this a ROCm node?"
"$PYTHON" -c "import torch; assert torch.version.hip, 'torch without ROCm'; print('torch', torch.__version__)" || { echo "need a ROCm torch in $PYTHON"; exit 1; }

run "$PYTHON" -m pip install -q -e "$SGLANG_DIR/python"

if [ ! -d "$AITER_DIR/.git" ]; then
  run git clone https://github.com/ROCm/aiter "$AITER_DIR"
  run git -C "$AITER_DIR" checkout 4ad99832
  run git -C "$AITER_DIR" submodule update --init --recursive
fi
# FlyDSL XCD-swizzle fix (workgroup remap was not a bijection when the count is not a multiple of 8)
if ! git -C "$AITER_DIR" apply --check -R "$M3_ROOT/aiter_flydsl_xcd_swizzle_fix.patch" 2>/dev/null; then
  run git -C "$AITER_DIR" apply "$M3_ROOT/aiter_flydsl_xcd_swizzle_fix.patch"
fi
run "$PYTHON" -m pip install -q -e "$AITER_DIR"

# tuned fp4 fused-MoE rows for the M3 shape; aiter merges its config files into /tmp/aiter_configs (path fixed in aiter)
run mkdir -p /tmp/aiter_configs
run cp "$M3_ROOT/tuned_fmoe_m3_gfx950.csv" /tmp/aiter_configs/tuned_fmoe.csv

if [ -z "${SKIP_MODELS:-}" ]; then
  run mkdir -p "$MODELS_DIR"
  dl() { [ -f "$2/config.json" ] || run "$PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download('$1', local_dir='$2')"; }
  dl amd/MiniMax-M3-MXFP4 "$MODEL"
  dl Inferact/MiniMax-M3-EAGLE3-GQA "$DRAFT_MODEL"
fi

# SemiAnalysis AIPerf fork at InferenceX's pinned commit (stock aiperf rejects --trace-idle-gap-cap-seconds)
if [ ! -x "$AIPERF_BIN" ]; then
  run "$PYTHON" -m venv "$AIPERF_VENV"
  [ -d "$AIPERF_DIR/.git" ] || run git clone https://github.com/SemiAnalysisAI/aiperf "$AIPERF_DIR"
  run git -C "$AIPERF_DIR" checkout 754356e9
  run "$AIPERF_VENV/bin/pip" install -q -e "$AIPERF_DIR"
fi
echo "setup complete (work dir $M3_WORK)"
