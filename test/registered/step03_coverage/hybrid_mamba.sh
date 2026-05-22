#!/bin/bash
# Hybrid mamba/SSM + transformer attention (NemotronH).
# Default linear-attn backend (mamba) + default MHA backend for attention layers.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="hybrid_mamba"
# NemotronH is a hybrid mamba-attention model; small variant downloadable.
MODEL_PATH="${MODEL_PATH:-nvidia/Nemotron-H-8B-Base-8K}"

LAUNCH_ARGS=(
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

READY_TIMEOUT=1800
run_server_smoke
