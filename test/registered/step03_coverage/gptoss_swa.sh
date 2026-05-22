#!/bin/bash
# gpt-oss-20b exercises sliding-window attention (SWA).
# Backend defaults to triton/flashinfer; SWA path stresses the
# attention-init code that step03 unifies.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="gptoss_swa"
MODEL_PATH="${MODEL_PATH:-openai/gpt-oss-20b}"

LAUNCH_ARGS=(
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

READY_TIMEOUT=1800
run_server_smoke
