#!/bin/bash
# triton attention backend + eager mode (no cuda graph).
# Smoke for prefill + decode path without any graph capture.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="triton_eager"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"

LAUNCH_ARGS=(
    --attention-backend triton
    --disable-cuda-graph
    --mem-fraction-static 0.7
    --max-running-requests 8
    --tp-size 1
)

run_server_smoke
