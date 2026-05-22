#!/bin/bash
# DSV4-Flash + EAGLE speculative decoding + CUDA graph.
# Motivating real workload: this is what step03 must keep working.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="dsv4_eagle"
MODEL_PATH="${MODEL_PATH:-/flash_model}"

LAUNCH_ARGS=(
    --attention-backend dsv4
    --moe-runner-backend flashinfer_mxfp4
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --tp-size 4
    --mem-fraction-static 0.8
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --chunked-prefill-size 4096
    --disable-flashinfer-autotune
    --disable-piecewise-cuda-graph
)

# DSV4 EAGLE auto-enables spec v2 in the hook.
READY_TIMEOUT=2400
run_server_smoke
