#!/bin/bash
# dsv4 attention backend + CUDA graph (DeepSeek-V4-Flash model on B200).
# Requires PR #26024 routing fix (be24b5c4b1) to be present so the
# FP4-packed experts dispatch to DEEP_GEMM. Cherry-picked into the test
# branch — see test/registered/step03_coverage/README.md.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="dsv4_cudagraph"
# DSV4-Flash is the on-cluster checkpoint (yangminl mount via /flash_model).
# Caller mounts /mnt/vast/yangminl/models/DeepSeek-V4-Flash:/flash_model
# or relies on the HF download (slow, large).
MODEL_PATH="${MODEL_PATH:-/flash_model}"

LAUNCH_ARGS=(
    --attention-backend dsv4
    --moe-runner-backend flashinfer_mxfp4
    --tp-size 4
    --mem-fraction-static 0.8
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --chunked-prefill-size 4096
    --disable-flashinfer-autotune
    --disable-piecewise-cuda-graph
)

READY_TIMEOUT=2400
run_server_smoke
