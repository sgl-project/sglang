#!/bin/bash
# triton attention + EAGLE speculative decoding + CUDA graph.
# Llama-2-7b target + sglang-EAGLE draft (both cached on cluster).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="triton_eagle"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-2-7b-chat-hf}"

LAUNCH_ARGS=(
    --attention-backend triton
    --speculative-algorithm EAGLE
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B
    --speculative-num-steps 3
    --speculative-eagle-topk 4
    --speculative-num-draft-tokens 8
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

# Dummy load means the EAGLE draft head shares architecture with target.
# EAGLE2 needs spec v1 (default), so we leave SGLANG_ENABLE_SPEC_V2 unset.
run_server_smoke
