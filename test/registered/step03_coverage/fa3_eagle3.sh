#!/bin/bash
# fa3 + EAGLE3 (multi-layer draft) + CUDA graph.
# Uses Llama-3.1-8B target + sglang EAGLE3 draft.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
step03_preamble

TEST_NAME="fa3_eagle3"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

LAUNCH_ARGS=(
    --attention-backend fa3
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B
    --speculative-num-steps 5
    --speculative-eagle-topk 4
    --speculative-num-draft-tokens 8
    --mem-fraction-static 0.7
    --cuda-graph-max-bs 4
    --max-running-requests 4
    --disable-piecewise-cuda-graph
    --tp-size 1
)

run_server_smoke
