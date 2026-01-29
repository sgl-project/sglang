#!/bin/bash
# PCG (Piecewise CUDA Graph) stress test runner.
#
# Usage:
#   bash test/pcg_test/run.sh                          # run all, resume if plan exists
#   bash test/pcg_test/run.sh --repeat 5               # 5 runs per test
#   bash test/pcg_test/run.sh --suite stage-a-test-1    # single suite
#   bash test/pcg_test/run.sh --reset                   # fresh start
#   bash test/pcg_test/run.sh --status                  # print progress and exit

set -uo pipefail

export HF_HOME="/home/tensormesh/yuwei/huggingface"
export HF_HUB_CACHE="/home/tensormesh/yuwei/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="/home/tensormesh/yuwei/huggingface/hub"
mkdir -p "$HF_HUB_CACHE"
export CUDA_VISIBLE_DEVICES="3"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "$SCRIPT_DIR/run.py" "$@"
