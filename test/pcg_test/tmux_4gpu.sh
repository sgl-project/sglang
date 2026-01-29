#!/bin/bash
# Launch PCG 4-GPU stress test in a detached tmux session.
# Only generic 4-GPU suites (no H200/B200 hardware requirement).
#
# Usage:
#   bash test/pcg_test/tmux_4gpu.sh                     # start in background
#   bash test/pcg_test/tmux_4gpu.sh --reset              # fresh start
#   tmux attach -t pcg_4gpu                              # attach to see output
#   tmux kill-session -t pcg_4gpu                        # stop it

set -uo pipefail

SESSION="pcg_4gpu"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES="1,2,4,7"
export HF_HOME="/home/tensormesh/yuwei/huggingface"
export HF_HUB_CACHE="/home/tensormesh/yuwei/huggingface/hub4"
export HUGGINGFACE_HUB_CACHE="/home/tensormesh/yuwei/huggingface/hub4"
mkdir -p "$HF_HUB_CACHE"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" \
    "source /home/tensormesh/yuwei/sgl/.venv/bin/activate && export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' HF_TOKEN='$HF_TOKEN' HF_HOME='$HF_HOME' HF_HUB_CACHE='$HF_HUB_CACHE' HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'; python3 $SCRIPT_DIR/run_4gpu.py $*; echo 'Done. Press enter to exit.'; read"

echo "PCG 4-GPU stress test started in tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  status:  python3 $SCRIPT_DIR/run_4gpu.py --status"
echo "  stop:    tmux kill-session -t $SESSION"
