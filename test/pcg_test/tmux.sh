#!/bin/bash
# Launch PCG stress test in a detached tmux session.
#
# Usage:
#   bash test/pcg_test/tmux.sh                          # start in background
#   bash test/pcg_test/tmux.sh --reset                  # fresh start
#   tmux attach -t pcg_test                             # attach to see output
#   tmux kill-session -t pcg_test                       # stop it

SESSION="pcg_test"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" "source /home/tensormesh/yuwei/sgl/.venv/bin/activate && bash $SCRIPT_DIR/run.sh $*; echo 'Done. Press enter to exit.'; read"

echo "PCG stress test started in tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  status:  bash $SCRIPT_DIR/run.sh --status"
echo "  stop:    tmux kill-session -t $SESSION"
