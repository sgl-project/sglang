#!/usr/bin/env bash
# Post-sharegpt bench_eval sweep across {hellaswag, winogrande, gsm8k,
# mmlu, gpqa}. Uses only the 6 hot% dispatch variants (no thr* variants
# — threshold variants are most useful when you already know the policy
# curve and want to fine-tune it; the hot% sweep gives a 0-100% BF16
# coverage curve per task that's more diagnostic).
#
# Waits for any running `bash run_sweep.sh sharegpt` to finish before
# starting. Each task's failures (e.g. unknown lm_eval task name,
# gated dataset) are absorbed by run_sweep.sh's per-cell `|| true`, so
# one bad task doesn't block the others.
#
# Usage:
#   bash run_bench_eval_all.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

echo "[$(date '+%F %T')] Waiting for any in-flight run_sweep.sh sharegpt..."
# Bracket trick avoids pgrep self-match: the pattern '[r]un_sweep.sh sharegpt'
# is not equal to the literal string 'run_sweep.sh sharegpt' that appears in
# pgrep's own argv, so pgrep won't match its own invocation.
while pgrep -f '[r]un_sweep.sh sharegpt' > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date '+%F %T')] Sharegpt sweep done. Starting bench_eval pass."

TASKS=(hellaswag winogrande gsm8k mmlu gpqa)
export VARIANTS="hot0 hot20 hot40 hot60 hot80 hot100"

for task in "${TASKS[@]}"; do
    echo "============================================================"
    echo "  [$(date '+%F %T')] Task: $task  (6 variants × 6 mc = 36 cells)"
    echo "============================================================"
    bash run_sweep.sh "$task" || true
done

echo "[$(date '+%F %T')] All bench_eval tasks complete."
