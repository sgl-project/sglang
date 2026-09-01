#!/usr/bin/env bash
# Baseline: single TP=1 server on the whole GPU, benched at the same
# loads as the DP-like case for apples-to-apples comparison.
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export HF_HUB_CACHE=/cluster-storage/models/hub
export CUDA_VISIBLE_DEVICES=0
LOG_DIR=/root/mig-demo

pkill -f sglang.launch_server 2>/dev/null
sleep 3

nohup python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30000 --tp 1 \
    > "$LOG_DIR/single.log" 2>&1 &

ok=0
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1 && { ok=1; break; }
    sleep 5
done
[ "$ok" = 1 ] || { echo "server failed"; tail -5 "$LOG_DIR/single.log"; exit 1; }

grep -h max_total_num_tokens "$LOG_DIR/single.log"
python "$LOG_DIR/verify_and_bench.py" --port 30000 --label single-c32 --n 64 --max-new-tokens 128 --concurrency 32
python "$LOG_DIR/verify_and_bench.py" --port 30000 --label single-c16 --n 32 --max-new-tokens 128 --concurrency 16
python "$LOG_DIR/verify_and_bench.py" --port 30000 --label single-c8 --n 16 --max-new-tokens 128 --concurrency 8
python "$LOG_DIR/verify_and_bench.py" --port 30000 --label single-c1 --n 4 --max-new-tokens 64

pkill -f sglang.launch_server 2>/dev/null
true
