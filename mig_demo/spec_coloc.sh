#!/usr/bin/env bash
# Case 4: co-locate a speculative-decoding server (Qwen3-8B + EAGLE3 draft)
# with a 27B-class model (Qwen3-32B) on one GPU.
# Phase 1: spec-dec speedup (8B alone, spec on vs off)
# Phase 2: 32B alone baseline
# Phase 3: co-located pair under concurrent load
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export HF_HUB_CACHE=/cluster-storage/models/hub
export CUDA_VISIBLE_DEVICES=0
LOG_DIR=/root/mig-demo
BENCH="python $LOG_DIR/verify_and_bench.py"
SPEC_FLAGS="--speculative-algorithm EAGLE3 \
    --speculative-draft-model-path AngelSlim/Qwen3-8B_eagle3 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4"

start_server() { # port frac logfile extra-args...
    local port=$1 frac=$2 log=$3; shift 3
    nohup python -m sglang.launch_server --host 127.0.0.1 --port $port \
        --mem-fraction-static $frac "$@" > "$log" 2>&1 &
    for i in $(seq 1 90); do
        curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1 && return 0
        sleep 5
    done
    echo "server on $port FAILED"; tail -8 "$log"; exit 1
}

accept_len() { grep -o "accept len: [0-9.]*" "$1" | tail -3; }

pkill -f sglang.launch_server 2>/dev/null; sleep 3

echo "=== Phase 1a: Qwen3-8B + EAGLE3 spec-dec, alone ==="
start_server 30001 0.85 "$LOG_DIR/spec_on.log" --model-path Qwen/Qwen3-8B $SPEC_FLAGS
$BENCH --port 30001 --label spec-on-c1 --n 4 --max-new-tokens 128
$BENCH --port 30001 --label spec-on-c8 --n 16 --max-new-tokens 128 --concurrency 8
accept_len "$LOG_DIR/spec_on.log"
pkill -f sglang.launch_server 2>/dev/null; sleep 5

echo "=== Phase 1b: Qwen3-8B no spec, alone ==="
start_server 30001 0.85 "$LOG_DIR/spec_off.log" --model-path Qwen/Qwen3-8B
$BENCH --port 30001 --label spec-off-c1 --n 4 --max-new-tokens 128
$BENCH --port 30001 --label spec-off-c8 --n 16 --max-new-tokens 128 --concurrency 8
pkill -f sglang.launch_server 2>/dev/null; sleep 5

echo "=== Phase 2: Qwen3-32B, alone ==="
start_server 30000 0.85 "$LOG_DIR/alone32.log" --model-path Qwen/Qwen3-32B
$BENCH --port 30000 --label alone32-c1 --n 2 --max-new-tokens 128
$BENCH --port 30000 --label alone32-c8 --n 16 --max-new-tokens 128 --concurrency 8
pkill -f sglang.launch_server 2>/dev/null; sleep 5

echo "=== Phase 3: co-located 32B + (8B+spec) ==="
# Sequential startup: A allocates first, B gets fraction x total - used_A.
start_server 30000 0.55 "$LOG_DIR/coloc32.log" --model-path Qwen/Qwen3-32B
grep -h max_total_num_tokens "$LOG_DIR/coloc32.log"
start_server 30001 0.85 "$LOG_DIR/coloc_spec.log" --model-path Qwen/Qwen3-8B $SPEC_FLAGS
grep -h max_total_num_tokens "$LOG_DIR/coloc_spec.log"
$BENCH --port 30000 --label coloc-32B-c8 --n 16 --max-new-tokens 128 --concurrency 8 &
PA=$!
$BENCH --port 30001 --label coloc-spec8B-c8 --n 16 --max-new-tokens 128 --concurrency 8 &
PB=$!
wait $PA $PB
accept_len "$LOG_DIR/coloc_spec.log"

pkill -f sglang.launch_server 2>/dev/null
echo "=== CASE4-DONE ==="
true
