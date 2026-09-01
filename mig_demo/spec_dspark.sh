#!/usr/bin/env bash
# Case 5: Qwen/Qwen3.8-27B + RadixArk/Qwen3.8-27B-DSpark speculative decoding.
# DSPARK runs the draft IN-PROCESS (DSparkWorkerV2); draft-in-a-separate-process
# ("decoupled spec") is scaffolding-only in sglang 0.5.18 and does not run.
# Phase 1a: 27B + DSPARK alone (accept len + throughput)
# Phase 1b: 27B alone, no spec (baseline for speedup)
# Phase 2:  co-located (27B+DSPARK) + Qwen3-0.6B tenant, concurrent load
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export HF_HUB_CACHE=/cluster-storage/models/hub
export CUDA_VISIBLE_DEVICES=0
LOG_DIR=/root/mig-demo
BENCH="python $LOG_DIR/verify_and_bench.py"
MODEL=Qwen/Qwen3.8-27B
SPEC_FLAGS="--speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark"

start_server() { # port frac logfile extra-args...
    local port=$1 frac=$2 log=$3; shift 3
    nohup python -m sglang.launch_server --host 127.0.0.1 --port $port \
        --mem-fraction-static $frac "$@" > "$log" 2>&1 &
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1 && return 0
        sleep 5
    done
    echo "server on $port FAILED"; tail -12 "$log"; exit 1
}

accept_len() { grep -o "accept len: [0-9.]*" "$1" | tail -3; }

pkill -f sglang.launch_server 2>/dev/null; sleep 3

echo "=== Phase 1a: Qwen3.8-27B + DSPARK, alone ==="
start_server 30001 0.85 "$LOG_DIR/dspark_on.log" --model-path $MODEL $SPEC_FLAGS
$BENCH --port 30001 --label dspark-on-c1 --n 4 --max-new-tokens 128
$BENCH --port 30001 --label dspark-on-c8 --n 16 --max-new-tokens 128 --concurrency 8
accept_len "$LOG_DIR/dspark_on.log"
pkill -f sglang.launch_server 2>/dev/null; sleep 5

echo "=== Phase 1b: Qwen3.8-27B no spec, alone ==="
start_server 30001 0.85 "$LOG_DIR/dspark_off.log" --model-path $MODEL
$BENCH --port 30001 --label dspark-off-c1 --n 4 --max-new-tokens 128
$BENCH --port 30001 --label dspark-off-c8 --n 16 --max-new-tokens 128 --concurrency 8
pkill -f sglang.launch_server 2>/dev/null; sleep 5

echo "=== Phase 2: co-located (27B+DSPARK) + Qwen3-0.6B ==="
# Sequential startup: A allocates first, B gets fraction x total - used_A.
start_server 30000 0.65 "$LOG_DIR/coloc_dspark.log" --model-path $MODEL $SPEC_FLAGS
grep -h max_total_num_tokens "$LOG_DIR/coloc_dspark.log"
start_server 30001 0.73 "$LOG_DIR/coloc_06.log" --model-path Qwen/Qwen3-0.6B
grep -h max_total_num_tokens "$LOG_DIR/coloc_06.log"
$BENCH --port 30000 --label coloc-dspark27B-c8 --n 16 --max-new-tokens 128 --concurrency 8 &
PA=$!
$BENCH --port 30001 --label coloc-06-c8 --n 16 --max-new-tokens 128 --concurrency 8 &
PB=$!
wait $PA $PB
accept_len "$LOG_DIR/coloc_dspark.log"

pkill -f sglang.launch_server 2>/dev/null
echo "=== CASE5-DONE ==="
true
