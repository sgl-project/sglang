#!/usr/bin/env bash
# Run one sglang serving case on the single H200 and bench it.
# Usage: run_case.sh <label> <nnodes> <tp> <bench_n> <bench_tokens> [extra env as K=V ...]
set -uo pipefail

LABEL=$1; NNODES=$2; TP=$3; BN=$4; BT=$5; shift 5
for kv in "$@"; do export "$kv"; done

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/cluster-storage/models/hub

LOG_DIR=/root/mig-demo
pkill -f sglang.launch_server 2>/dev/null
sleep 3

common=(--model-path Qwen/Qwen3-0.6B --host 127.0.0.1
        --disable-custom-all-reduce --mem-fraction-static 0.35)
[ -n "${PP:-}" ] && common+=(--pp-size "$PP")

if [ "$NNODES" = "2" ]; then
    nohup python -m sglang.launch_server "${common[@]}" --nnodes 2 --tp "$TP" \
        --node-rank 0 --port 30000 --dist-init-addr 127.0.0.1:25000 \
        > "$LOG_DIR/case_${LABEL}_node0.log" 2>&1 &
    P0=$!
    nohup python -m sglang.launch_server "${common[@]}" --nnodes 2 --tp "$TP" \
        --node-rank 1 --port 30001 --dist-init-addr 127.0.0.1:25000 \
        > "$LOG_DIR/case_${LABEL}_node1.log" 2>&1 &
else
    nohup python -m sglang.launch_server "${common[@]}" --tp "$TP" --port 30000 \
        > "$LOG_DIR/case_${LABEL}_node0.log" 2>&1 &
    P0=$!
fi

ready=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$P0" 2>/dev/null; then break; fi
    sleep 5
done

if [ "$ready" != "1" ]; then
    echo "{\"label\": \"$LABEL\", \"status\": \"FAILED_TO_START\"}"
    grep -hiE "error|assert|duplicate|invalid|failed" "$LOG_DIR/case_${LABEL}_node0.log" 2>/dev/null | tail -4
    pkill -f sglang.launch_server 2>/dev/null
    exit 0
fi

python /root/mig-demo/verify_and_bench.py --port 30000 --label "$LABEL" --n "$BN" --max-new-tokens "$BT"
pkill -f sglang.launch_server 2>/dev/null
sleep 2
exit 0
