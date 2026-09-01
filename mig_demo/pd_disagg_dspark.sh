#!/usr/bin/env bash
# Case 6: PD-disaggregated serving of Qwen/Qwen3.8-27B on ONE H200:
#   prefill engine + decode engine (with in-process DSPARK draft) + mini_lb,
#   three processes co-located via --mem-fraction-static, KV moved over
#   mooncake forced-TCP (no RDMA NIC on this box).
# Phase 1: PD pair, decode runs DSPARK spec-dec (bench via LB :8000)
# Phase 2: PD pair, decode without spec (isolates spec gain under disagg)
set -uo pipefail

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export HF_HUB_CACHE=/cluster-storage/models/hub
export CUDA_VISIBLE_DEVICES=0
export MC_TCP_ENABLE_CONNECTION_POOL=true
LOG_DIR=/root/mig-demo
BENCH="python $LOG_DIR/verify_and_bench.py"
MODEL=Qwen/Qwen3.8-27B
# Text-only demo: disable the VL stack. The multimodal runtime reservation
# (mm_reservation_gb) otherwise eats the entire post-weight free memory of the
# co-located decode engine (weights 59.3G of ~78G free) -> "no memory for KV".
export SGLANG_VLM_CACHE_SIZE_MB=0  # zero the post-KV-sizing VL reservation (text-only demo)
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1  # mamba ratio 5->4 slots/req (frees persistent pool)
SPEC_FLAGS="--speculative-algorithm DSPARK \
    --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark"

wait_health() { # port logfile
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:$1/health > /dev/null 2>&1 && return 0
        sleep 5
    done
    echo "server on $1 FAILED"; tail -12 "$2"; exit 1
}

accept_len() { grep -o "accept len: [0-9.]*" "$1" | tail -3; }

kill_all() { # engines take ~10s to release GPU + bootstrap port after SIGTERM
    pkill -f sglang_router 2>/dev/null; pkill -f sglang.launch_server 2>/dev/null
    for i in $(seq 1 30); do
        pgrep -f "sglang.launch_server" > /dev/null || return 0
        sleep 1
    done
    pkill -9 -f sglang_router 2>/dev/null; pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2
}

start_pair() { # extra spec flags for decode ("none" = no spec)
    local spec=$1
    # Prefill first: 0.42 x 141GB = 59GB footprint (56GB weights + small KV).
    nohup python -m sglang.launch_server --host 127.0.0.1 --port 30000 \
        --model-path $MODEL \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake_tcp \
        --disaggregation-bootstrap-port 8998 --nccl-port 29501 \
        --mem-fraction-static 0.40 --max-running-requests 6 --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 > "$LOG_DIR/pd_prefill.log" 2>&1 &
    wait_health 30000 "$LOG_DIR/pd_prefill.log"
    grep -h max_total_num_tokens "$LOG_DIR/pd_prefill.log"
    # Decode second: gets 0.95 x total - used_prefill ~ 75GB (weights 59.5 + ~15GB KV).
    if [ "$spec" = "spec" ]; then SPEC=$SPEC_FLAGS; else SPEC=""; fi
    nohup python -m sglang.launch_server --host 127.0.0.1 --port 30001 \
        --model-path $MODEL \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend mooncake_tcp \
        --disaggregation-bootstrap-port 8998 --nccl-port 29502 \
        --mem-fraction-static 0.90 --cuda-graph-max-bs 6 --max-running-requests 6 --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 \
        $SPEC > "$LOG_DIR/pd_decode.log" 2>&1 &
    wait_health 30001 "$LOG_DIR/pd_decode.log"
    grep -h max_total_num_tokens "$LOG_DIR/pd_decode.log"
    # LB last (required front door: injects bootstrap room, fires at both).
    nohup python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
        --prefill http://127.0.0.1:30000 8998 --decode http://127.0.0.1:30001 \
        --host 127.0.0.1 --port 8000 > "$LOG_DIR/pd_lb.log" 2>&1 &
    wait_health 8000 "$LOG_DIR/pd_lb.log"
}

echo "=== Phase 1: PD-disagg, decode runs DSPARK ==="
kill_all
start_pair spec
$BENCH --port 8000 --label pd-dspark-c1 --n 4 --max-new-tokens 128
$BENCH --port 8000 --label pd-dspark-c6 --n 12 --max-new-tokens 128 --concurrency 6
accept_len "$LOG_DIR/pd_decode.log"
echo "=== Phase 2: PD-disagg, no spec ==="
kill_all
start_pair none
$BENCH --port 8000 --label pd-nospec-c1 --n 4 --max-new-tokens 128
$BENCH --port 8000 --label pd-nospec-c6 --n 12 --max-new-tokens 128 --concurrency 6
kill_all
echo "CASE6-DONE"

echo "=== CASE6-DONE ==="
true
