#!/bin/bash
# SWE-bench Lite (oracle) comparison: baseline / hicache / flexkv.
# Launches sglang three times against the same prompt set and prints a
# side-by-side summary. Assumes ``sglang.launch_server`` is on PATH and the
# model is at $MODEL (default /path/to/Qwen3-8B).
#
# Usage:
#     export MODEL=/path/to/Qwen3-8B
#     bash run_swebench.sh                  # runs all three configs
#     bash run_swebench.sh flexkv           # runs one config
set -u

MODEL=${MODEL:-/path/to/Qwen3-8B}
PORT=30000
GPU=0
MEM_FRACTION=0.32
CPU_CACHE_GB=80
OUTDIR=./bench_out_swebench
mkdir -p "$OUTDIR"

export MODEL_PATH=$MODEL

NUM_PROMPTS=120
MAX_INPUT_TOKENS=28000
MAX_NEW_TOKENS=32
QPS=2.0
CONCURRENCY=24

CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=(baseline hicache flexkv)
fi

cat > ./flexkv_swebench.yaml <<EOF
cpu_cache_gb: $CPU_CACHE_GB
EOF

start_server() {
    local mode=$1
    local extra=""
    case "$mode" in
        baseline) extra="" ;;
        hicache)  extra="--enable-hierarchical-cache --hicache-size $CPU_CACHE_GB" ;;
        flexkv)   extra="--enable-flexkv --flexkv-config-file ./flexkv_swebench.yaml" ;;
        *) echo "bad mode $mode" >&2; return 1 ;;
    esac
    echo ">>> launch $mode (GPU $GPU)"
    rm -f /tmp/sglang.log /tmp/flexkv_layerwise_eventfd.sock /tmp/flexkv_server*
    env CUDA_VISIBLE_DEVICES=$GPU SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        nohup python3 -m sglang.launch_server \
            --model-path "$MODEL" --port "$PORT" --tp-size 1 \
            --attention-backend triton \
            --mem-fraction-static $MEM_FRACTION --max-running-requests 32 \
            --chunked-prefill-size 16384 --context-length 32000 \
            --enable-metrics --enable-request-time-stats-logging \
            $extra > /tmp/sglang.log 2>&1 &
    SERVER_PID=$!
    echo "    pid=$SERVER_PID"
}

wait_ready() {
    local elapsed=0
    while [[ $elapsed -lt 360 ]]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "    server died"; tail -20 /tmp/sglang.log; return 1
        fi
        if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            echo "    ready (${elapsed}s)"; return 0
        fi
        sleep 5; elapsed=$((elapsed+5))
    done
    echo "    timeout"; return 1
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 $SERVER_PID 2>/dev/null || break
            sleep 1
        done
        kill -9 $SERVER_PID 2>/dev/null
        sleep 3
    fi
}

run_bench() {
    local mode=$1
    local out_json="$OUTDIR/${mode}.json"
    local out_log="$OUTDIR/${mode}.log"
    echo ">>> bench $mode (cold+warm, $NUM_PROMPTS prompts)"
    python3 ./swebench_bench.py \
        --base-url http://127.0.0.1:$PORT --model Qwen/Qwen3-8B \
        --label $mode --num-prompts $NUM_PROMPTS \
        --max-input-tokens $MAX_INPUT_TOKENS --max-new-tokens $MAX_NEW_TOKENS \
        --qps $QPS --concurrency $CONCURRENCY --seed 0 --passes 2 \
        --out "$out_json" > "$out_log" 2>&1
    local h2d=$(grep -c "H2D transfer" /tmp/sglang.log 2>/dev/null || echo 0)
    local d2h=$(grep -c "D2H transfer" /tmp/sglang.log 2>/dev/null || echo 0)
    echo "    H2D=$h2d  D2H=$d2h"
    cp /tmp/sglang.log "$OUTDIR/${mode}_server.log" 2>/dev/null || true
}

trap stop_server EXIT

for mode in "${CONFIGS[@]}"; do
    stop_server
    sleep 5
    start_server "$mode"
    if wait_ready; then
        run_bench "$mode"
    else
        echo "skipping $mode"
    fi
    stop_server
done

echo
echo "=== ALL DONE ==="
for mode in "${CONFIGS[@]}"; do
    echo
    echo "--- $mode ---"
    tail -15 "$OUTDIR/${mode}.log" 2>/dev/null
done
