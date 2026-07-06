#!/bin/bash
# Benchmark DCP (Decode Context Parallelism) serving performance.
#
# Runs accuracy (GSM8K) + throughput (bench_serving) across multiple DCP
# configurations: TP8 baseline, DCP8 AG+RS, DCP8 A2A, with FlashInfer and FA3.
#
# Usage:
#   bash benchmark/dcp/bench_dcp_serving.sh
#
# Prerequisites:
#   - 8x H100 GPUs
#   - DeepSeek-V2 model downloaded
#   - SGLang installed
#
# Output:
#   Results saved to benchmark/dcp/results/<branch>_<hash>/<config>/

set -euo pipefail

HOST=127.0.0.1
PORT=8188
MODEL=deepseek-ai/DeepSeek-V2
CONTEXT_LENGTH=163840
MAX_CC=512

BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short=7 HEAD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${SCRIPT_DIR}/results/${BRANCH}_${HASH}"

COMMON_ENV="SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1"

COMMON_ARGS="--model-path $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --enable-cache-report --log-level info --tp-size 8 \
--max-running-requests $MAX_CC --chunked-prefill-size 32768 \
--context-length $CONTEXT_LENGTH --disable-radix-cache --enable-symm-mem"

CONCURRENCIES=(1 2 4 8 16 32 64 128 256 512)

# Config format: NAME|BACKEND|MEM_FRAC|DCP_SIZE|DCP_COMM
CONFIGS=(
    "tp8_flashinfer|flashinfer|0.85|0|"
    "tp8_fa3|fa3|0.85|0|"
    "tp8_dcp8_agrs_flashinfer|flashinfer|0.85|8|ag_rs"
    "tp8_dcp8_agrs_fa3|fa3|0.83|8|ag_rs"
    "tp8_dcp8_a2a_flashinfer|flashinfer|0.85|8|a2a"
    "tp8_dcp8_a2a_fa3|fa3|0.83|8|a2a"
)

wait_for_server() {
    local max_wait=600
    local elapsed=0
    echo "Waiting for server on ${HOST}:${PORT} ..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q 200; then
            echo "Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: Server not ready within ${max_wait}s"
    return 1
}

kill_server() {
    echo "Killing server on port ${PORT} ..."
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    sleep 10
}

run_accuracy() {
    local output_dir="$1"
    local acc_file="${output_dir}/accuracy_gsm8k.txt"
    echo "Running accuracy test -> ${acc_file}"
    python3 benchmark/gsm8k/bench_sglang.py \
        --parallel 64 \
        --host "$HOST" --port "$PORT" 2>&1 | tee "$acc_file"
}

run_perf() {
    local output_dir="$1"
    echo "Running perf benchmarks -> ${output_dir}"
    for C in "${CONCURRENCIES[@]}"; do
        NUM_PROMPTS=$((C * 5))
        FILE_NAME="${output_dir}/cc${C}.txt"

        echo "--- Concurrency=$C, Prompts=$NUM_PROMPTS -> $FILE_NAME ---"

        python3 -m sglang.bench_serving --backend sglang \
            --host "$HOST" --port "$PORT" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len 4000 \
            --random-output-len 1500 \
            --random-range-ratio 0.1 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$C" \
            --disable-ignore-eos 2>&1 | tee "$FILE_NAME"
    done
}

start_server() {
    local cfg_name="$1"
    local backend="$2"
    local mem_frac="$3"
    local dcp="$4"
    local dcp_comm="$5"

    local extra_args=""
    if [ "$dcp" -gt 0 ]; then
        extra_args="--dcp-size ${dcp} --dcp-comm-backend ${dcp_comm}"
    fi

    echo "======================================================="
    echo "Starting: ${cfg_name} (backend=${backend} mem=${mem_frac} dcp=${dcp} comm=${dcp_comm})"
    echo "======================================================="

    eval "${COMMON_ENV} python3 -m sglang.launch_server ${COMMON_ARGS} \
        --mem-fraction-static ${mem_frac} \
        --attention-backend ${backend} \
        ${extra_args}" &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"
}

# ---- Main loop ----
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r CFG_NAME BACKEND MEM_FRAC DCP DCP_COMM <<< "$cfg"

    OUTPUT_DIR="${BASE_OUTPUT}/${CFG_NAME}"
    mkdir -p "$OUTPUT_DIR"

    kill_server
    start_server "$CFG_NAME" "$BACKEND" "$MEM_FRAC" "$DCP" "$DCP_COMM"

    if ! wait_for_server; then
        echo "Skipping ${CFG_NAME} due to server start failure"
        kill_server
        continue
    fi

    run_accuracy "$OUTPUT_DIR"
    run_perf "$OUTPUT_DIR"
    kill_server
done

echo ""
echo "======================================================="
echo "All benchmarks complete! Results in: ${BASE_OUTPUT}/"
echo "Plot results: python3 benchmark/dcp/plot_dcp_bench.py --results-dir ${BASE_OUTPUT}"
echo "======================================================="
