#!/usr/bin/env bash
# Run bench_serving for all three precision configs across multiple request rates.
# Launches each server once and iterates all request rates before tearing down.
#
# Usage:
#   bash bench_all.sh [num_prompts]
#
# Examples:
#   bash bench_all.sh          # default: 512 prompts
#   bash bench_all.sh 1000     # 1000 prompts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
HETER_CONFIG="$SCRIPT_DIR/../partial_bf16/heter_config.json"

PORT=30000
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

NUM_PROMPTS="${1:-512}"
REQUEST_RATES=(16 64 256)
OUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
wait_for_server() {
    echo "Waiting for server at $BASE_URL ..."
    local max_wait=600
    local elapsed=0
    while ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            exit 1
        fi
    done
    echo "Server ready (${elapsed}s)"
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Killing server process tree (pid=$SERVER_PID) ..."
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
}
trap kill_server EXIT

launch_server() {
    local label="$1"; shift
    echo ""
    echo "============================================================"
    echo "  Launching server: $label"
    echo "  cmd: python3 -m sglang.launch_server $*"
    echo "============================================================"
    python3 -m sglang.launch_server "$@" &
    SERVER_PID=$!
    wait_for_server
}

run_bench() {
    local tag="$1"
    local rr="$2"
    local output_file="$OUT_DIR/${tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"

    echo ""
    echo "--- bench: $tag  rr=$rr  n=$NUM_PROMPTS → $output_file"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --dataset-name sharegpt \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rr" \
        --output-file "$output_file"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Benchmarking: bf16, int4, heter"
echo "  request_rates=${REQUEST_RATES[*]}  num_prompts=$NUM_PROMPTS"
echo "============================================================"

# --- BF16 ---
launch_server "bf16" \
    --model-path "$BF16_MODEL" --host "$HOST" --port "$PORT" --trust-remote-code
for RR in "${REQUEST_RATES[@]}"; do
    run_bench "bf16" "$RR"
done
kill_server

# --- INT4 ---
launch_server "int4" \
    --model-path "$INT4_MODEL" --host "$HOST" --port "$PORT" --trust-remote-code
for RR in "${REQUEST_RATES[@]}"; do
    run_bench "int4" "$RR"
done
kill_server

# --- Heter (partial BF16) ---
launch_server "heter_partial" \
    --model-path "$BF16_MODEL" --host "$HOST" --port "$PORT" --trust-remote-code \
    --heter-precision-config "$HETER_CONFIG"
for RR in "${REQUEST_RATES[@]}"; do
    run_bench "heter_partial" "$RR"
done
kill_server

echo ""
echo "============================================================"
echo "  All done. Results in $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/"
