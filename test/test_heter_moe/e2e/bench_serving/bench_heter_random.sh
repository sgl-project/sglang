#!/usr/bin/env bash
# Bench heter_partial with random dispatch policy at rr=16, 64, 512.
# Compare against expert_load results from bench_all.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HETER_CONFIG="$SCRIPT_DIR/heter_config_random.json"

PORT=30000
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

NUM_PROMPTS="${1:-512}"
REQUEST_RATES=(16 64 512)
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

# ---------------------------------------------------------------------------
# Launch & bench
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Heter partial (random policy)"
echo "  request_rates=${REQUEST_RATES[*]}  num_prompts=$NUM_PROMPTS"
echo "============================================================"

python3 -m sglang.launch_server \
    --model-path "$BF16_MODEL" \
    --host "$HOST" --port "$PORT" \
    --trust-remote-code \
    --heter-precision-config "$HETER_CONFIG" &
SERVER_PID=$!
wait_for_server

for RR in "${REQUEST_RATES[@]}"; do
    OUTPUT_FILE="$OUT_DIR/heter_random_rr${RR}_n${NUM_PROMPTS}.jsonl"
    echo ""
    echo "--- bench: heter_random  rr=$RR  n=$NUM_PROMPTS → $OUTPUT_FILE"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --dataset-name sharegpt \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RR" \
        --output-file "$OUTPUT_FILE"
done

kill_server

echo ""
echo "============================================================"
echo "  Done. Results in $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/"
