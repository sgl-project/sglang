#!/usr/bin/env bash
# Position-invariance sweep for heter-moe configs on a single GPU.
#
# 12 heter configs (A/B/C × n8/n16/n32/n64) × 4 request rates (16/64/256/1024)
# = 48 bench_serving runs over sharegpt.  One server per config (4 rr's per
# server launch) to amortize the ~2-min startup.
#
# Usage:
#   GPU_ID=7 bash run_invariance.sh                    # default NUM_PROMPTS=1024
#   GPU_ID=7 NUM_PROMPTS=2048 bash run_invariance.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
OUT_DIR="$SCRIPT_DIR/results_rr"
mkdir -p "$OUT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
PORT="${PORT:-30007}"
GPU_ID="${GPU_ID:-7}"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"

REQUEST_RATES=(16 64 256 1024)
LAYOUTS=(split_horizontal split_vertical random_assignment)
SIZES=(n8 n16 n32)

# Conda env (contains flashinfer, sglang, etc.)
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

SERVER_PID=""

launch_server() {
    local config="$1"
    local log_file="$2"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$PORT" \
        --trust-remote-code \
        --heter-precision-config "$config" > "$log_file" 2>&1 &
    SERVER_PID=$!
}

wait_server() {
    local max_wait=900 elapsed=0
    while ! curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ERROR: server died early (see log)"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: server didn't start within ${max_wait}s"
            return 1
        fi
    done
    echo "  server ready (${elapsed}s)"
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 2
        pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap kill_server EXIT INT TERM

echo "============================================================"
echo "  Invariance sweep"
echo "  GPU:          $GPU_ID"
echo "  port:         $PORT"
echo "  num_prompts:  $NUM_PROMPTS"
echo "  layouts:      ${LAYOUTS[*]}"
echo "  sizes:        ${SIZES[*]}"
echo "  rr's:         ${REQUEST_RATES[*]}"
echo "  out_dir:      $OUT_DIR"
echo "============================================================"

START_TS=$(date +%s)

for layout in "${LAYOUTS[@]}"; do
    for size in "${SIZES[@]}"; do
        tag="heter_${layout}_${size}"
        config="$CONFIG_DIR/heter_config_${layout}_${size}.json"
        if [ ! -f "$config" ]; then
            echo "[$tag] MISSING config: $config  (skip)"
            continue
        fi

        server_log="$OUT_DIR/${tag}_server.log"
        echo ""
        echo "[$tag] launching server on GPU $GPU_ID  (log: $server_log)"
        launch_server "$config" "$server_log"
        if ! wait_server; then
            kill_server
            sleep 3
            continue
        fi

        for rr in "${REQUEST_RATES[@]}"; do
            out="$OUT_DIR/${tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"
            bench_log="$OUT_DIR/${tag}_rr${rr}_bench.log"
            if [ -f "$out" ]; then
                echo "[$tag rr=$rr] already exists, skipping"
                continue
            fi
            curl -s -X POST "http://${HOST}:${PORT}/flush_cache" > /dev/null 2>&1 || true
            sleep 1
            echo "[$tag rr=$rr] bench n=$NUM_PROMPTS  →  $out"
            if ! python3 -m sglang.bench_serving \
                    --backend sglang \
                    --base-url "http://${HOST}:${PORT}" \
                    --dataset-name sharegpt \
                    --num-prompts "$NUM_PROMPTS" \
                    --request-rate "$rr" \
                    --output-file "$out" > "$bench_log" 2>&1; then
                echo "  bench failed (see $bench_log)"
            fi
        done

        kill_server
        sleep 5
    done
done

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All done in $((END_TS - START_TS))s. Results in $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/" | head -60
