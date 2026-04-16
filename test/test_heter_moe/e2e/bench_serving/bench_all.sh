#!/usr/bin/env bash
# Run bench_serving for all precision configs across multiple request rates.
# Runs up to NUM_GPUS configs concurrently, one server per GPU.
#
# Usage:
#   bash bench_all.sh [num_prompts]
#
# Examples:
#   bash bench_all.sh          # default: 512 prompts
#   bash bench_all.sh 1000     # 1000 prompts
#
# GPU selection: honors CUDA_VISIBLE_DEVICES if set, otherwise uses all
# GPUs reported by `nvidia-smi -L`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
HETER_PARTIAL_CONFIG="$SCRIPT_DIR/../partial_bf16/heter_config.json"
HETER_RANDOM_CONFIG="$SCRIPT_DIR/heter_config_random.json"

HOST="127.0.0.1"
BASE_PORT=30000

NUM_PROMPTS="${1:-1024}"
REQUEST_RATES=(16 64 256 1024)
OUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_LIST < <(nvidia-smi -L | awk '{print NR-1}')
fi
NUM_GPUS=${#GPU_LIST[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: no GPUs detected"
    exit 1
fi

# ---------------------------------------------------------------------------
# Configs: tag | launch args (excluding --host/--port, which are added per-slot)
# ---------------------------------------------------------------------------
CONFIGS=(
    "bf16|--model-path $BF16_MODEL --trust-remote-code"
    "int4|--model-path $INT4_MODEL --trust-remote-code"
    "heter_partial|--model-path $BF16_MODEL --trust-remote-code --heter-precision-config $HETER_PARTIAL_CONFIG"
    "heter_random|--model-path $BF16_MODEL --trust-remote-code --heter-precision-config $HETER_RANDOM_CONFIG"
)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
declare -a ACTIVE_PIDS=()
cleanup() {
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    pkill -f "sglang.launch_server" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Per-config worker: launch server on assigned GPU/port, run all request
# rates, tear down. Logs go to <tag>_server.log.
# ---------------------------------------------------------------------------
run_one() {
    local tag="$1"
    local gpu_id="$2"
    local port="$3"
    shift 3
    local base_url="http://${HOST}:${port}"
    local log_file="$OUT_DIR/${tag}_server.log"

    echo "[$tag] launching on GPU $gpu_id port $port (log: $log_file)"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -m sglang.launch_server \
        --host "$HOST" --port "$port" "$@" > "$log_file" 2>&1 &
    local server_pid=$!

    local max_wait=600 elapsed=0
    while ! curl -s "${base_url}/health" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[$tag] ERROR: server died, see $log_file"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "[$tag] ERROR: server did not start within ${max_wait}s"
            kill -TERM "$server_pid" 2>/dev/null || true
            return 1
        fi
    done
    echo "[$tag] server ready (${elapsed}s)"

    for rr in "${REQUEST_RATES[@]}"; do
        local output_file="$OUT_DIR/${tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"
        echo "[$tag] flush radix cache before rr=$rr"
        curl -s -X POST "${base_url}/flush_cache" >> "$log_file" 2>&1 || true
        sleep 1
        echo "[$tag] bench rr=$rr  n=$NUM_PROMPTS → $output_file"
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url "$base_url" \
            --dataset-name sharegpt \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$rr" \
            --output-file "$output_file" >> "$log_file" 2>&1
    done

    echo "[$tag] stopping server (pid=$server_pid)"
    pkill -TERM -P "$server_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Dispatch in batches of NUM_GPUS
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Benchmarking ${#CONFIGS[@]} configs on ${NUM_GPUS} GPU(s): ${GPU_LIST[*]}"
echo "  request_rates=${REQUEST_RATES[*]}  num_prompts=$NUM_PROMPTS"
echo "============================================================"

idx=0
batch=0
while [ $idx -lt ${#CONFIGS[@]} ]; do
    batch=$((batch + 1))
    echo ""
    echo "--- batch $batch ---"
    ACTIVE_PIDS=()
    declare -a BATCH_TAGS=()
    for ((slot=0; slot<NUM_GPUS && idx<${#CONFIGS[@]}; slot++, idx++)); do
        entry="${CONFIGS[$idx]}"
        tag="${entry%%|*}"
        args="${entry#*|}"
        gpu="${GPU_LIST[$slot]}"
        port=$((BASE_PORT + idx))
        # shellcheck disable=SC2086
        run_one "$tag" "$gpu" "$port" $args &
        ACTIVE_PIDS+=($!)
        BATCH_TAGS+=("$tag")
    done
    fail=0
    for i in "${!ACTIVE_PIDS[@]}"; do
        pid="${ACTIVE_PIDS[$i]}"
        if ! wait "$pid"; then
            echo "ERROR: config '${BATCH_TAGS[$i]}' failed"
            fail=1
        fi
    done
    ACTIVE_PIDS=()
    if [ $fail -ne 0 ]; then
        echo "WARNING: one or more configs in batch $batch failed; continuing"
    fi
done

echo ""
echo "============================================================"
echo "  All done. Results in $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/"
