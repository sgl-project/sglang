#!/usr/bin/env bash
# Bench every precision config across an extended request-rate sweep.
# Round-robins request rates across all *idle* GPUs (one server per GPU at a
# time per config), so a long sweep finishes faster than a single-GPU run.
#
# Differences vs bench_all.sh:
#   - REQUEST_RATES = {16, 32, 64, 128, 256, 512, 1024, 2048}  (8 points)
#   - GPU pool = currently *idle* GPUs (no compute / no memory in use), not
#     just every visible GPU.
#   - Within each config, request rates are dispatched round-robin across the
#     idle pool; each (config, rr) pair gets a fresh server on its own GPU.
#
# Usage:
#   bash bench_all_rr.sh [num_prompts]
#
# Examples:
#   bash bench_all_rr.sh          # default: 1024 prompts
#   bash bench_all_rr.sh 2048     # 2048 prompts per (config, rr) pair

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
INT4_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B-GPTQ-Int4/snapshots/9b534e4318b7ebc3c961a839f13eb18b1833f441"
HETER_PARTIAL_CONFIG="$SCRIPT_DIR/../partial_bf16/heter_config.json"
HETER_RANDOM_CONFIG="$SCRIPT_DIR/heter_config_random.json"
HETER_THRESH128_CONFIG="$SCRIPT_DIR/heter_config_thresh128.json"

HOST="127.0.0.1"
BASE_PORT=31000

NUM_PROMPTS="${1:-1024}"
REQUEST_RATES=(16 32 64 128 256 512 1024 2048)
OUT_DIR="$SCRIPT_DIR/results_rr"
mkdir -p "$OUT_DIR"

# Idle thresholds (a GPU counts as "available" only if both are below).
IDLE_MEM_MB="${IDLE_MEM_MB:-500}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-5}"

# ---------------------------------------------------------------------------
# Idle GPU discovery
# ---------------------------------------------------------------------------
discover_idle_gpus() {
    local raw idx util mem
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Honor explicit override; trust the user.
        IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
        return
    fi
    GPU_LIST=()
    while IFS=, read -r idx util mem; do
        idx="${idx// /}"; util="${util// /}"; mem="${mem// /}"
        if [ "$util" -lt "$IDLE_UTIL_PCT" ] && [ "$mem" -lt "$IDLE_MEM_MB" ]; then
            GPU_LIST+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                       --format=csv,noheader,nounits)
}

discover_idle_gpus
NUM_GPUS=${#GPU_LIST[@]}
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: no idle GPUs found (mem<${IDLE_MEM_MB}MB, util<${IDLE_UTIL_PCT}%)."
    echo "       Override with CUDA_VISIBLE_DEVICES=... bash bench_all_rr.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Configs: tag | launch args (excluding --host/--port, which are added per-slot)
# ---------------------------------------------------------------------------
CONFIGS=(
    # "bf16|--model-path $BF16_MODEL --trust-remote-code"
    # "int4|--model-path $INT4_MODEL --trust-remote-code"
    # "heter_partial|--model-path $BF16_MODEL --trust-remote-code --heter-precision-config $HETER_PARTIAL_CONFIG"
    # "heter_random|--model-path $BF16_MODEL --trust-remote-code --heter-precision-config $HETER_RANDOM_CONFIG"
    "heter_thresh128|--model-path $BF16_MODEL --trust-remote-code --heter-precision-config $HETER_THRESH128_CONFIG"
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
# Per-(config, rr) worker: launch one server on its own GPU, run a single
# request rate, tear down. One GPU is fully owned for the duration.
# ---------------------------------------------------------------------------
run_one_rr() {
    local tag="$1"
    local gpu_id="$2"
    local port="$3"
    local rr="$4"
    local args="$5"

    local base_url="http://${HOST}:${port}"
    local log_file="$OUT_DIR/${tag}_rr${rr}_server.log"
    local output_file="$OUT_DIR/${tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"

    echo "[$tag rr=$rr] launching on GPU $gpu_id port $port (log: $log_file)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 -m sglang.launch_server \
        --host "$HOST" --port "$port" $args > "$log_file" 2>&1 &
    local server_pid=$!

    local max_wait=600 elapsed=0
    while ! curl -s "${base_url}/health" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[$tag rr=$rr] ERROR: server died, see $log_file"
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "[$tag rr=$rr] ERROR: server did not start within ${max_wait}s"
            kill -TERM "$server_pid" 2>/dev/null || true
            return 1
        fi
    done
    echo "[$tag rr=$rr] server ready (${elapsed}s)"

    curl -s -X POST "${base_url}/flush_cache" >> "$log_file" 2>&1 || true
    sleep 1
    echo "[$tag rr=$rr] bench n=$NUM_PROMPTS → $output_file"
    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$base_url" \
        --dataset-name sharegpt \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rr" \
        --output-file "$output_file" >> "$log_file" 2>&1
    local bench_status=$?

    echo "[$tag rr=$rr] stopping server (pid=$server_pid)"
    pkill -TERM -P "$server_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    return $bench_status
}

# ---------------------------------------------------------------------------
# Round-robin scheduler: for each config, dispatch its 8 request rates
# round-robin across the idle GPU pool. Wait for all workers of a config
# before moving to the next one (avoids two configs fighting for the same
# big-model weights cache and keeps logs grouped).
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  bench_all_rr"
echo "  configs:        ${#CONFIGS[@]}"
echo "  request_rates:  ${REQUEST_RATES[*]}"
echo "  idle GPUs:      ${GPU_LIST[*]}  (n=$NUM_GPUS)"
echo "  num_prompts:    $NUM_PROMPTS"
echo "============================================================"

port_counter=0
for entry in "${CONFIGS[@]}"; do
    tag="${entry%%|*}"
    args="${entry#*|}"

    echo ""
    echo "===== config: $tag ====="

    # Round-robin the rr's across GPUs in waves of NUM_GPUS in-flight tasks.
    rr_idx=0
    while [ $rr_idx -lt ${#REQUEST_RATES[@]} ]; do
        ACTIVE_PIDS=()
        declare -a WAVE_TAGS=()
        for ((slot=0; slot<NUM_GPUS && rr_idx<${#REQUEST_RATES[@]}; slot++, rr_idx++)); do
            rr="${REQUEST_RATES[$rr_idx]}"
            gpu="${GPU_LIST[$slot]}"
            port=$((BASE_PORT + port_counter))
            port_counter=$((port_counter + 1))
            run_one_rr "$tag" "$gpu" "$port" "$rr" "$args" &
            ACTIVE_PIDS+=($!)
            WAVE_TAGS+=("$tag rr=$rr")
        done
        for i in "${!ACTIVE_PIDS[@]}"; do
            pid="${ACTIVE_PIDS[$i]}"
            if ! wait "$pid"; then
                echo "ERROR: ${WAVE_TAGS[$i]} failed"
            fi
        done
        ACTIVE_PIDS=()
    done
done

echo ""
echo "============================================================"
echo "  All done. Results in $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/"
