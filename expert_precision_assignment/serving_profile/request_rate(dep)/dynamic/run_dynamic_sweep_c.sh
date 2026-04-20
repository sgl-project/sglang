#!/usr/bin/env bash
# Sweep C: granular hot_pct sweep with policy=expert_load.
# 11 configs (hot ∈ {0,10,...,100}) × 6 rr's {24,32,40,48,54,64} = 66 runs,
# distributed across GPUs 4,5,6,7 (3/3/3/2).
#
# Usage:
#   NUM_PROMPTS=1024 bash run_dynamic_sweep_c.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_C_DIR="$SCRIPT_DIR/configs/sweep_c"
OUT_C="$SCRIPT_DIR/results/sweep_c"
mkdir -p "$OUT_C"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
REQUEST_RATES=(24 32 40 48 54 64)

declare -A GPU_JOBS
GPU_JOBS[4]="c_hot0:$SWEEP_C_DIR/hot0.json c_hot10:$SWEEP_C_DIR/hot10.json c_hot20:$SWEEP_C_DIR/hot20.json"
GPU_JOBS[5]="c_hot30:$SWEEP_C_DIR/hot30.json c_hot40:$SWEEP_C_DIR/hot40.json c_hot50:$SWEEP_C_DIR/hot50.json"
GPU_JOBS[6]="c_hot60:$SWEEP_C_DIR/hot60.json c_hot70:$SWEEP_C_DIR/hot70.json c_hot80:$SWEEP_C_DIR/hot80.json"
GPU_JOBS[7]="c_hot90:$SWEEP_C_DIR/hot90.json c_hot100:$SWEEP_C_DIR/hot100.json"

declare -A GPU_PORT
GPU_PORT[4]=31104
GPU_PORT[5]=31105
GPU_PORT[6]=31106
GPU_PORT[7]=31107

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

run_one_config() {
    local gpu=$1
    local label=$2
    local config=$3
    local port=$4

    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $label] MISSING config: $config" >&2
        return 1
    fi

    local server_log="$OUT_C/${label}_server.log"
    echo "[gpu$gpu $label] launching server on port $port"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$port" \
        --trust-remote-code \
        --heter-precision-config "$config" > "$server_log" 2>&1 &
    local server_pid=$!

    local elapsed=0
    while ! curl -s "http://${HOST}:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[gpu$gpu $label] server died during startup (see $server_log)" >&2
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge 900 ]; then
            echo "[gpu$gpu $label] server didn't start within 900s" >&2
            kill -KILL "$server_pid" 2>/dev/null || true
            return 1
        fi
    done
    echo "[gpu$gpu $label] server ready after ${elapsed}s"

    for rr in "${REQUEST_RATES[@]}"; do
        local out="$OUT_C/${label}_rr${rr}_n${NUM_PROMPTS}.jsonl"
        local bench_log="$OUT_C/${label}_rr${rr}_bench.log"
        if [ -f "$out" ]; then
            echo "[gpu$gpu $label rr=$rr] already exists, skip"
            continue
        fi
        curl -s -X POST "http://${HOST}:${port}/flush_cache" > /dev/null 2>&1 || true
        sleep 1
        echo "[gpu$gpu $label rr=$rr] bench → $out"
        if ! python3 -m sglang.bench_serving \
                --backend sglang \
                --base-url "http://${HOST}:${port}" \
                --dataset-name sharegpt \
                --sharegpt-context-len "$SHAREGPT_CONTEXT_LEN" \
                --num-prompts "$NUM_PROMPTS" \
                --request-rate "$rr" \
                --output-file "$out" > "$bench_log" 2>&1; then
            echo "[gpu$gpu $label rr=$rr] bench failed (see $bench_log)" >&2
        fi
    done

    pkill -TERM -P "$server_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[gpu$gpu $label] server stopped"
    sleep 3
}

run_gpu_worker() {
    local gpu=$1
    local port=${GPU_PORT[$gpu]}
    local jobs=${GPU_JOBS[$gpu]}
    for entry in $jobs; do
        local label="${entry%%:*}"
        local config="${entry#*:}"
        run_one_config "$gpu" "$label" "$config" "$port" || true
    done
    echo "[gpu$gpu] DONE"
}

echo "============================================================"
echo "  Sweep C (policy=expert_load, granular hot_pct)"
echo "  GPUs:         4,5,6,7"
echo "  num_prompts:  $NUM_PROMPTS"
echo "  rr's:         ${REQUEST_RATES[*]}"
echo "  sharegpt ctx: $SHAREGPT_CONTEXT_LEN"
echo "  out:          $OUT_C"
for g in 4 5 6 7; do
    echo "  gpu$g:        ${GPU_JOBS[$g]} (port ${GPU_PORT[$g]})"
done
echo "============================================================"

START_TS=$(date +%s)

for gpu in 4 5 6 7; do
    run_gpu_worker "$gpu" > "$SCRIPT_DIR/results/gpu${gpu}_worker_c.log" 2>&1 &
done
wait

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All GPU workers done in $((END_TS - START_TS))s."
echo "  Results: $OUT_C/"
echo "============================================================"
