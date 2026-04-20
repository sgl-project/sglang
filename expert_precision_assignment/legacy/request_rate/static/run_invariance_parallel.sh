#!/usr/bin/env bash
# Parallel re-sweep of vertical + random heter-moe configs on GPUs 4-7,
# after the per-layer gc fix in heter_moe.py (build_heter_moe_modules).
# Results go to results_rr_gc_fix/ so originals stay untouched for A/B.
#
# 6 configs (vertical/random × n8/n16/n32) × 4 request rates (16/64/256/1024)
# = 24 bench runs distributed across 4 GPUs.
#
# Usage:
#   NUM_PROMPTS=1024 bash run_invariance_parallel.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
OUT_DIR="$SCRIPT_DIR/results_rr_gc_fix"
mkdir -p "$OUT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
REQUEST_RATES=(16 64 256 1024)

# Per-GPU config list (space-separated). 2+1+2+1 balanced across 4 GPUs.
declare -A GPU_CONFIGS
GPU_CONFIGS[4]="split_vertical_n8 split_vertical_n32"
GPU_CONFIGS[5]="split_vertical_n16"
GPU_CONFIGS[6]="random_assignment_n8 random_assignment_n32"
GPU_CONFIGS[7]="random_assignment_n16"

declare -A GPU_PORT
GPU_PORT[4]=30024
GPU_PORT[5]=30025
GPU_PORT[6]=30026
GPU_PORT[7]=30027

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

run_one_config() {
    local gpu=$1
    local tag=$2  # e.g. split_vertical_n32
    local port=$3
    local out_tag="heter_${tag}"

    local config="$CONFIG_DIR/heter_config_${tag}.json"
    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $tag] MISSING config: $config" >&2
        return 1
    fi

    local server_log="$OUT_DIR/${out_tag}_server.log"
    echo "[gpu$gpu $tag] launching server on port $port"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$port" \
        --trust-remote-code \
        --heter-precision-config "$config" > "$server_log" 2>&1 &
    local server_pid=$!

    # Wait up to 15 min for ready
    local elapsed=0
    while ! curl -s "http://${HOST}:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[gpu$gpu $tag] server died during startup (see $server_log)" >&2
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge 900 ]; then
            echo "[gpu$gpu $tag] server didn't start within 900s" >&2
            kill -KILL "$server_pid" 2>/dev/null || true
            return 1
        fi
    done
    echo "[gpu$gpu $tag] server ready after ${elapsed}s"

    for rr in "${REQUEST_RATES[@]}"; do
        local out="$OUT_DIR/${out_tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"
        local bench_log="$OUT_DIR/${out_tag}_rr${rr}_bench.log"
        if [ -f "$out" ]; then
            echo "[gpu$gpu $tag rr=$rr] already exists, skip"
            continue
        fi
        curl -s -X POST "http://${HOST}:${port}/flush_cache" > /dev/null 2>&1 || true
        sleep 1
        echo "[gpu$gpu $tag rr=$rr] bench → $out"
        if ! python3 -m sglang.bench_serving \
                --backend sglang \
                --base-url "http://${HOST}:${port}" \
                --dataset-name sharegpt \
                --num-prompts "$NUM_PROMPTS" \
                --request-rate "$rr" \
                --output-file "$out" > "$bench_log" 2>&1; then
            echo "[gpu$gpu $tag rr=$rr] bench failed (see $bench_log)" >&2
        fi
    done

    # Kill server and its children
    pkill -TERM -P "$server_pid" 2>/dev/null || true
    kill -TERM "$server_pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$server_pid" 2>/dev/null || true
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "[gpu$gpu $tag] server stopped"
    # Give CUDA context a moment to release
    sleep 3
}

run_gpu_worker() {
    local gpu=$1
    local port=${GPU_PORT[$gpu]}
    local configs=${GPU_CONFIGS[$gpu]}
    for tag in $configs; do
        run_one_config "$gpu" "$tag" "$port" || true
    done
    echo "[gpu$gpu] DONE"
}

echo "============================================================"
echo "  Parallel invariance re-sweep (post gc fix)"
echo "  GPUs:        4,5,6,7"
echo "  num_prompts: $NUM_PROMPTS"
echo "  rr's:        ${REQUEST_RATES[*]}"
echo "  out_dir:     $OUT_DIR"
for g in 4 5 6 7; do
    echo "  gpu$g:       ${GPU_CONFIGS[$g]} (port ${GPU_PORT[$g]})"
done
echo "============================================================"

START_TS=$(date +%s)

for gpu in 4 5 6 7; do
    run_gpu_worker "$gpu" > "$OUT_DIR/gpu${gpu}_worker.log" 2>&1 &
done
wait

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All GPU workers done in $((END_TS - START_TS))s."
echo "  Results: $OUT_DIR/"
echo "============================================================"
ls -l "$OUT_DIR/" | head -80
