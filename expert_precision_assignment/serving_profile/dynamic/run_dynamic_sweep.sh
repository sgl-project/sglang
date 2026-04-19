#!/usr/bin/env bash
# Dynamic serving-profile sweep on Qwen3-30B-A3B with a FIXED expert
# int4/heter assignment (generated via policy/static/assign_experts.py for
# SLO max_concurrency=256, max_prompt_len=2048, max_output_len=2048).
#
# Sweep A (policy=random):        hot_pct ∈ {0,20,40,60,80,100}        (6 configs)
# Sweep B (policy=expert_batch):  threshold ∈ {32,64,128,256,512}       (5 configs)
# Request rates (both sweeps):    rr ∈ {8,16,32,64,128,256}             (6 rr's)
# Total runs: 11 × 6 = 66, distributed across GPUs 4,5,6,7.
#
# Usage:
#   NUM_PROMPTS=1024 bash run_dynamic_sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_A_DIR="$SCRIPT_DIR/configs/sweep_a"
SWEEP_B_DIR="$SCRIPT_DIR/configs/sweep_b"
OUT_A="$SCRIPT_DIR/results/sweep_a"
OUT_B="$SCRIPT_DIR/results/sweep_b"
mkdir -p "$OUT_A" "$OUT_B"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"  # max_prompt_len+max_output_len
REQUEST_RATES=(8 16 32 64 128 256)

# Per-GPU workload — "label:configfile:outdir" entries, space-separated.
# 11 configs split 3/3/3/2 across GPUs 4-7.
declare -A GPU_JOBS
GPU_JOBS[4]="a_hot0:$SWEEP_A_DIR/hot0.json:$OUT_A a_hot20:$SWEEP_A_DIR/hot20.json:$OUT_A a_hot40:$SWEEP_A_DIR/hot40.json:$OUT_A"
GPU_JOBS[5]="a_hot60:$SWEEP_A_DIR/hot60.json:$OUT_A a_hot80:$SWEEP_A_DIR/hot80.json:$OUT_A a_hot100:$SWEEP_A_DIR/hot100.json:$OUT_A"
GPU_JOBS[6]="b_thr32:$SWEEP_B_DIR/thr32.json:$OUT_B b_thr64:$SWEEP_B_DIR/thr64.json:$OUT_B b_thr128:$SWEEP_B_DIR/thr128.json:$OUT_B"
GPU_JOBS[7]="b_thr256:$SWEEP_B_DIR/thr256.json:$OUT_B b_thr512:$SWEEP_B_DIR/thr512.json:$OUT_B"

declare -A GPU_PORT
GPU_PORT[4]=31004
GPU_PORT[5]=31005
GPU_PORT[6]=31006
GPU_PORT[7]=31007

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

run_one_config() {
    local gpu=$1
    local label=$2       # e.g. a_hot40
    local config=$3
    local out_dir=$4
    local port=$5
    local out_tag="$label"

    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $label] MISSING config: $config" >&2
        return 1
    fi

    local server_log="$out_dir/${out_tag}_server.log"
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
        local out="$out_dir/${out_tag}_rr${rr}_n${NUM_PROMPTS}.jsonl"
        local bench_log="$out_dir/${out_tag}_rr${rr}_bench.log"
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
        local rest="${entry#*:}"
        local config="${rest%%:*}"
        local out_dir="${rest##*:}"
        run_one_config "$gpu" "$label" "$config" "$out_dir" "$port" || true
    done
    echo "[gpu$gpu] DONE"
}

echo "============================================================"
echo "  Dynamic serving-profile sweep"
echo "  GPUs:         4,5,6,7"
echo "  num_prompts:  $NUM_PROMPTS"
echo "  rr's:         ${REQUEST_RATES[*]}"
echo "  sharegpt ctx: $SHAREGPT_CONTEXT_LEN"
echo "  out_a:        $OUT_A"
echo "  out_b:        $OUT_B"
for g in 4 5 6 7; do
    echo "  gpu$g:        ${GPU_JOBS[$g]} (port ${GPU_PORT[$g]})"
done
echo "============================================================"

START_TS=$(date +%s)

for gpu in 4 5 6 7; do
    run_gpu_worker "$gpu" > "$SCRIPT_DIR/results/gpu${gpu}_worker.log" 2>&1 &
done
wait

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All GPU workers done in $((END_TS - START_TS))s."
echo "  Results: $OUT_A/  and  $OUT_B/"
echo "============================================================"
