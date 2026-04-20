#!/usr/bin/env bash
# Accuracy sweep on Qwen3-30B-A3B over heter-MoE configurations.
#
# Sweep 1 (sweep_static_assignment): 9 configs, K varies via
#   policy/static/assign_experts.py --force_k {0,384,...,3072}; dispatch
#   patched to expert_batch threshold=0 (all heter always BF16).
#
# Sweep 2 (sweep_dynamic_dispatch): 11 configs at K=3072; dispatch
#   policy varies (6 random hot-pct + 5 expert_batch threshold).
#
# Total: 20 runs. Distributed 5/5/5/5 across GPUs 4-7.
# Per config we run ONE bench_eval on gsm8k at max_concurrency=256.
#
# Usage:
#   bash run_accuracy_sweep.sh               # full sweep
#   TASK=gsm8k bash run_accuracy_sweep.sh    # override task
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_CFG_DIR="$SCRIPT_DIR/configs/sweep_static_assignment"
DYN_BASE_DIR="$SCRIPT_DIR/configs/sweep_dynamic_dispatch"

TASK="${TASK:-gsm8k}"
OUT_STATIC="$SCRIPT_DIR/results/$TASK/sweep_static_assignment"
OUT_DYN="$SCRIPT_DIR/results/$TASK/sweep_dynamic_dispatch"
mkdir -p "$OUT_STATIC" "$OUT_DYN" "$SCRIPT_DIR/results"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"

# bench_eval gsm8k settings
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

# Per-GPU job lists — "label:configfile:outdir", space-separated.
# 20 runs split 5/5/5/5 across GPUs 4-7.
declare -A GPU_JOBS
GPU_JOBS[4]="\
s_K0:$STATIC_CFG_DIR/K0/heter_config.json:$OUT_STATIC \
s_K384:$STATIC_CFG_DIR/K384/heter_config.json:$OUT_STATIC \
s_K768:$STATIC_CFG_DIR/K768/heter_config.json:$OUT_STATIC \
s_K1152:$STATIC_CFG_DIR/K1152/heter_config.json:$OUT_STATIC \
s_K1536:$STATIC_CFG_DIR/K1536/heter_config.json:$OUT_STATIC"

GPU_JOBS[5]="\
s_K1920:$STATIC_CFG_DIR/K1920/heter_config.json:$OUT_STATIC \
s_K2304:$STATIC_CFG_DIR/K2304/heter_config.json:$OUT_STATIC \
s_K2688:$STATIC_CFG_DIR/K2688/heter_config.json:$OUT_STATIC \
s_K3072:$STATIC_CFG_DIR/K3072/heter_config.json:$OUT_STATIC \
d_hot0:$DYN_BASE_DIR/sweep_a/hot0.json:$OUT_DYN"

GPU_JOBS[6]="\
d_hot20:$DYN_BASE_DIR/sweep_a/hot20.json:$OUT_DYN \
d_hot40:$DYN_BASE_DIR/sweep_a/hot40.json:$OUT_DYN \
d_hot60:$DYN_BASE_DIR/sweep_a/hot60.json:$OUT_DYN \
d_hot80:$DYN_BASE_DIR/sweep_a/hot80.json:$OUT_DYN \
d_hot100:$DYN_BASE_DIR/sweep_a/hot100.json:$OUT_DYN"

GPU_JOBS[7]="\
d_thr32:$DYN_BASE_DIR/sweep_b/thr32.json:$OUT_DYN \
d_thr64:$DYN_BASE_DIR/sweep_b/thr64.json:$OUT_DYN \
d_thr128:$DYN_BASE_DIR/sweep_b/thr128.json:$OUT_DYN \
d_thr256:$DYN_BASE_DIR/sweep_b/thr256.json:$OUT_DYN \
d_thr512:$DYN_BASE_DIR/sweep_b/thr512.json:$OUT_DYN"

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
    local out_dir=$4
    local port=$5
    local out="$out_dir/${label}.json"
    local server_log="$out_dir/${label}_server.log"
    local bench_log="$out_dir/${label}_bench.log"

    if [ -f "$out" ]; then
        echo "[gpu$gpu $label] result exists, skip ($out)"
        return 0
    fi
    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $label] MISSING config: $config" >&2
        return 1
    fi

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

    curl -s -X POST "http://${HOST}:${port}/flush_cache" > /dev/null 2>&1 || true
    sleep 1

    echo "[gpu$gpu $label] bench_eval $TASK -> $out"
    if ! python3 -m sglang.bench_eval \
            --task "$TASK" \
            --base-url "http://${HOST}:${port}" \
            --backend sglang \
            --model "$BF16_MODEL" \
            --tokenizer "$BF16_MODEL" \
            --num-fewshot "$NUM_FEWSHOT" \
            --max-gen-toks "$MAX_GEN_TOKS" \
            --request-rate "$REQUEST_RATE" \
            --max-concurrency "$MAX_CONCURRENCY" \
            --apply-chat-template \
            --output-file "$out" > "$bench_log" 2>&1; then
        echo "[gpu$gpu $label] bench_eval FAILED (see $bench_log)" >&2
    fi

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
echo "  Accuracy profile sweep"
echo "  task:             $TASK"
echo "  GPUs:             4,5,6,7"
echo "  num_fewshot:      $NUM_FEWSHOT"
echo "  max_gen_toks:     $MAX_GEN_TOKS"
echo "  max_concurrency:  $MAX_CONCURRENCY"
echo "  request_rate:     $REQUEST_RATE"
echo "  out_static:       $OUT_STATIC"
echo "  out_dynamic:      $OUT_DYN"
for g in 4 5 6 7; do
    echo "  gpu$g (port ${GPU_PORT[$g]}): ${GPU_JOBS[$g]}"
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
echo "  Results: $OUT_STATIC/  and  $OUT_DYN/"
echo "============================================================"
