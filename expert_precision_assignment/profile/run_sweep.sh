#!/usr/bin/env bash
# Unified 6×11 sweep for profile/. Dispatches based on $TASK:
#   sharegpt     → python -m sglang.bench_serving       (efficiency)
#   <anything>   → python -m sglang.bench_eval --task <task>  (accuracy + perf)
#
# Same config grid (profile/configs/mc{mc}/variants/*.json) and the same
# (mc, variant) × GPU round-robin scheduling for every task — the only
# thing that changes is the bench command and the results subdir.
#
# Usage:
#   bash run_sweep.sh sharegpt
#   bash run_sweep.sh gsm8k
#   bash run_sweep.sh mmlu_flan_cot_zeroshot
#
# Env overrides:
#   NUM_PROMPTS=1024                   (sharegpt only)
#   SHAREGPT_CONTEXT_LEN=4096          (sharegpt only)
#   NUM_FEWSHOT=5  MAX_GEN_TOKS=512    (bench_eval only)
#   APPLY_CHAT_TEMPLATE=1              (bench_eval only; default on)
set -uo pipefail

TASK="${1:-}"
if [ -z "$TASK" ]; then
    echo "Usage: bash run_sweep.sh <task>     (e.g. sharegpt, gsm8k, mmlu_flan_cot_zeroshot)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer per-task configs (configs/<task>/mc{mc}/...) when present — these
# are regenerated from a calibrated amortized SLO and give a tighter K.
# Fall back to the flat configs/mc{mc}/... layout (worst-case SLO) if the
# task-specific tree wasn't generated.
if [ -d "$SCRIPT_DIR/configs/$TASK" ]; then
    CFG_DIR="$SCRIPT_DIR/configs/$TASK"
    CFG_SOURCE="per-task (calibrated)"
else
    CFG_DIR="$SCRIPT_DIR/configs"
    CFG_SOURCE="flat (worst-case)"
fi
OUT_DIR="$SCRIPT_DIR/results/$TASK"
mkdir -p "$OUT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"

# sharegpt knobs
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
# bench_eval knobs
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

MC_LIST=(8 16 32 64 128 256)
# VARIANTS env override: `VARIANTS="thr128 hot0" bash run_sweep.sh ...` — for
# smoke tests. Leave unset for the full 6×11 grid.
if [ -n "${VARIANTS:-}" ]; then
    read -r -a VARIANTS <<< "$VARIANTS"
else
    VARIANTS=(hot0 hot20 hot40 hot60 hot80 hot100 thr32 thr64 thr128 thr256 thr512)
fi
GPUS=(4 5 6 7)

declare -a ALL_PAIRS=()
for mc in "${MC_LIST[@]}"; do
    for v in "${VARIANTS[@]}"; do
        ALL_PAIRS+=("${mc}:${v}")
    done
done

declare -A GPU_PAIRS
for gpu in "${GPUS[@]}"; do
    GPU_PAIRS[$gpu]=""
done
for i in "${!ALL_PAIRS[@]}"; do
    g_idx=$(( i % ${#GPUS[@]} ))
    gpu=${GPUS[$g_idx]}
    GPU_PAIRS[$gpu]+="${ALL_PAIRS[$i]} "
done

declare -A GPU_PORT
declare -A GPU_NCCL_PORT
for gpu in "${GPUS[@]}"; do
    GPU_PORT[$gpu]=$((31300 + gpu))
    GPU_NCCL_PORT[$gpu]=$((41300 + gpu))
done

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

# Per-task result-file extension & existence check.
if [ "$TASK" = "sharegpt" ]; then
    RESULT_EXT="_n${NUM_PROMPTS}.jsonl"
else
    RESULT_EXT=".json"
fi

run_bench() {
    local port=$1
    local mc=$2
    local out=$3
    local log=$4
    if [ "$TASK" = "sharegpt" ]; then
        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url "http://${HOST}:${port}" \
            --dataset-name sharegpt \
            --sharegpt-context-len "$SHAREGPT_CONTEXT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$mc" \
            --output-details \
            --output-file "$out" > "$log" 2>&1
    else
        local ct_flag=()
        [ "$APPLY_CHAT_TEMPLATE" = "1" ] && ct_flag=(--apply-chat-template)
        python3 -m sglang.bench_eval \
            --task "$TASK" \
            --base-url "http://${HOST}:${port}" \
            --backend sglang \
            --model "$BF16_MODEL" \
            --tokenizer "$BF16_MODEL" \
            --num-fewshot "$NUM_FEWSHOT" \
            --max-gen-toks "$MAX_GEN_TOKS" \
            --request-rate inf \
            --max-concurrency "$mc" \
            "${ct_flag[@]}" \
            --output-file "$out" > "$log" 2>&1
    fi
}

run_one_variant() {
    local gpu=$1
    local mc=$2
    local variant=$3
    local port=$4
    local label="mc${mc}_${variant}"
    local config="$CFG_DIR/mc${mc}/variants/${variant}.json"

    if [ ! -f "$config" ]; then
        echo "[gpu$gpu $label] MISSING config: $config" >&2
        return 1
    fi

    local out="$OUT_DIR/${label}${RESULT_EXT}"
    if [ -f "$out" ]; then
        echo "[gpu$gpu $label] result exists, skip"
        return 0
    fi

    local server_log="$OUT_DIR/${label}_server.log"
    local nccl_port=${GPU_NCCL_PORT[$gpu]}
    echo "[gpu$gpu $label] launching server on port $port (nccl $nccl_port, max-running=$mc)"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$port" \
        --nccl-port "$nccl_port" \
        --trust-remote-code \
        --max-running-requests "$mc" \
        --cuda-graph-max-bs "$mc" \
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
    local bench_log="$OUT_DIR/${label}_bench.log"
    echo "[gpu$gpu $label mc=$mc] $TASK bench → $out"
    if ! run_bench "$port" "$mc" "$out" "$bench_log"; then
        echo "[gpu$gpu $label] bench failed (see $bench_log)" >&2
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
    local pairs=${GPU_PAIRS[$gpu]}
    for entry in $pairs; do
        local mc="${entry%%:*}"
        local variant="${entry#*:}"
        run_one_variant "$gpu" "$mc" "$variant" "$port" || true
    done
    echo "[gpu$gpu] DONE"
}

echo "============================================================"
echo "  Sweep: $TASK  (6 mc × 11 variants = 66)"
echo "  GPUs:      ${GPUS[*]}"
echo "  variants:  ${VARIANTS[*]}"
echo "  configs:   $CFG_DIR  [$CFG_SOURCE]"
echo "  out:       $OUT_DIR"
if [ "$TASK" = "sharegpt" ]; then
    echo "  bench:     bench_serving (n=$NUM_PROMPTS, ctx=$SHAREGPT_CONTEXT_LEN)"
else
    echo "  bench:     bench_eval --task=$TASK (fewshot=$NUM_FEWSHOT, max_gen=$MAX_GEN_TOKS)"
fi
for g in "${GPUS[@]}"; do
    npairs=$(echo ${GPU_PAIRS[$g]} | wc -w)
    echo "  gpu$g ($npairs pairs, port ${GPU_PORT[$g]}): ${GPU_PAIRS[$g]}"
done
echo "============================================================"

START_TS=$(date +%s)

for gpu in "${GPUS[@]}"; do
    run_gpu_worker "$gpu" > "$OUT_DIR/gpu${gpu}_worker.log" 2>&1 &
done
wait

END_TS=$(date +%s)
echo ""
echo "============================================================"
echo "  All GPU workers done in $((END_TS - START_TS))s."
echo "  Results: $OUT_DIR/"
echo "============================================================"

echo ""
echo "Collecting summary..."
python3 "$SCRIPT_DIR/collect_results.py" --results_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/summary.csv" || true
echo "  → $OUT_DIR/summary.csv"
