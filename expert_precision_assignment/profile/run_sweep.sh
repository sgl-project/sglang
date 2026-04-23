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
#   FEWSHOT_AS_MULTITURN=1             (bench_eval only; default on — wraps
#                                       each fewshot exemplar as its own
#                                       chat turn so model learns to stop)
#   RULER_MAX_SEQ=65536                (flips RULER defaults: fewshot=0,
#                                       max_gen=128, no chat template, sets
#                                       --metadata, --context-length, YaRN)
#   BENCH_TASK=niah_single_2           (when TASK is a virtual task name like
#                                       ruler_niah_64k — routes bench_eval to
#                                       the actual RULER subtask)
#   LIMIT=128                          (caps --limit on bench_eval)
#   NIAH_CACHE_DIR=<path>              (opt-in disk cache for RULER niah_*
#                                       dataset construction; amortizes the
#                                       ~minutes-per-build cost across the
#                                       N parallel workers)
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
# RULER_MAX_SEQ (e.g. 8192, 65536, 131072, 262144) flips defaults for the
# RULER subtasks — matches run_calib.sh semantics. When set:
#   * num_fewshot=0, max_gen_toks=128 (RULER YAMLs)
#   * apply_chat_template=0, fewshot_as_multiturn=0 (RULER is raw completion;
#     the instruct-chat wrap gives 0% on Qwen3)
#   * METADATA = '{"max_seq_lengths":[$RULER_MAX_SEQ]}'
#   * Server gets --context-length $((RULER_MAX_SEQ+512)) and YaRN scaling
#     when that budget exceeds Qwen3's native 40960.
RULER_MAX_SEQ="${RULER_MAX_SEQ:-}"
# bench_eval knobs
if [ -n "$RULER_MAX_SEQ" ]; then
    NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
    MAX_GEN_TOKS="${MAX_GEN_TOKS:-128}"
    APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
    FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-0}"
    METADATA="${METADATA:-{\"max_seq_lengths\":[$RULER_MAX_SEQ]}}"
else
    NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
    MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
    APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
    # Wrap each fewshot exemplar as its own <|im_start|>user/assistant turn so
    # the model sees N examples of "assistant closes with <|im_end|>" before
    # emitting its own answer. Without this, chat-tuned models under the
    # Question:/Answer: fewshot scaffold don't learn to stop and run to
    # MAX_GEN_TOKS on every req. MUST match the value used in run_calib.sh —
    # the KV sizing depends on the output-length distribution this flag
    # produces.
    FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-1}"
    METADATA="${METADATA:-}"
fi
# LIMIT caps eval docs for quick/smoke sweeps (passed as --limit to bench_eval).
LIMIT="${LIMIT:-}"
# INCLUDE_PER_DOC=1 → passes --include-per-doc to bench_eval so the output
# JSONL carries per-sample prompts/responses/scores (for manual inspection
# on smoke runs). Off by default since full sweeps don't need the bloat.
INCLUDE_PER_DOC="${INCLUDE_PER_DOC:-0}"
# GEN_KWARGS_JSON → JSON dict merged into generation_kwargs (overrides task
# YAML). Required for BBH CoT, whose default `until=["Q","\n\n"]` triggers
# on any capital Q (e.g. words like "Quail", "Questions") and kills CoT
# responses mid-token. Use '{"until":["\nQ:","</s>"]}' for BBH.
GEN_KWARGS_JSON="${GEN_KWARGS_JSON:-}"
# bench_eval.py defaults temperature=0.7 (non-thinking), which overrides the
# per-task greedy setting and causes runaway generation up to MAX_GEN_TOKS.
# Force greedy here; override with TEMPERATURE=<x> if needed.
TEMPERATURE="${TEMPERATURE:-0}"
# Per-task system instructions (math/MCQ). bench_eval.py docstring:
#   Math: 'Please reason step by step, and put your final answer within \boxed{}.'
#   MCQ:  'Please show your choice in the answer field with only the choice letter...'
if [ -z "${SYSTEM_INSTRUCTION+x}" ]; then
    case "$TASK" in
        math*)
            SYSTEM_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
            ;;
        mmlu*|gpqa*|hellaswag|winogrande|arc_*)
            SYSTEM_INSTRUCTION='Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
            ;;
        *)
            # gsm8k intentionally left empty: the 5-shot template already ends
            # every example with "#### N", which is the format the strict-match
            # filter extracts. Adding a \boxed{} instruction creates a conflict
            # (model completes \boxed{}, then keeps generating since the fewshot
            # "#### N" stop signal isn't what it's producing → hits max_gen_toks
            # on 100% of requests).
            SYSTEM_INSTRUCTION=""
            ;;
    esac
fi

# MC_LIST env override: `MC_LIST="32 64 128 256" bash run_sweep.sh ...` —
# for restricted sweeps. Leave unset for the full 6-point ladder.
if [ -n "${MC_LIST:-}" ]; then
    read -r -a MC_LIST <<< "$MC_LIST"
else
    MC_LIST=(256 128 64 32 16 8)
fi
# VARIANTS env override: `VARIANTS="thr128 hot0" bash run_sweep.sh ...` — for
# smoke tests. Leave unset for the full 6×11 grid.
if [ -n "${VARIANTS:-}" ]; then
    read -r -a VARIANTS <<< "$VARIANTS"
else
    VARIANTS=(hot0 hot20 hot40 hot60 hot80 hot100)
fi
if [ -n "${GPUS:-}" ]; then
    read -r -a GPUS <<< "$GPUS"
else
    GPUS=(0 1 2 3 4 5 6 7)
fi

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
            --output-file "$out" > "$log" 2>&1
    else
        local ct_flag=()
        [ "$APPLY_CHAT_TEMPLATE" = "1" ] && ct_flag=(--apply-chat-template)
        local mt_flag=()
        [ "$FEWSHOT_AS_MULTITURN" = "1" ] && mt_flag=(--fewshot-as-multiturn)
        local sys_flag=()
        [ -n "$SYSTEM_INSTRUCTION" ] && sys_flag=(--system-instruction "$SYSTEM_INSTRUCTION")
        local meta_flag=()
        [ -n "$METADATA" ] && meta_flag=(--metadata "$METADATA")
        local lim_flag=()
        [ -n "$LIMIT" ] && lim_flag=(--limit "$LIMIT")
        local perdoc_flag=()
        [ "$INCLUDE_PER_DOC" = "1" ] && perdoc_flag=(--include-per-doc)
        local genkw_flag=()
        [ -n "$GEN_KWARGS_JSON" ] && genkw_flag=(--gen-kwargs-json "$GEN_KWARGS_JSON")
        python3 -m sglang.bench_eval \
            --task "${BENCH_TASK:-$TASK}" \
            --base-url "http://${HOST}:${port}" \
            --backend sglang-oai \
            --model "$BF16_MODEL" \
            --tokenizer "$BF16_MODEL" \
            --num-fewshot "$NUM_FEWSHOT" \
            --max-gen-toks "$MAX_GEN_TOKS" \
            --request-rate inf \
            --max-concurrency "$mc" \
            --temperature "$TEMPERATURE" \
            "${ct_flag[@]}" \
            "${mt_flag[@]}" \
            "${sys_flag[@]}" \
            "${meta_flag[@]}" \
            "${lim_flag[@]}" \
            "${perdoc_flag[@]}" \
            "${genkw_flag[@]}" \
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
    # RULER: size the context window to RULER_MAX_SEQ (plus headroom); YaRN
    # rope scaling when the target exceeds Qwen3's native 40960. sglang's
    # ModelConfig._derive_context_length reads max_position_embeddings
    # directly from the hf config (40960) — it does NOT recompute from
    # rope_scaling overrides — so any ctx > 40960 requires the explicit
    # SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 escape hatch to start up.
    local server_ctx_flag=()
    local allow_long_ctx=0
    if [ -n "$RULER_MAX_SEQ" ]; then
        local ctx=$((RULER_MAX_SEQ + 512))
        server_ctx_flag=(--context-length "$ctx")
        if [ "$ctx" -gt 40960 ]; then
            local factor
            factor=$(awk -v c=$ctx 'BEGIN{printf "%.4f", c/40960.0}')
            # transformers v5 exposes a single merged `rope_parameters` dict
            # (rope_theta + rope_type + scaling kwargs). rope_theta is no
            # longer a top-level attribute and rope_scaling overrides don't
            # feed back into rope_parameters. Override rope_parameters
            # directly — sglang's get_rope_config reads from it. Qwen3-30B's
            # native rope_theta is 1_000_000.
            server_ctx_flag+=(--json-model-override-args "{\"rope_parameters\":{\"rope_theta\":1000000.0,\"rope_type\":\"yarn\",\"factor\":$factor,\"original_max_position_embeddings\":40960}}")
            allow_long_ctx=1
        fi
    fi
    echo "[gpu$gpu $label] launching server on port $port (nccl $nccl_port, max-running=$mc)"
    CUDA_VISIBLE_DEVICES="$gpu" \
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="$allow_long_ctx" \
    python3 -m sglang.launch_server \
        --model-path "$BF16_MODEL" \
        --host "$HOST" --port "$port" \
        --nccl-port "$nccl_port" \
        --trust-remote-code \
        --max-running-requests "$mc" \
        --cuda-graph-max-bs "$mc" \
        "${server_ctx_flag[@]}" \
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
echo "  Sweep: $TASK  (${#MC_LIST[@]} mc × ${#VARIANTS[@]} variants = $((${#MC_LIST[@]}*${#VARIANTS[@]})))"
echo "  GPUs:      ${GPUS[*]}"
echo "  variants:  ${VARIANTS[*]}"
echo "  configs:   $CFG_DIR  [$CFG_SOURCE]"
echo "  out:       $OUT_DIR"
if [ "$TASK" = "sharegpt" ]; then
    echo "  bench:     bench_serving (n=$NUM_PROMPTS, ctx=$SHAREGPT_CONTEXT_LEN)"
else
    echo "  bench:     bench_eval --task=$TASK (fewshot=$NUM_FEWSHOT, max_gen=$MAX_GEN_TOKS, T=$TEMPERATURE)"
    echo "             apply_chat_template=$APPLY_CHAT_TEMPLATE  fewshot_as_multiturn=$FEWSHOT_AS_MULTITURN"
    [ -n "$SYSTEM_INSTRUCTION" ] && echo "  sys_instr: $SYSTEM_INSTRUCTION"
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
