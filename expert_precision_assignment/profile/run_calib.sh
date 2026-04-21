#!/usr/bin/env bash
# KV calibration pass for a task — always on full BF16 (no heter config),
# so the measured output-length distribution and accuracy are a clean
# baseline, independent of any heter-precision policy.
#
# Dispatches on $TASK:
#   sharegpt     → bench_serving --output-details (JSONL has input_lens/
#                  output_lens directly)
#   <anything>   → bench_eval --output-file (merged report JSONL; this
#                  script adds input_lens/output_lens at top level via
#                  report.merge_report, and also reports accuracy)
#
# Then invokes calib_kv.py --bench_details_jsonl on the produced JSONL
# to emit kv_calib/<task>.json with μ/σ of total_len.
# gen_heter_configs.py --calib_json consumes that file.
#
# Usage:
#   bash run_calib.sh sharegpt
#   bash run_calib.sh gsm8k
#   NUM_PROMPTS=512 bash run_calib.sh sharegpt
#   CALIB_MC=64 MAX_GEN_TOKS=512 bash run_calib.sh gsm8k
#   FEWSHOT_AS_MULTITURN=0 bash run_calib.sh gsm8k   # disable multiturn wrap
#   CALIB_LIMIT=256 bash run_calib.sh gsm8k          # use 256 docs instead of 128
#   CALIB_LIMIT= bash run_calib.sh gsm8k             # full task (no cap)
set -uo pipefail

TASK="${1:-}"
if [ -z "$TASK" ]; then
    echo "Usage: bash run_calib.sh <task>   (e.g. sharegpt, gsm8k)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "$SCRIPT_DIR/../policy/heter_assign" && pwd)"
OUT_DIR="$SCRIPT_DIR/kv_calib"
mkdir -p "$OUT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
PORT="${CALIB_PORT:-31304}"
NCCL_PORT="${CALIB_NCCL_PORT:-41304}"
GPU="${CALIB_GPU:-4}"
CALIB_MC="${CALIB_MC:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"
# bench_eval knobs (match run_sweep.sh defaults so calib matches sweep).
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
# Cap the number of eval docs — calibration only needs enough samples to
# estimate μ/σ of total_len, not the whole test set. 128 → stderr ≈ σ/11.3,
# plenty for KV sizing headroom. Set CALIB_LIMIT= (empty) to use the full task.
CALIB_LIMIT="${CALIB_LIMIT:-128}"
# Wrap each fewshot exemplar as its own <|im_start|>user/assistant turn so the
# model sees N examples of "assistant closes with <|im_end|>" before emitting
# its own answer. Without this, chat-tuned models under the Question:/Answer:
# fewshot scaffold don't learn to stop and run to MAX_GEN_TOKS on every req.
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-1}"
if [ -z "${SYSTEM_INSTRUCTION+x}" ]; then
    case "$TASK" in
        math*)
            SYSTEM_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
            ;;
        mmlu*|gpqa*|hellaswag|winogrande|arc_*)
            SYSTEM_INSTRUCTION='Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
            ;;
        *)
            # gsm8k intentionally has no system instruction: the 5-shot template
            # already ends every example with "#### N", which is what
            # strict-match extracts. Adding a \boxed{} instruction creates a
            # conflict with the fewshot format → 100% runaway to max_gen_toks.
            SYSTEM_INSTRUCTION=""
            ;;
    esac
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

if [ "$TASK" = "sharegpt" ]; then
    DETAILS_JSONL="$OUT_DIR/calib_sharegpt_n${NUM_PROMPTS}.jsonl"
else
    DETAILS_JSONL="$OUT_DIR/calib_${TASK}_mc${CALIB_MC}.jsonl"
fi
CALIB_JSON="$OUT_DIR/${TASK}.json"
SERVER_LOG="$OUT_DIR/server_${TASK}_bf16.log"

echo "============================================================"
echo "  KV calibration — task=$TASK  (FULL BF16, no heter config)"
if [ "$TASK" = "sharegpt" ]; then
    echo "  bench_serving mc=$CALIB_MC num_prompts=$NUM_PROMPTS"
else
    echo "  bench_eval mc=$CALIB_MC fewshot=$NUM_FEWSHOT max_gen=$MAX_GEN_TOKS T=$TEMPERATURE limit=${CALIB_LIMIT:-full}"
    echo "  apply_chat_template=$APPLY_CHAT_TEMPLATE  fewshot_as_multiturn=$FEWSHOT_AS_MULTITURN"
    [ -n "$SYSTEM_INSTRUCTION" ] && echo "  sys_instr: $SYSTEM_INSTRUCTION"
fi
echo "  server: gpu=$GPU port=$PORT"
echo "  → $DETAILS_JSONL"
echo "  → $CALIB_JSON"
echo "============================================================"

# 1. Launch server (full BF16 — no --heter-precision-config).
rm -f "$DETAILS_JSONL"
echo "Launching server..."
CUDA_VISIBLE_DEVICES="$GPU" python3 -m sglang.launch_server \
    --model-path "$BF16_MODEL" \
    --host "$HOST" --port "$PORT" \
    --nccl-port "$NCCL_PORT" \
    --trust-remote-code \
    --max-running-requests "$CALIB_MC" \
    --cuda-graph-max-bs "$CALIB_MC" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Shutting down server..."
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$SERVER_PID" 2>/dev/null || true
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 2. Wait for health
elapsed=0
while ! curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died during startup (see $SERVER_LOG)" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if [ $elapsed -ge 600 ]; then
        echo "Server didn't start within 600s" >&2
        exit 1
    fi
done
echo "Server ready after ${elapsed}s"

# 3. Run bench pass with per-request input_lens / output_lens capture.
if [ "$TASK" = "sharegpt" ]; then
    echo "Running bench_serving (n=$NUM_PROMPTS)..."
    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://${HOST}:${PORT}" \
        --dataset-name sharegpt \
        --sharegpt-context-len "$SHAREGPT_CONTEXT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$CALIB_MC" \
        --output-details \
        --output-file "$DETAILS_JSONL"
else
    ct_flag=()
    [ "$APPLY_CHAT_TEMPLATE" = "1" ] && ct_flag=(--apply-chat-template)
    mt_flag=()
    [ "$FEWSHOT_AS_MULTITURN" = "1" ] && mt_flag=(--fewshot-as-multiturn)
    sys_flag=()
    [ -n "$SYSTEM_INSTRUCTION" ] && sys_flag=(--system-instruction "$SYSTEM_INSTRUCTION")
    lim_flag=()
    [ -n "$CALIB_LIMIT" ] && lim_flag=(--limit "$CALIB_LIMIT")
    echo "Running bench_eval (task=$TASK) with per-doc output..."
    python3 -m sglang.bench_eval \
        --task "$TASK" \
        --base-url "http://${HOST}:${PORT}" \
        --backend sglang-oai \
        --model "$BF16_MODEL" \
        --tokenizer "$BF16_MODEL" \
        --num-fewshot "$NUM_FEWSHOT" \
        --max-gen-toks "$MAX_GEN_TOKS" \
        --request-rate inf \
        --max-concurrency "$CALIB_MC" \
        --temperature "$TEMPERATURE" \
        --include-per-doc \
        "${ct_flag[@]}" \
        "${mt_flag[@]}" \
        "${sys_flag[@]}" \
        "${lim_flag[@]}" \
        --output-file "$DETAILS_JSONL"
fi

# 4. Run calib_kv to parse μ/σ of total_len (input+output) from the
#    per-request arrays.
echo "Running calib_kv..."
python3 "$POLICY_DIR/calib_kv.py" \
    --bench_details_jsonl "$DETAILS_JSONL" \
    --out_file "$CALIB_JSON"

if [ "$TASK" != "sharegpt" ]; then
    echo ""
    echo "Accuracy from bench_eval (BF16 baseline):"
    python3 -c "
import json, sys
with open('$DETAILS_JSONL') as f:
    rec = json.loads(f.readline())
acc = rec.get('accuracy', {})
n = rec.get('n_samples', {})
for k, v in acc.items():
    print(f'  {k}: {v}')
if n:
    print(f'  n_samples: {n}')
"
fi

echo "============================================================"
echo "  DONE. Calibration JSON: $CALIB_JSON"
echo "============================================================"
