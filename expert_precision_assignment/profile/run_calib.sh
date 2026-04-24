#!/usr/bin/env bash
# KV calibration pass for a task — always on full BF16 (no heter config),
# so the measured output-length distribution and accuracy are a clean
# baseline, independent of any heter-precision policy.
#
# Dispatches on $TASK (three modes, auto-detected):
#   sharegpt                    → MODE=sharegpt   (bench_serving --dataset-name sharegpt)
#   prompts/<task>.jsonl exists → MODE=openai     (bench_serving --dataset-name openai,
#                                                  for user-prepared prompts e.g. supergpqa,
#                                                  ifbench, livecodebench_v6 — scoring is
#                                                  offline via scoring/score_traces_<task>.py)
#   else                        → MODE=bench_eval (lm-eval-harness tasks like gsm8k, mmlu*)
#
# All three modes write a JSONL with top-level `input_lens`/`output_lens` arrays
# that calib_kv.py consumes to emit kv_calib/<task>.json with μ/σ of total_len.
# gen_heter_configs.py --calib_json consumes that file.
#
# Usage:
#   bash run_calib.sh sharegpt
#   bash run_calib.sh gsm8k
#   bash run_calib.sh supergpqa            # after `python prompts/prepare_prompts_supergpqa.py`
#   NUM_PROMPTS=512 bash run_calib.sh sharegpt
#   CALIB_MC=64 MAX_GEN_TOKS=512 bash run_calib.sh gsm8k
#   FEWSHOT_AS_MULTITURN=0 bash run_calib.sh gsm8k   # disable multiturn wrap
#   CALIB_LIMIT=256 bash run_calib.sh gsm8k          # use 256 docs instead of 128
#   CALIB_LIMIT= bash run_calib.sh gsm8k             # full task (no cap)
#   NUM_PROMPTS=128 bash run_calib.sh supergpqa      # calib on first 128 prompts
set -uo pipefail

TASK="${1:-}"
if [ -z "$TASK" ]; then
    echo "Usage: bash run_calib.sh <task>   (e.g. sharegpt, gsm8k)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "$SCRIPT_DIR/../policy/heter_assign" && pwd)"
OUT_DIR="$SCRIPT_DIR/kv_calib"
PROMPTS_DIR="$SCRIPT_DIR/prompts"
PROMPTS_JSONL="$PROMPTS_DIR/${TASK}.jsonl"
mkdir -p "$OUT_DIR"

if [ "$TASK" = "sharegpt" ]; then
    MODE="sharegpt"
elif [ -f "$PROMPTS_JSONL" ]; then
    MODE="openai"
else
    MODE="bench_eval"
fi

# Runtime knobs — precedence: explicit env > recipe's runtime.* > hardcoded default.
# Recipe vars (RECIPE_RUNTIME__*) are set by run_pipeline.sh when called via
# recipe; standalone `bash run_calib.sh <task>` falls through to the defaults.
BF16_MODEL="${BF16_MODEL:-${RECIPE_RUNTIME__MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}}"
HOST="${HOST:-${RECIPE_RUNTIME__HOST:-127.0.0.1}}"
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

# Resolve Python: explicit $PYTHON > recipe's runtime.python > hardcoded default.
# Hard-fail if the chosen interpreter can't import sglang — the previous silent
# fallback to system python3 created a trap where the server died with
# ModuleNotFoundError only after 60+ seconds of startup.
PYTHON="${PYTHON:-${RECIPE_RUNTIME__PYTHON:-/data/junzhou/.venv-bfcl/bin/python}}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: runtime python not executable: $PYTHON" >&2
    echo "  Fix: set PYTHON=/path/to/venv/bin/python, or runtime.python in the recipe." >&2
    exit 1
fi
if ! "$PYTHON" -c "import sglang" >/dev/null 2>&1; then
    echo "ERROR: sglang not importable under $PYTHON" >&2
    echo "  Fix: set PYTHON=/path/to/venv with sglang, or runtime.python in the recipe." >&2
    exit 1
fi
# Put the venv's bin dir on PATH so FlashInfer's JIT can find `ninja`.
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
case ":$PATH:" in *":$PYTHON_BIN_DIR:"*) ;; *) export PATH="$PYTHON_BIN_DIR:$PATH" ;; esac

case "$MODE" in
    sharegpt)
        DETAILS_JSONL="$OUT_DIR/calib_sharegpt_n${NUM_PROMPTS}.jsonl"
        ;;
    openai)
        DETAILS_JSONL="$OUT_DIR/calib_${TASK}_n${NUM_PROMPTS}.jsonl"
        ;;
    bench_eval)
        DETAILS_JSONL="$OUT_DIR/calib_${TASK}_mc${CALIB_MC}.jsonl"
        ;;
esac
CALIB_JSON="$OUT_DIR/${TASK}.json"
SERVER_LOG="$OUT_DIR/server_${TASK}_bf16.log"

echo "============================================================"
echo "  KV calibration — task=$TASK  mode=$MODE  (FULL BF16, no heter config)"
case "$MODE" in
    sharegpt)
        echo "  bench_serving mc=$CALIB_MC num_prompts=$NUM_PROMPTS"
        ;;
    openai)
        echo "  bench_serving --dataset-name openai --dataset-path $PROMPTS_JSONL"
        echo "  num_prompts=$NUM_PROMPTS mc=$CALIB_MC (scoring is offline)"
        ;;
    bench_eval)
        echo "  bench_eval mc=$CALIB_MC fewshot=$NUM_FEWSHOT max_gen=$MAX_GEN_TOKS T=$TEMPERATURE limit=${CALIB_LIMIT:-full}"
        echo "  apply_chat_template=$APPLY_CHAT_TEMPLATE  fewshot_as_multiturn=$FEWSHOT_AS_MULTITURN"
        [ -n "$SYSTEM_INSTRUCTION" ] && echo "  sys_instr: $SYSTEM_INSTRUCTION"
        ;;
esac
echo "  server: gpu=$GPU port=$PORT"
echo "  → $DETAILS_JSONL"
echo "  → $CALIB_JSON"
echo "============================================================"

# 1. Launch server (full BF16 — no --heter-precision-config).
rm -f "$DETAILS_JSONL"
echo "Launching server..."
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m sglang.launch_server \
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
case "$MODE" in
    sharegpt)
        echo "Running bench_serving (sharegpt, n=$NUM_PROMPTS)..."
        "$PYTHON" -m sglang.bench_serving \
            --backend sglang \
            --base-url "http://${HOST}:${PORT}" \
            --dataset-name sharegpt \
            --sharegpt-context-len "$SHAREGPT_CONTEXT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CALIB_MC" \
            --output-details \
            --output-file "$DETAILS_JSONL"
        ;;
    openai)
        echo "Running bench_serving (openai, dataset-path=$PROMPTS_JSONL, n=$NUM_PROMPTS)..."
        "$PYTHON" -m sglang.bench_serving \
            --backend sglang-oai-chat \
            --base-url "http://${HOST}:${PORT}" \
            --dataset-name openai \
            --dataset-path "$PROMPTS_JSONL" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CALIB_MC" \
            --output-details \
            --output-file "$DETAILS_JSONL"
        ;;
    bench_eval)
        ct_flag=()
        [ "$APPLY_CHAT_TEMPLATE" = "1" ] && ct_flag=(--apply-chat-template)
        mt_flag=()
        [ "$FEWSHOT_AS_MULTITURN" = "1" ] && mt_flag=(--fewshot-as-multiturn)
        sys_flag=()
        [ -n "$SYSTEM_INSTRUCTION" ] && sys_flag=(--system-instruction "$SYSTEM_INSTRUCTION")
        lim_flag=()
        [ -n "$CALIB_LIMIT" ] && lim_flag=(--limit "$CALIB_LIMIT")
        echo "Running bench_eval (task=$TASK) with per-doc output..."
        "$PYTHON" -m sglang.bench_eval \
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
        ;;
esac

# 4. Run calib_kv to parse μ/σ of total_len (input+output) from the
#    per-request arrays.
echo "Running calib_kv..."
"$PYTHON" "$POLICY_DIR/calib_kv.py" \
    --bench_details_jsonl "$DETAILS_JSONL" \
    --out_file "$CALIB_JSON"

if [ "$MODE" = "bench_eval" ]; then
    echo ""
    echo "Accuracy from bench_eval (BF16 baseline):"
    "$PYTHON" -c "
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
elif [ "$MODE" = "openai" ]; then
    echo ""
    echo "Note: accuracy for MODE=openai is computed offline via scoring/score_traces_${TASK}.py"
    echo "      against $DETAILS_JSONL + prompts/${TASK}.meta.jsonl"
fi

echo "============================================================"
echo "  DONE. Calibration JSON: $CALIB_JSON"
echo "============================================================"
