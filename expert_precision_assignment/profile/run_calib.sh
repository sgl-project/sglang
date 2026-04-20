#!/usr/bin/env bash
# KV calibration pass for a task.
#
# Runs one bench_serving (sharegpt) or bench_eval (gsm8k/mmlu/...) pass
# with --output-details, captures per-request (input_len, output_len),
# then invokes calib_kv.py to emit kv_calib/<task>.json with μ/σ of
# total_len. gen_heter_configs.py --calib_json consumes that file.
#
# Usage:
#   bash run_calib.sh sharegpt
#   CALIB_MC=64 bash run_calib.sh sharegpt
#   NUM_PROMPTS=512 bash run_calib.sh sharegpt
set -uo pipefail

TASK="${1:-}"
if [ -z "$TASK" ]; then
    echo "Usage: bash run_calib.sh <task>   (e.g. sharegpt)" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$(cd "$SCRIPT_DIR/../policy/heter_assign" && pwd)"
CFG_DIR="$SCRIPT_DIR/configs"
OUT_DIR="$SCRIPT_DIR/kv_calib"
mkdir -p "$OUT_DIR"

BF16_MODEL="/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
HOST="127.0.0.1"
PORT="${CALIB_PORT:-31304}"
NCCL_PORT="${CALIB_NCCL_PORT:-41304}"
GPU="${CALIB_GPU:-4}"
CALIB_MC="${CALIB_MC:-128}"
CALIB_VARIANT="${CALIB_VARIANT:-thr128}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-4096}"

# shellcheck disable=SC1091
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate sglang 2>/dev/null || source activate sglang

if [ "$TASK" != "sharegpt" ]; then
    echo "run_calib.sh currently supports sharegpt only (bench_serving --output-details path)." >&2
    echo "For bench_eval tasks (gsm8k, mmlu), use calib_kv.py --task=<task> for prompt stats." >&2
    exit 2
fi

CFG="$CFG_DIR/mc${CALIB_MC}/variants/${CALIB_VARIANT}.json"
if [ ! -f "$CFG" ]; then
    echo "Missing calibration config: $CFG" >&2
    echo "Run gen_heter_configs.py first (worst-case is fine for calibration)." >&2
    exit 1
fi

DETAILS_JSONL="$OUT_DIR/calib_mc${CALIB_MC}_${CALIB_VARIANT}_n${NUM_PROMPTS}.jsonl"
CALIB_JSON="$OUT_DIR/${TASK}.json"
SERVER_LOG="$OUT_DIR/server_mc${CALIB_MC}_${CALIB_VARIANT}.log"

echo "============================================================"
echo "  KV calibration — task=$TASK"
echo "  mc=$CALIB_MC variant=$CALIB_VARIANT num_prompts=$NUM_PROMPTS"
echo "  server: gpu=$GPU port=$PORT"
echo "  → $DETAILS_JSONL"
echo "  → $CALIB_JSON"
echo "============================================================"

# 1. Launch server
rm -f "$DETAILS_JSONL"
echo "Launching server..."
CUDA_VISIBLE_DEVICES="$GPU" python3 -m sglang.launch_server \
    --model-path "$BF16_MODEL" \
    --host "$HOST" --port "$PORT" \
    --nccl-port "$NCCL_PORT" \
    --trust-remote-code \
    --max-running-requests "$CALIB_MC" \
    --cuda-graph-max-bs "$CALIB_MC" \
    --heter-precision-config "$CFG" > "$SERVER_LOG" 2>&1 &
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

# 3. Run bench_serving with --output-details
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

# 4. Run calib_kv to parse μ/σ
echo "Running calib_kv..."
python3 "$POLICY_DIR/calib_kv.py" \
    --bench_details_jsonl "$DETAILS_JSONL" \
    --out_file "$CALIB_JSON"

echo "============================================================"
echo "  DONE. Calibration JSON: $CALIB_JSON"
echo "============================================================"
