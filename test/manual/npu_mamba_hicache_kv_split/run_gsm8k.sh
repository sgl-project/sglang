#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
export HICACHE_RATIO="${HICACHE_RATIO:-8}"
source "$HERE/env.sh"
export TP_SIZE="$(awk -F, '{print NF}' <<< "$NPU_DEVICES")"

export GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:?set GSM8K_TRAIN_PATH to train.jsonl}"
export GSM8K_TEST_PATH="${GSM8K_TEST_PATH:?set GSM8K_TEST_PATH to test.jsonl}"
export GSM8K_NUM_QUESTIONS="${GSM8K_NUM_QUESTIONS:-200}"
# 8-shot keeps prompts long enough for full KV pages and Mamba checkpoints on
# the track grid; UNIQUE_PAD_TOKENS then appends a per-question unique pad so
# every question owns evictable unique pages (see unique_pad in the test).
export GSM8K_NUM_SHOTS="${GSM8K_NUM_SHOTS:-8}"
export UNIQUE_PAD_TOKENS="${UNIQUE_PAD_TOKENS:-640}"
export GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
export PAGE_SIZE="${PAGE_SIZE:-128}"
export FILLER_PROMPT_TOKENS="${FILLER_PROMPT_TOKENS:-4096}"
export MAX_FILLER_REQUESTS="${MAX_FILLER_REQUESTS:-12}"

OUT=${1:?usage: bash run_gsm8k.sh <output_dir>}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
SERVER_PID=""
cleanup() {
  set +e
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  echo "结果目录: $OUT"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend \
  --tp-size "$TP_SIZE" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --page-size "$PAGE_SIZE" \
  --context-length "$CONTEXT_LENGTH" --mamba-ssm-dtype bfloat16 \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through --hicache-io-backend kernel_ascend \
  --enable-metrics --disable-cuda-graph > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for ((i=0; i<STARTUP_TIMEOUT; i++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if (( ! ready )); then
  echo "server did not become ready; see $OUT/server.log" >&2
  exit 1
fi

# TEST_CASE selects which client runs against the same server:
#   gsm8k   (default): standard few-shot prompts; evict before each replay,
#                      require Host hit, Mamba H2D and the same parsed answer.
#   stress            : per-question A0->A->E->R with cache flushes between
#                      questions (see the stress script docstring; unique
#                      padding, not a standard accuracy baseline).
TEST_CASE="${TEST_CASE:-gsm8k}"
case "$TEST_CASE" in
  gsm8k)
    CLIENT="$HERE/test_qwen3_5_4b_gsm8k_mamba_hicache.py" ;;
  stress)
    CLIENT="$HERE/test_qwen3_5_4b_host_reload_stress.py" ;;
  *)
    echo "unknown TEST_CASE: $TEST_CASE" >&2; exit 1 ;;
esac

python3 -u "$CLIENT" \
  --output "$OUT/result.json" > "$OUT/eval.log" 2>&1 || {
    cat "$OUT/eval.log"
    exit 1
  }
cat "$OUT/eval.log"
