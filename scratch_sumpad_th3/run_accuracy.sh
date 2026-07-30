#!/usr/bin/env bash
# Accuracy check for the MAX-prefill-padding UB hypothesis.
#
# PR #10414 ("Fix cutlass moe accuracy drop caused by attention UB from DP padding
# mode") avoided MAX_LEN on extend batches. The breakable prefill CUDA graph force
# (#30898) re-introduces MAX_LEN on extend, and additionally marks the fabricated
# idle-rank dummy rows as real via num_token_non_padded.fill_(num_tokens), which
# disables the MoE padded-region masking in layers/moe/topk.py. This script tests
# whether that costs accuracy on an FP8 MoE model with moe_ep_size > 1 (the only
# configuration where num_token_non_padded is not None).
set -uo pipefail

# DeepSeek-V2-Lite-Chat is MLA + MoE, which matches the production shape the original
# accuracy report came from, and its weights are actually present in the cluster cache
# (the Qwen3-30B-A3B-FP8 entry there is a config-only stub).
MODEL=${MODEL:-/cluster-storage/models/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}
OUT_ROOT=${OUT_ROOT:-/scratch/sumpad_acc}
PORT=${PORT:-32100}
BASE="http://127.0.0.1:${PORT}"
NUM_QUESTIONS=${NUM_QUESTIONS:-400}
NUM_SHOTS=${NUM_SHOTS:-5}

CONFIG=${1:?usage: run_accuracy.sh <config-name>}

case "$CONFIG" in
  # Current default: breakable prefill CG, so extend batches that fit a graph
  # bucket get MAX_LEN forced and idle ranks fabricate full-length dummy rows.
  default-breakable) GRAPH_ARGS=(); DBG_MODE="" ;;
  # What #10414 guarantees today on the eager path: SUM_LEN, no fabrication.
  sum-disabled)      GRAPH_ARGS=(--cuda-graph-backend-prefill disabled); DBG_MODE="sum" ;;
  # What deleting the #10414 condition would produce on the eager path.
  max-disabled)      GRAPH_ARGS=(--cuda-graph-backend-prefill disabled); DBG_MODE="max" ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

OUT="$OUT_ROOT/$CONFIG"
mkdir -p "$OUT"

exec_command() {
  echo "+ $*" | tee -a "$OUT/commands.log"
  "$@"
}

export SGLANG_DBG_DP_PAD_MODE="$DBG_MODE"
export SGLANG_DBG_DP_PAD_LOG=1
echo "config=$CONFIG SGLANG_DBG_DP_PAD_MODE='$DBG_MODE' graph_args='${GRAPH_ARGS[*]}'" | tee "$OUT/env.log"

# launch_server spawns scheduler children that outlive a plain kill of the parent, so
# clear any leftover server on this port before binding it again.
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
sleep 20

exec_command python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --tp 8 --dp 8 --enable-dp-attention \
  --ep-size 8 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.82 \
  --host 127.0.0.1 --port "$PORT" \
  "${GRAPH_ARGS[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
  if curl -sf "$BASE/health_generate" > /dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SERVER_DIED see $OUT/server.log" | tee -a "$OUT/env.log"; exit 1
  fi
  sleep 10
done

run_gsm8k() {
  local label=$1 parallel=$2
  # Low parallelism is the point: it leaves DP ranks idle during prefill, which is
  # exactly when MAX_LEN fabricates dummy rows. High parallelism is the control.
  exec_command python -m sglang.test.few_shot_gsm8k \
    --num-questions "$NUM_QUESTIONS" \
    --num-shots "$NUM_SHOTS" \
    --parallel "$parallel" \
    --port "$PORT" > "$OUT/gsm8k_${label}.log" 2>&1
  echo "$label accuracy: $(grep -iE '^accuracy|Accuracy' "$OUT/gsm8k_${label}.log" | tail -2 | tr '\n' ' ')"
}

run_gsm8k skewed 4
run_gsm8k dense 64

grep -c "\[DPPAD\]" "$OUT/server.log" > "$OUT/dppad_lines.txt" 2>/dev/null
echo "DONE" > "$OUT/DONE"
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
