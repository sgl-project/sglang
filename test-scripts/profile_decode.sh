#!/usr/bin/env bash
# Profile a RUNNING glm-sparse server (started by start_glm_sparse_server.sh).
# Focus: DECODE. Workload: input=16k output=1k (prefill ~1 step, rest is decode).
#
# Usage:
#   bash test-scripts/profile_decode.sh
#   PORT=30000 NUM_STEPS=80 CONC=16 bash test-scripts/profile_decode.sh
#
# Output: chrome traces (*.trace.json.gz) under $OUT_DIR, one per TP rank.
# View at https://ui.perfetto.dev or chrome://tracing
set -euo pipefail
source /root/paddlejob/inference-public/denghaodong/code/sglang/.venv/bin/activate

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
INPUT_LEN="${INPUT_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
CONC="${CONC:-16}"                 # concurrent decode requests during profiling
NUM_STEPS="${NUM_STEPS:-100}"      # forward steps to capture (decode steps)
OUT_DIR="${OUT_DIR:-/tmp/glm_profile/decode_$(date '+%Y%m%d_%H%M%S')}"
BASE="http://${HOST}:${PORT}"

mkdir -p "$OUT_DIR"
echo "[profile] out_dir=$OUT_DIR  in=$INPUT_LEN out=$OUTPUT_LEN conc=$CONC steps=$NUM_STEPS"

# 0) sanity: server alive
curl -sf "${BASE}/health" >/dev/null || { echo "server not ready on $BASE"; exit 1; }

# 1) fire background load FIRST so the server is in steady decode when profiling starts.
#    long output keeps it in decode for the whole profile window.
echo "[profile] launching background decode load ($CONC reqs)..."
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --dataset-name random-ids \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$CONC" \
  --max-concurrency "$CONC" \
  --seed 1 \
  > "$OUT_DIR/bench.log" 2>&1 &
BENCH_PID=$!

# 2) wait until requests have finished prefill and are decoding.
#    (prefill of 16k on TP8 fp8 takes a moment; give it a head start)
sleep "${WARMUP_SEC:-12}"

# 3) start profiler with num_steps -> auto-stops after NUM_STEPS forward steps.
#    activities CPU+GPU, no stack/shapes (lighter, cleaner decode trace).
echo "[profile] starting profiler for $NUM_STEPS steps..."
curl -sf -X POST "${BASE}/start_profile" \
  -H 'Content-Type: application/json' \
  -d "{
        \"output_dir\": \"${OUT_DIR}\",
        \"num_steps\": ${NUM_STEPS},
        \"activities\": [\"CPU\", \"GPU\"],
        \"with_stack\": false,
        \"record_shapes\": false
      }" && echo

# 4) num_steps auto-stops the profiler; wait for traces to flush, then for load to finish.
echo "[profile] profiler running (auto-stops after $NUM_STEPS steps); waiting for bench load to drain..."
wait "$BENCH_PID" || true
sleep 5

echo "[profile] DONE. traces:"
ls -lh "$OUT_DIR"/*.trace.json.gz 2>/dev/null || echo "  (no traces found — check server log for 'Profiling done')"
echo "[profile] open a *.trace.json.gz at https://ui.perfetto.dev"
