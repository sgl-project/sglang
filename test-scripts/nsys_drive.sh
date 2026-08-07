#!/usr/bin/env bash
# Drive an nsys-wrapped server (started by nsys_server.sh) through ONE decode
# capture window. Uses activities=[CUDA_PROFILER] so /start_profile triggers
# cudaProfilerStart() (opens nsys capture range) and /stop_profile closes it.
#
# Workload: input=16k output=1k, decode-focused (matches the torch-profile run).
#
#   MODE=sparse bash test-scripts/nsys_drive.sh
#   MODE=dense  bash test-scripts/nsys_drive.sh
set -euo pipefail
source /root/paddlejob/inference-public/denghaodong/code/sglang/.venv/bin/activate

MODE="${MODE:?MODE required (dense|sparse) - just for logging}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
INPUT_LEN="${INPUT_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
CONC="${CONC:-16}"
WARMUP_SEC="${WARMUP_SEC:-12}"      # let 16k prefill drain -> steady decode
CAPTURE_SEC="${CAPTURE_SEC:-15}"    # how long to keep the nsys window open
BASE="http://${HOST}:${PORT}"

# 0) wait for server readiness (nsys + TP8 + fp8 loads slowly)
echo "[nsys_drive] mode=$MODE waiting for server on $BASE ..."
for ((i=0;i<1200;i+=5)); do
  if curl -sf "${BASE}/health" >/dev/null 2>&1; then echo "[nsys_drive] ready (~${i}s)"; break; fi
  sleep 5
done
curl -sf "${BASE}/health" >/dev/null || { echo "server not ready"; exit 1; }

# 1) background decode load (long output keeps it decoding through the window)
echo "[nsys_drive] launching decode load (conc=$CONC in=$INPUT_LEN out=$OUTPUT_LEN)"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --dataset-name random-ids \
  --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$CONC" --max-concurrency "$CONC" --seed 1 \
  > "/tmp/glm_nsys/bench_${MODE}.log" 2>&1 &
BENCH_PID=$!

# 2) wait for steady decode, then OPEN nsys capture range
sleep "$WARMUP_SEC"
echo "[nsys_drive] start_profile (opens nsys cudaProfiler range)"
curl -sf -X POST "${BASE}/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"activities": ["CUDA_PROFILER"]}' && echo

# 3) hold the window over a chunk of decode steps
sleep "$CAPTURE_SEC"

# 4) CLOSE nsys capture range -> server keeps running (capture-range-end=stop)
echo "[nsys_drive] stop_profile (closes nsys range)"
curl -sf -X POST "${BASE}/stop_profile" && echo

# 5) let load finish; nsys flushes the .nsys-rep when capture-range ends
wait "$BENCH_PID" || true
echo "[nsys_drive] DONE. report: /tmp/glm_nsys/glm_${MODE}.nsys-rep"
echo "[nsys_drive] kill the nsys server (Ctrl-C / pkill) so it finalizes the .nsys-rep"
