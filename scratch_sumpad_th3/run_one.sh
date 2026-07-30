#!/bin/bash
# usage: run_one.sh <variant: default|no10414|sum|max> <graph: breakable|disabled> <phase: bench|evidence|profile> <outdir>
set -x

VARIANT=$1
GRAPH=$2
PHASE=$3
OUT=$4

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-31500}
UNIFORM_BS=${UNIFORM_BS:-64}
ISL=${ISL:-2048}
SKEW_ISL=${SKEW_ISL:-16384}
SERVE_NUM_PROMPTS_UNIFORM=${SERVE_NUM_PROMPTS_UNIFORM:-256}
SERVE_NUM_PROMPTS_SKEW=${SERVE_NUM_PROMPTS_SKEW:-16}

mkdir -p "$OUT"

case "$VARIANT" in
  default) PAD_MODE="" ;;
  *) PAD_MODE="$VARIANT" ;;
esac

DBG_LOG=0
DBG_PROBE=0
if [ "$PHASE" = "evidence" ]; then
  DBG_LOG=1
  DBG_PROBE=1
fi

export SGLANG_DBG_DP_PAD_MODE="$PAD_MODE"
export SGLANG_DBG_DP_PAD_LOG="$DBG_LOG"
export SGLANG_DBG_SHAPE_PROBE="$DBG_PROBE"
export SGLANG_TORCH_PROFILER_DIR="$OUT/profile"
export HF_HOME=${HF_HOME:-/cluster-storage/models}
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"

python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 --dp 8 --enable-dp-attention \
  --disable-radix-cache \
  --cuda-graph-backend-prefill "$GRAPH" \
  --mem-fraction-static 0.85 \
  --host 127.0.0.1 --port "$PORT" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    break
  fi
  sleep 10
done

if [ "$READY" != "1" ]; then
  echo "SERVER_NOT_READY" | tee "$OUT/RESULT_NOTREADY"
  kill -9 $SERVER_PID 2>/dev/null
  sleep 20
  pkill -9 -f "sglang::" 2>/dev/null
  echo "DONE_MARKER phase=$PHASE variant=$VARIANT graph=$GRAPH status=notready"
  exit 0
fi

BASE="http://127.0.0.1:$PORT"

server_alive () {
  curl -s --max-time 20 "http://127.0.0.1:$PORT/health_generate" > /dev/null 2>&1
}

run_bench_one_batch () {
  local name=$1
  local bs=$2
  local isl=$3
  shift 3
  if ! server_alive; then
    echo "SKIP bob_${name}: server not alive" | tee -a "$OUT/SKIPPED"
    return
  fi
  timeout 420 python3 -m sglang.benchmark.one_batch_server \
    --model None --base-url "$BASE" \
    --batch-size "$bs" --input-len "$isl" --output-len 1 \
    --show-report \
    "$@" \
    > "$OUT/bob_${name}.log" 2>&1
  echo "bob_${name} exit=$?"
}

run_bench_serving () {
  local name=$1
  local conc=$2
  local nprompts=$3
  local isl=$4
  if ! server_alive; then
    echo "SKIP serving_${name}: server not alive" | tee -a "$OUT/SKIPPED"
    return
  fi
  timeout 600 python3 -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --dataset-name random --random-input-len "$isl" --random-output-len 1 \
    --random-range-ratio 1.0 \
    --num-prompts "$nprompts" --max-concurrency "$conc" \
    > "$OUT/serving_${name}.log" 2>&1
  echo "serving_${name} exit=$?"
}

if [ "$PHASE" = "bench" ]; then
  run_bench_one_batch uniform "$UNIFORM_BS" "$ISL"
  run_bench_one_batch skew 1 "$SKEW_ISL"
  run_bench_serving uniform 64 "$SERVE_NUM_PROMPTS_UNIFORM" "$ISL"
  run_bench_serving skew 1 "$SERVE_NUM_PROMPTS_SKEW" "$SKEW_ISL"
elif [ "$PHASE" = "evidence" ]; then
  run_bench_one_batch uniform "$UNIFORM_BS" "$ISL"
  run_bench_one_batch skew 1 "$SKEW_ISL"
elif [ "$PHASE" = "profile" ]; then
  run_bench_one_batch uniform_prof "$UNIFORM_BS" "$ISL" --profile --profile-activities GPU --profile-by-stage
  run_bench_one_batch skew_prof 1 "$SKEW_ISL" --profile --profile-activities GPU --profile-by-stage
fi

curl -s "$BASE/flush_cache" > /dev/null 2>&1
kill -INT $SERVER_PID 2>/dev/null
sleep 30
kill -9 $SERVER_PID 2>/dev/null
sleep 10
pkill -9 -f "sglang::" 2>/dev/null
sleep 5
echo "DONE_MARKER phase=$PHASE variant=$VARIANT graph=$GRAPH status=ok"
