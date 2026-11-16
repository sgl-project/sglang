#!/usr/bin/env bash
# Startup cost of pauseable breakable-graph capture: launch the CuTe DSL NVFP4
# reload-suite server with and without SGLANG_MEMORY_SAVER_CUDA_GRAPH, record the
# target prefill/decode capture lines, then shut the server down.
#   bash capture_time_ab.sh <gpus> <run name> [rounds]
set -uo pipefail
GPUS=${1:?gpus}
RUN=${2:?run name}
ROUNDS=${3:-2}
ROOT=${ROOT:-/hai-workspace/moe-prefill-cg}
TREE=${TREE:-sglang}
LOG_DIR=$ROOT/logs/$RUN
mkdir -p "$LOG_DIR"
cd "$ROOT/$TREE"
export PYTHONPATH="$ROOT/$TREE/python"
export CUDA_VISIBLE_DEVICES=$GPUS
export SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=1
export FLASHINFER_NVFP4_4OVER6=1 FLASHINFER_NVFP4_4OVER6_ERR_MODE=MSE
export FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH=1 FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1
PORT=$((31000 + ${GPUS:0:1} * 100))

for round in $(seq 1 "$ROUNDS"); do
  for saver in 0 1; do
    log=$LOG_DIR/round$round-saver$saver.log
    SGLANG_MEMORY_SAVER_CUDA_GRAPH=$saver python3 -m sglang.launch_server \
      --model-path nvidia/Qwen3-30B-A3B-NVFP4 --tp-size 4 --ep-size 4 \
      --fp4-gemm-backend flashinfer_cutedsl --moe-runner-backend flashinfer_cutedsl \
      --moe-a2a-backend none --enable-deterministic-inference --enable-memory-saver \
      --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 &
    pid=$!
    until grep -q "The server is fired up\|Traceback" "$log"; do sleep 5; done
    kill -TERM "$pid"; wait "$pid"
    sleep 20
    echo "round=$round saver=$saver"
    grep -a "TP0.*Capture target .* CUDA graph end" "$log" | sed 's/.*Capture/Capture/'
  done
done | tee "$LOG_DIR/summary.txt"
