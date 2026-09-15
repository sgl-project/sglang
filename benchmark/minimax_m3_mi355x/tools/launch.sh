#!/bin/bash
# Usage: TAG=... [MODEL=...] [EXTRA="..."] [ENVS="A=1 B=2"] bash launch.sh   (locations from env.sh)
source "$(dirname "$0")/env.sh"
: "${TAG:=run}" "${TP:=4}" "${PORT:=30000}" "${GPUS:=0,1,2,3}" "${MEMFRAC:=0.85}" "${CHUNK:=8192}" "${KV:=fp8_e4m3}" "${EXTRA:=}" "${ENVS:=}"
export PYTHONPATH=$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}
LOG=$LOGS_DIR/server_${TAG}.log
export HIP_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS
export SGLANG_USE_AITER=1
export HF_HUB_OFFLINE=1
for kv in $ENVS; do export "$kv"; done
cd "$SGLANG_DIR"
echo "GIT: $(git rev-parse HEAD) branch=$(git branch --show-current)" > "$LOG"
env | grep -E "^(SGLANG|AITER|ROCM|HIP|NCCL|RCCL)" >> "$LOG"
set -x
exec setsid "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name MiniMax-M3 \
  --trust-remote-code --tp $TP --port $PORT --host 0.0.0.0 \
  --kv-cache-dtype $KV \
  --chunked-prefill-size $CHUNK \
  --mem-fraction-static $MEMFRAC \
  --reasoning-parser auto --tool-call-parser auto \
  --enable-metrics --enable-cache-report \
  --watchdog-timeout 3600 \
  $EXTRA >> "$LOG" 2>&1
