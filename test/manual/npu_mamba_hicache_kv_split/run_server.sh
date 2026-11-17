#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
TP=$(awk -F, '{print NF}' <<< "$NPU_DEVICES")

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu \
  --tp-size "$TP" \
  --enable-hierarchical-cache \
  --enable-metrics \
  --hicache-ratio "$HICACHE_RATIO" \
  --hicache-io-backend kernel_ascend \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --mamba-ssm-dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
