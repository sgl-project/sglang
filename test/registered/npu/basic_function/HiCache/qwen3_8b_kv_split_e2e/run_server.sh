#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
LAYOUT=${1:?usage: bash run_server.sh layer_first|page_first}

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --trust-remote-code \
  --device npu --attention-backend ascend \
  --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" --page-size "$PAGE_SIZE" \
  --enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel_ascend --hicache-mem-layout "$LAYOUT" \
  --enable-metrics --disable-cuda-graph &
SERVER_PID=$!
trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
wait "$SERVER_PID"
