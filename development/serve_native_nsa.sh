#!/usr/bin/env bash
# Reference SGLang server invocation for the native_nsa baseline.
#
# Targets DeepSeek-V3.2 (FP8) on a single H200 node, 8-way TP, page=64,
# matching the operating point in DEC-1 (h200-10-220-51-6 or
# h200-10-220-51-8). Pair with development/benchmark_baseline.sh to
# populate the native_nsa column of the two-column comparison.
#
# Double Sparsity is intentionally NOT enabled here — the baseline runs
# unmodified NSA selection so the comparison row pair is honest.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3.2}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-64}"
LOG_DIR="${LOG_DIR:-$(pwd)/development/logs}"

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/server_native_nsa_$(date +%Y%m%d-%H%M%S).log"
echo ">>> Starting native_nsa baseline server"
echo "    model        = ${MODEL_PATH}"
echo "    port         = ${PORT}"
echo "    tp_size      = ${TP_SIZE}"
echo "    kv_cache     = ${KV_CACHE_DTYPE}"
echo "    page_size    = ${PAGE_SIZE}"
echo "    log          = ${LOG_FILE}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --page-size "${PAGE_SIZE}" \
  --trust-remote-code \
  2>&1 | tee "${LOG_FILE}"
