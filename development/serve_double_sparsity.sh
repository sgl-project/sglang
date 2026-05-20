#!/usr/bin/env bash
# Reference SGLang server invocation for the standalone Double Sparsity path.
#
# Mirrors development/serve_native_nsa.sh but adds --enable-double-sparsity
# and --double-sparsity-config. Targets DeepSeek-V3.2 (FP8) on a single H200
# node, 8-way TP, page=64. Mutually exclusive with --enable-hisparse at
# startup (per DEC-8).
#
# The selection kernels and FP8-aware page_signature_write are still under
# development. SGLANG_DS_ALLOW_PLACEHOLDER=1 lets a server boot with the
# placeholder selector for development / smoke tests; production traffic
# must wait for the real kernels (later milestones).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3.2}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CHANNEL_MASK_PATH="${CHANNEL_MASK_PATH:-/models/dsv32-fp8-channel-mask.safetensors}"
TOP_K="${TOP_K:-2048}"
DEVICE_BUFFER_SIZE="${DEVICE_BUFFER_SIZE:-4096}"
LOG_DIR="${LOG_DIR:-$(pwd)/development/logs}"
mkdir -p "${LOG_DIR}"

DS_CONFIG=$(printf '{"top_k": %s, "page_size": %s, "channel_mask_path": "%s", "device_buffer_size": %s}' \
  "${TOP_K}" "${PAGE_SIZE}" "${CHANNEL_MASK_PATH}" "${DEVICE_BUFFER_SIZE}")

LOG_FILE="${LOG_DIR}/server_double_sparsity_$(date +%Y%m%d-%H%M%S).log"
echo ">>> Starting Double Sparsity server (standalone)"
echo "    model            = ${MODEL_PATH}"
echo "    port             = ${PORT}"
echo "    tp_size          = ${TP_SIZE}"
echo "    kv_cache         = ${KV_CACHE_DTYPE}"
echo "    page_size        = ${PAGE_SIZE}"
echo "    channel_mask     = ${CHANNEL_MASK_PATH}"
echo "    top_k            = ${TOP_K}"
echo "    device_buffer    = ${DEVICE_BUFFER_SIZE}"
echo "    log              = ${LOG_FILE}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --page-size "${PAGE_SIZE}" \
  --enable-double-sparsity \
  --double-sparsity-config "${DS_CONFIG}" \
  --trust-remote-code \
  2>&1 | tee "${LOG_FILE}"
