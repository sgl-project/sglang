#!/usr/bin/env bash
# Run inside the candidate SGLang image, on an explicitly selected GPU host.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-M3 checkpoint path or model ID}"
mode="${MODE:-planned}"
export SGLANG_USE_AITER=1
# This recipe measures native SGLang. A nonempty unmatched whitelist disables
# installed general plugins; an empty value means load-all in this revision.
export SGLANG_PLUGINS=none
unset SGLANG_PLATFORM
# The integration sets main KV explicitly and keeps index KV in NHD.
export SGLANG_AITER_KV_CACHE_LAYOUT=nhd
case "${mode}" in
  baseline)
    export SGLANG_MINIMAX_FLYDSL_DECODE=0
    export SGLANG_MINIMAX_FLYDSL_PLAN=0
    ;;
  static)
    export SGLANG_MINIMAX_FLYDSL_DECODE=1
    export SGLANG_MINIMAX_FLYDSL_PLAN=0
    ;;
  planned)
    export SGLANG_MINIMAX_FLYDSL_DECODE=1
    export SGLANG_MINIMAX_FLYDSL_PLAN=1
    ;;
  *) echo 'MODE must be baseline, static, or planned' >&2; exit 2 ;;
esac
echo "MiniMax-M3 decode mode: ${mode}"
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --tp "${TP:-4}" \
  --dtype bfloat16 \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --page-size "${PAGE_SIZE:-16}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
  --disable-radix-cache \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30000}" \
  "$@"
