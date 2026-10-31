#!/usr/bin/env bash
# Best validated MI355X profile: MXFP4 + kernel fusion + Plan B + FlyDSL MoE.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/data/Muse-Spark-1.2-816B-A42B-Quark-MXFP4}"
PORT="${PORT:-30000}"
ATTN="${ATTN:-aiter}"
CTX="${CTX:-32768}"
TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_USE_AITER_UNIFIED_ATTN="${SGLANG_USE_AITER_UNIFIED_ATTN:-1}"
export MUSE_MOE_BACKEND="${MUSE_MOE_BACKEND:-flydsl}"
export SGLANG_MUSE_FLYDSL_DECODE="${SGLANG_MUSE_FLYDSL_DECODE:-0}"
ATTN_ARGS=()
if [[ "${ATTN}" == "aiter" ]]; then
  ATTN_ARGS=(--page-size "${PAGE_SIZE:-16}")
elif [[ -n "${PAGE_SIZE:-}" ]]; then
  ATTN_ARGS=(--page-size "${PAGE_SIZE}")
fi

# Avoid resolving /sgl-workspace/aiter as a namespace package when the caller
# starts from /sgl-workspace.
cd /tmp
exec python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --tp-size "${TP_SIZE}" --ep-size "${EP_SIZE}" \
  --chat-template "${MODEL}/chat_template.jinja" \
  --load-format auto \
  --context-length "${CTX}" \
  --mem-fraction-static 0.90 \
  --attention-backend "${ATTN}" \
  "${ATTN_ARGS[@]}" \
  --host 0.0.0.0 --port "${PORT}" \
  "$@"
