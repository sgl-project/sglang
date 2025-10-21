#!/bin/bash
set -euo pipefail

# K2 preset
export MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-K2-Instruct}"
export QUANTIZATION="${QUANTIZATION:-fp8}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-kimi_k2}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
export TP_SIZE="${TP_SIZE:-16}"
export NNODES="${NNODES:-2}"

unset K2_PRESET # avoids inf loop

exec /app/entrypoint.sh
