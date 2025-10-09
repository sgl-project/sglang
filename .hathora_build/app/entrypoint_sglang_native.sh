#!/bin/bash
set -euo pipefail

##############################################################################
# SGLang Native Server Entrypoint for Hathora
# 
# This script uses SGLang's built-in HTTP server instead of custom implementation
# Benefits:
# - Full OpenAI API compatibility
# - Better performance and features
# - Less maintenance overhead
# - Production-proven infrastructure
##############################################################################

echo "=================================================="
echo "SGLang Native Server for Hathora"
echo "=================================================="

# Load deployment configuration
if [ -n "${DEPLOYMENT_CONFIG_JSON:-}" ]; then
    echo "Loading configuration from DEPLOYMENT_CONFIG_JSON..."
    export MODEL_PATH=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.model_id // .model_path // ""')
    export TP_SIZE=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.tp_size // 1')
    export DTYPE=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.dtype // "auto"')
    export QUANTIZATION=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.quantization // ""')
    export KV_CACHE_DTYPE=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.kv_cache_dtype // "auto"')
    export MAX_TOTAL_TOKENS=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.max_total_tokens // ""')
    export HF_TOKEN=$(echo "$DEPLOYMENT_CONFIG_JSON" | jq -r '.hf_token // ""')
fi

# Required variables
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Optional variables
QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-1.0}"
LOG_LEVEL="${LOG_LEVEL:-info}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

# GLM-specific settings
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-}"

# Auto-detect GLM model settings
MODEL_PATH_LOWER=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')

# Note: GLM models use their built-in HuggingFace tokenizer chat template by default.
# Only set custom chat template if explicitly provided via CHAT_TEMPLATE env var.

# Auto-detect tool parser for GLM-4.5/4.6
if [[ "$MODEL_PATH_LOWER" == *"glm-4.5"* ]] || [[ "$MODEL_PATH_LOWER" == *"glm-4.6"* ]]; then
    if [ -z "$TOOL_CALL_PARSER" ]; then
        TOOL_CALL_PARSER="glm45"
        echo "Auto-detected GLM-4.5/4.6 model, setting tool-call-parser=glm45"
    fi
fi

# Embedding mode
IS_EMBEDDING="${IS_EMBEDDING:-false}"
if [[ "$MODEL_PATH_LOWER" == *"embedding"* ]]; then
    IS_EMBEDDING="true"
    echo "Auto-detected embedding model from path"
fi

# Speculative decoding
SPEC_DECODE="${SPEC_DECODE:-false}"
SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-}"
SPECULATIVE_DRAFT_MODEL_PATH="${SPECULATIVE_DRAFT_MODEL_PATH:-}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-1}"

# Memory settings
# Match serve_hathora.py behavior: expandable_segments=true (default) disables memory_saver
ENABLE_EXPANDABLE_SEGMENTS="${ENABLE_EXPANDABLE_SEGMENTS:-true}"
ENABLE_MEMORY_SAVER="${ENABLE_MEMORY_SAVER:-true}"

# If expandable segments enabled, disable memory saver (they conflict)
if [ "$ENABLE_EXPANDABLE_SEGMENTS" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    ENABLE_MEMORY_SAVER="false"
    echo "Note: Expandable segments enabled → memory saver disabled (avoids conflict)"
fi

# Set HF token
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

# Log configuration
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  TP Size: $TP_SIZE"
echo "  Data Type: $DTYPE"
echo "  Quantization: ${QUANTIZATION:-none}"
echo "  KV Cache dtype: $KV_CACHE_DTYPE"
echo "  Max Total Tokens: ${MAX_TOTAL_TOKENS:-auto}"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"
echo "  Metrics: $ENABLE_METRICS"
echo "  Chat Template: ${CHAT_TEMPLATE:-default}"
echo "  Tool Call Parser: ${TOOL_CALL_PARSER:-default}"
echo "  Embedding Mode: $IS_EMBEDDING"
echo "  Expandable Segments: $ENABLE_EXPANDABLE_SEGMENTS"
echo "  Memory Saver: $ENABLE_MEMORY_SAVER"
echo "  CUDA Graph Max BS: ${CUDA_GRAPH_MAX_BS:-auto}"
echo "=================================================="

# Build command arguments
ARGS=(
    --model-path "$MODEL_PATH"
    --tp-size "$TP_SIZE"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --host "$HOST"
    --port "$PORT"
    --log-level "$LOG_LEVEL"
    --schedule-conservativeness "$SCHEDULE_CONSERVATIVENESS"
    --skip-server-warmup
    --scheduler-recv-interval 0
)

# Optional arguments
[ -n "$QUANTIZATION" ] && ARGS+=(--quantization "$QUANTIZATION")
[ -n "$MAX_TOTAL_TOKENS" ] && ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
[ -n "$MEM_FRACTION_STATIC" ] && ARGS+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
[ -n "$MAX_RUNNING_REQUESTS" ] && ARGS+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
[ -n "$CUDA_GRAPH_MAX_BS" ] && ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
[ -n "$CHUNKED_PREFILL_SIZE" ] && ARGS+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
[ -n "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template "$CHAT_TEMPLATE")
[ -n "$TOOL_CALL_PARSER" ] && ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")

# Embedding mode
if [ "$IS_EMBEDDING" = "true" ]; then
    ARGS+=(--is-embedding --disable-radix-cache)
fi

# Metrics
if [ "$ENABLE_METRICS" = "true" ]; then
    ARGS+=(--enable-metrics)
fi

# Trust remote code
if [ "$TRUST_REMOTE_CODE" = "true" ]; then
    ARGS+=(--trust-remote-code)
fi

# Memory settings  
# Only enable memory saver if explicitly requested (can interfere with CUDA graph)
if [ "$ENABLE_MEMORY_SAVER" = "true" ]; then
    ARGS+=(--enable-memory-saver)
    echo "Note: Memory saver enabled (may affect CUDA graph for some models)"
fi

# Speculative decoding
if [ "$SPEC_DECODE" = "true" ] && [ -n "$SPECULATIVE_ALGORITHM" ]; then
    ARGS+=(--speculative-algorithm "$SPECULATIVE_ALGORITHM")
    [ -n "$SPECULATIVE_DRAFT_MODEL_PATH" ] && ARGS+=(--speculative-draft-model-path "$SPECULATIVE_DRAFT_MODEL_PATH")
    ARGS+=(--speculative-num-steps "$SPECULATIVE_NUM_STEPS")
fi

# Disaggregation
[ -n "${DISAGGREGATION_DECODE_TP:-}" ] && ARGS+=(--disaggregation-decode-tp "$DISAGGREGATION_DECODE_TP")
[ -n "${DISAGGREGATION_PREFILL_PP:-}" ] && ARGS+=(--disaggregation-prefill-pp "$DISAGGREGATION_PREFILL_PP")

# Custom all-reduce
ARGS+=(--disable-custom-all-reduce)

echo "Launching SGLang server..."
echo "Command: python -m sglang.launch_server ${ARGS[*]}"
echo "=================================================="

# Execute server
exec python -m sglang.launch_server "${ARGS[@]}"

