#!/bin/bash
# Launch SGLang server with attention token capture enabled
#
# This script launches the Qwen3-Next-80B model with all necessary flags
# for attention capture and fingerprint generation.
#
# Usage:
#   ./launch_server_with_attention.sh        # Default settings
#   ./launch_server_with_attention.sh --fp8  # Use FP8 model
#   ./launch_server_with_attention.sh --small # Use smaller model for testing

set -e

# Default settings
MODEL="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"
PORT=8000
TOP_K=32
ATTENTION_WINDOW=0  # 0 = all tokens, or set to limit for long context

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --small)
            # Use smaller model for quick testing
            MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            shift
            ;;
        --qwen-7b)
            MODEL="Qwen/Qwen2.5-7B-Instruct"
            shift
            ;;
        --qwen-72b)
            MODEL="Qwen/Qwen2.5-72B-Instruct"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --attention-window)
            ATTENTION_WINDOW="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--small|--qwen-7b|--qwen-72b] [--port PORT] [--top-k N] [--attention-window N]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Launching SGLang Server with Attention Capture"
echo "============================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Attention Top-K: $TOP_K"
echo "Attention Window: $ATTENTION_WINDOW"
echo "============================================================"
echo ""

# Launch server with attention capture
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
python3 -m sglang.launch_server \
    --model "$MODEL" \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 18 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --mem-fraction-static 0.89 \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
    --context-length 1010000 \
    --tool-call-parser qwen \
    --host 0.0.0.0 \
    --port "$PORT" \
    --kv-cache-dtype bfloat16 \
    --max-running-requests 16 \
    --disable-radix-cache \
    \
    --return-attention-tokens \
    --attention-tokens-top-k "$TOP_K" \
    --attention-tokens-max 4096 \
    --attention-tokens-stride 1 \
    --attention-tokens-window "$ATTENTION_WINDOW" \
    --attention-capture-layers last
