#!/bin/bash

# ==============================================================================
# SGLang Startup Script for MiniMax-M2.5-AWQ on 2x A800 GPUs
# Optimization Target: 192k Context Length & High Throughput
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Environment Variables for MoE & Performance
# ------------------------------------------------------------------------------

# Use DeepEP for optimized All-to-All communication (Crucial for MoE EP)
export SGLANG_MOE_A2A_BACKEND=deepep

# Optimize DeepEP dispatch for Ampere (A800) using BF16
export SGLANG_DEEPEP_BF16_DISPATCH=1

# Increase max dispatch tokens per rank for better throughput in long-context
# Default is 128, increased to 256 for 192k context loads
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# ------------------------------------------------------------------------------
# 2. Environment Variables for Long Context Stability
# ------------------------------------------------------------------------------

# Increase safety factor for RoPE cache expansion to prevent frequent reallocations
# Recommended 3-4 for >100k context
export SGLANG_SPEC_EXPANSION_SAFETY_FACTOR=3

# Reserve larger safety margin for RoPE cache
export SGLANG_ROPE_CACHE_SAFETY_MARGIN=1024

# ------------------------------------------------------------------------------
# 3. Compilation & Kernel Optimizations
# ------------------------------------------------------------------------------

# Enable PyTorch Compile (Torch 2.0+) for graph capture and fusion
# Expect some warmup latency on the first few requests
export SGLANG_ENABLE_TORCH_COMPILE=1

# Use custom Triton kernel cache to speed up JIT and reduce overhead
export SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE=1

# ------------------------------------------------------------------------------
# 4. Launch Server
# ------------------------------------------------------------------------------

# Set your actual model path here
MODEL_PATH="/path/to/minimax-m2.5-awq"

echo "Starting SGLang Server for $MODEL_PATH on 2 GPUs..."

python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp-size 1 \
  --ep-size 2 \
  --quantization awq \
  --kv-cache-dtype bf16 \
  --context-length 192000 \
  --enable-dynamic-chunking \
  --chunked-prefill-size 8192 \
  --moe-runner-backend triton \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000

# Notes:
# - --ep-size 2: Enables Expert Parallelism across 2 GPUs.
# - --tp-size 1: Keeps dense layers on single GPU (assumes weights fit).
#   If OOM occurs, try --tp-size 2 --ep-size 1 (Standard TP) instead.
# - --chunked-prefill-size 8192: Balances memory usage and prefill speed for long ctx.
