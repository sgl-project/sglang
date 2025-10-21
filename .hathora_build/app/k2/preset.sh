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

# DP Attention
export DP_SIZE="${DP_SIZE:-16}"
export ENABLE_DP_ATTENTION="${ENABLE_DP_ATTENTION:-true}"
export ENABLE_DP_LM_HEAD="${ENABLE_DP_LM_HEAD:-true}"

# Expert Load Balancing (EPLB)
# if we want to distribute the load when doing EP. This can be disabled if perf goes down.
export ENABLE_EPLB="${ENABLE_EPLB:-true}"
export EPLB_ALGORITHM="${EPLB_ALGORITHM:-auto}"
export EPLB_REBALANCE_NUM_ITERATIONS="${EPLB_REBALANCE_NUM_ITERATIONS:-1000}"

# performance optimizations
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1024}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_cutlass}"

# Disable DeepGemm JIT
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

unset K2_PRESET # avoids inf loop

exec /app/entrypoint.sh
