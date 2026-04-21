#!/usr/bin/env bash
# Server launch for Qwen3.5-397B-A17B-FP8 with AsymGEMM masked GEMM backend.
#
# Key changes from previous version:
#   - CUDA graph is now ENABLED (--disable-cuda-graph removed).  The AsymGEMM
#     masked GEMM kernels were rewritten to use a fixed gridDim.y = num_groups
#     instead of the routing-dependent list_size-1, making them CUDA-graph safe.
#     Requires the updated AsymGEMM C++ library to be compiled and installed.
#     If the old binary is still in place, re-add --disable-cuda-graph.
#
#   - SGLANG_MASKED_GEMM_CHUNK_SIZE=8 enables the chunked scatter optimisation:
#     intermediate gateup/down buffers are (8, m, *) instead of (G, m, *),
#     and down_output is scattered directly into (num_tokens, K).
#     Expected peak intermediate memory: ~125 MB vs ~4 GB for Qwen-72B scale.
#     Tune the chunk size to a multiple of num_groups for best efficiency.
#
#   - --mem-fraction-static raised from 0.25 → 0.85 because the ~4 GB
#     intermediate-tensor spike no longer occurs during the forward pass.

SGLANG_MASKED_GEMM_CHUNK_SIZE=8 \
python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.5-397B-A17B-FP8 \
  --disable-radix-cache \
  --moe-runner-backend asym_gemm \
  --mem-fraction-static 0.85
