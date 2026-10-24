#!/usr/bin/env bash
set -euo pipefail
# Run from tested SGLang checkout with its venv active. Single-node computation only.
# Install separately from repository root: pip install -e "python"
# Add --disable-custom-all-reduce to reproduce the control arm.
# Performance only: do not use this fixed-acceptance launcher for accuracy.
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export SGLANG_K3_NVFP4_ROUTE_QUANT=1
export SGLANG_K3_KV_B_PROJ_CUTLASS=1
export SGLANG_K3_NVFP4_CAKE=0
export SGLANG_K3_DCP_CUTE_PREFILL=0
export SGLANG_K3_BCG_MHA=0
export SGLANG_SIMULATE_ACC_LEN=4
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed
python -m sglang.launch_server --model nvidia/Kimi-K3-NVFP4 --trust-remote-code --host 127.0.0.1 --port 30000 --tp-size 8 --mem-fraction-static 0.85 --context-length 1048576 --kv-cache-dtype fp8_e4m3 --speculative-algorithm DFLASH --speculative-draft-model-path modal-labs/Kimi-K3-DFlash --speculative-dflash-block-size 8 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --moe-runner-backend flashinfer_trtllm --load-format fastsafetensors --enable-metrics --enable-cache-report --enable-linear-replayssm-spec --fp8-gemm-backend flashinfer_trtllm --enable-symm-mem --mamba-full-memory-ratio 0.015 --dcp-size 1 --max-running-requests 4 --chunked-prefill-size 16384 --max-prefill-tokens 16384 --disable-radix-cache "$@"
