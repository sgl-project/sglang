# TileLang GEMM Tuning

This directory contains scripts for tuning TileLang FP8 blockwise GEMM kernels.

## Overview

TileLang GEMM supports 4 kernel variants:
- `base`: Standard GEMM with A_scale (M, K//128), B_scale (N//128, K//128)
- `swapAB`: Swapped A/B inputs, better for small M
- `splitK`: Split-K parallelism for small M
- `splitK_swapAB`: Combined Split-K and SwapAB

## Usage

### 1. Tune for specific (N, K) dimensions

```bash
# Single GPU tuning
python tune_tilelang_gemm.py --N 4096 --K 8192

# Multi-GPU parallel tuning (recommended)
python tune_tilelang_gemm.py --N 4096 --K 8192 --num-gpus 4

# Tune for a model's weight shapes
python tune_tilelang_gemm.py --model Qwen/Qwen3-4B-FP8 --tp 1
```

### 2. Benchmark (baseline: DeepGEMM on Hopper, Triton on Ada)

```bash
python benchmark_tilelang_gemm.py --N 4096 --K 8192

# Output to CSV
python benchmark_tilelang_gemm.py --N 4096 --K 8192 --output results.csv
```

## Configuration Files

Tuned configurations are saved to:
```
python/sglang/srt/layers/tilelang_gemm_wrapper/core/config/
├── N=4096,K=8192,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128, 128].json
├── N=4096,K=14336,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128, 128].json
└── ...
```

## Prerequisites

Install tilelang:
```bash
pip install "tilelang>=0.1.7"
```

For multi-GPU tuning, install Ray:
```bash
pip install ray
```

## Integration with SGLang

After tuning, enable TileLang GEMM in SGLang:
```bash
python -m sglang.launch_server --model xxx --fp8-gemm-backend tilelang
```

Optionally specify a custom config directory:
```bash
export SGLANG_TILELANG_GEMM_CONFIG_DIR=/path/to/configs
```

## Module Structure

The TileLang GEMM implementation is fully integrated into sglang:

```
python/sglang/srt/layers/tilelang_gemm_wrapper/
├── __init__.py              # Public API
├── configurer.py            # Environment configuration
├── entrypoint.py            # Main entry point
└── core/
    ├── __init__.py
    ├── config_loader.py     # Config file management
    ├── kernel_registry.py   # Kernel type registry
    ├── tuner.py             # Ray-based multi-GPU tuner
    ├── wrapper.py           # Main wrapper class
    ├── config/              # Tuned configurations
    └── kernels/             # TileLang kernel implementations
        ├── base.py
        ├── swap_ab.py
        ├── split_k.py
        └── split_k_swap_ab.py
```
