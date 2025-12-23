# TileLang GEMM Configurations

This directory contains pre-tuned kernel configurations for TileLang GEMM.

## File Format

Each configuration file is named with the following format:
```
N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[128, 128].json
```

Where:
- `N`, `K`: Weight matrix dimensions
- `device_name`: GPU device name (e.g., `NVIDIA_H100_80GB_HBM3`)
- `dtype`: Data type, fixed to `fp8_w8a8`
- `block_shape`: Block shape for quantization, fixed to `[128, 128]`

Example: `N=4096,K=8192,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json`

## Configuration Structure

```json
{
    "1": {
        "kernel_type": "splitK_swapAB",
        "block_M": 64,
        "block_N": 128,
        "block_K": 128,
        "num_stages": 2,
        "threads": 256,
        "split_k": 4,
        "c_scale_local": true,
        "b_scale_shm": false,
        "latency_ms": 0.045,
        "tflops": 123.45
    },
    "128": {
        "kernel_type": "base",
        ...
    }
}
```

- Keys are M values (batch sizes)
- The wrapper automatically selects the closest M configuration for a given input

## Generating Configurations

Use the tuner to generate configurations:

```bash
# From tilelang-study directory
python -m tilelang_gemm.tuner --N 4096 --K 8192 --num_gpus 4
```

## Kernel Types

- `base`: Standard GEMM, A_scale (M, K//128), B_scale (N//128, K//128)
- `swapAB`: Swapped A/B, A_scale (M//128, K//128), B_scale (N, K//128)
- `splitK`: Split-K parallelism, uses C_partial buffer
- `splitK_swapAB`: Combined Split-K and SwapAB
