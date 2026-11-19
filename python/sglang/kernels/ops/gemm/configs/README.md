# W8A8 FP8 Kernel Configurations

This directory contains tuned tile configurations for the W8A8 FP8 matrix
multiplication kernels: block-quantized (`_w8a8_block_fp8_matmul`) and
per-token/per-channel (`triton_scaled_mm`).

## Configuration File Format

Configuration files are named using the following pattern:
```
N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json
```

Where:
- `N`: Output dimension (number of columns in weight matrix)
- `K`: Input dimension (number of columns in activation matrix)
- `DEVICE_NAME`: GPU device name with spaces replaced by underscores (e.g., `NVIDIA_H100_80GB_HBM3`)
- `BLOCK_N`, `BLOCK_K`: Block quantization granularity (typically `[128,128]`)

## Per-token x per-channel (channelwise) FP8

`triton_scaled_mm` is the per-token/per-channel counterpart and uses a
separate table for the same weight shape. The name and body are fixed by
`get_w8a8_channelwise_fp8_configs` in `../fp8_kernel.py`:

```
N={N},K={K},device_name={DEVICE_NAME},dtype=fp8_w8a8_channelwise.json
```

The body maps token counts to a tile config:

```json
{
    "64": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "num_warps": 4,
        "num_stages": 4
    },
    "512": null
}
```

A `null` point means CUTLASS is faster at that M. Keep the point (so a
neighbouring M does not snap onto it) but fall back to CUTLASS; a missing
file or shape means the default path.
