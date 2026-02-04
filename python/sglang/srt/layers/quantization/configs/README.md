# W8A8 Block FP8 Kernel Configurations

This directory contains optimized kernel configurations for the W8A8 block FP8 matrix multiplication kernel.

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
