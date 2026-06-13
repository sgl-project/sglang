# Fused MoE Triton Kernel Configurations

This directory contains tuned configurations for different settings of the fused_moe kernel.

## Configuration Parameters

Each configuration file is generated based on the following parameters:

- **E** (number of experts): Total number of experts in the MoE layer
- **N** (intermediate size): The intermediate/hidden dimension size
  - For Tensor Parallelism (TP): `N = original_intermediate_size / tp_size`
  - Example: Mixtral has N = 14336. For TP=2, N = 7168; for TP=4, N = 3584
- **device_name**: GPU device name from `torch.cuda.get_device_name()`
  - Examples: `NVIDIA_H100_80GB_HBM3`, `NVIDIA_A100-SXM4-80GB`, `NVIDIA_GeForce_RTX_4090`
- **dtype**: Data type for computation
  - Supported types: `fp8_w8a8`, `int8_w8a8`, `int8_w8a16`, `int4_w4a16`, etc.
  - Determines precision and quantization scheme for weights and activations
- **block_shape**: Block quantization shape (for DeepSeek V3/R1 models)
  - Defines granularity for block-wise quantization, specified as `[block_n, block_k]`
  - Example: DeepSeek V3 commonly uses `[128, 128]` for efficient block-wise FP8 quantization
- **tp_size**: Tensor Parallelism size (affects N parameter)
- **ep_size**: Expert Parallelism size (affects E parameter when EP is enabled)
- **per_channel_quant**: Whether per-channel quantization is used

## Configuration File Format

Each JSON file contains a mapping from **M** (batch size) to the optimal kernel configuration for that batch size. The configuration includes parameters like `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `GROUP_M`, number of warps, and pipeline stages.

**Filename Format**:
```
E={E},N={N},device_name={device_name},dtype={dtype}[,block_shape={block_shape}][,per_channel_quant={bool}].json
```

## Generating Configuration Files

To generate new configuration files for your specific hardware and model settings, use the tuning tools:

**📖 Full Documentation**: [Tuning Triton MoE Kernels](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton)

After tuning, move the generated JSON files to this directory to use them in SGLang.
