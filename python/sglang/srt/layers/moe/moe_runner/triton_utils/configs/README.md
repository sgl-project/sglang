# Fused MoE Triton Kernel Configurations

This directory contains tuned configurations for different settings of the fused_moe kernel.

## Configuration Parameters

Each configuration file is generated based on the following parameters:

- **E** (number of experts): Total number of experts in the MoE layer
- **N** (intermediate size): The intermediate/hidden dimension size
  - For Tensor Parallelism (TP): `N = original_intermediate_size / tp_size`
  - Example: Mixtral has N = 14336. For TP=2, N = 7168; for TP=4, N = 3584
- **device_name**: GPU device name from `torch.cuda.get_device_name()`, with spaces
  replaced by underscores (that single `replace(" ", "_")` is the only
  sanitization the loader applies — see `get_config_file_name`)
  - Examples: `NVIDIA_H100_80GB_HBM3`, `NVIDIA_A100-SXM4-80GB`, `NVIDIA_GeForce_RTX_4090`
  - On AMD this is the marketing name, not the gfx arch: `AMD_Instinct_MI300X`,
    and on a virtualized MI350X `torch.cuda.get_device_name()` reports
    `AMD Instinct MI350X VF` → `AMD_Instinct_MI350X_VF`. A file named after the
    arch (e.g. `gfx950`) can never be found.
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

Each JSON file is a **flat** mapping from **M** (batch size) to the optimal kernel
configuration for that batch size. Keys must be integer-castable strings (the loader
does `{int(key): val for key, val in json.load(f).items()}`), and each value is a flat
dict using the kernel's `tl.constexpr` names — `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages` (plus `waves_per_eu` on
ROCm). Nested or differently-named keys are not read by anything.

**Filename Format**:
```
E={E},N={N},device_name={device_name},dtype={dtype}[,block_shape={block_shape}][,per_channel_quant={bool}][_down].json
```

## Generating Configuration Files

To generate new configuration files for your specific hardware and model settings, use the tuning tools:

**📖 Full Documentation**: [Tuning Triton MoE Kernels](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton)

After tuning, move the generated JSON files into the **Triton-version subdirectory**
of this directory, not into this directory itself:

```
configs/triton_<triton.__version__ with dots replaced by underscores>/<filename>
```

e.g. Triton 3.6.0 → `configs/triton_3_6_0/E=128,N=384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json`

`get_moe_configs` only ever probes `configs/triton_*/`: it tries the running
Triton version's directory first, then falls back to the other `triton_*`
directories newest-first (`os.listdir(configs_root)` filtered on
`startswith("triton_")`). **A file left at the root of `configs/` is never
enumerated and can never be loaded** — the lookup silently falls through to
`get_default_config()`, so a mislocated table looks like "tuning had no effect"
rather than an error. Check the startup log for the
`Using MoE kernel config from ...` line to confirm your file was picked up.
