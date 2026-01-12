# W8A8 Block-wise Quantization Kernel Tuning

Auto-tune Triton FP8/INT8 block-wise quantization kernels for optimal performance.

## When to Use Triton FP8 Block-wise Quantization Kernel vs DeepGEMM

**Use Triton FP8 Block-wise Quantization Kernel when:**
- Output dtype is NOT `bfloat16` (e.g., `float16`, `float32`)
- DeepGEMM is disabled (environment variable `SGLANG_ENABLE_JIT_DEEPGEMM=0`)
- Running on GPUs with compute capability < SM90 (DeepGEMM requires SM90+)
- You need cross-platform compatibility (Triton works on both NVIDIA and AMD GPUs)

**Use DeepGEMM when:**
- Output dtype is `bfloat16` AND DeepGEMM is enabled
- Running on NVIDIA GPUs with compute capability >= SM90 (e.g., H100, H200)
- Need maximum performance for production workloads (DeepGEMM is highly optimized for Hopper architecture)

**Note:** DeepGEMM requires CUDA compute capability >= 9.0 (SM90+). It is specifically optimized for NVIDIA Hopper GPUs (H100/H200).

The kernel selection logic in SGLang automatically chooses DeepGEMM when conditions are met (see `w8a8_block_fp8_matmul` function in `fp8_kernel.py`), otherwise falls back to Triton implementation.

## Quick Start

**Default (DeepSeek-V3):**
```bash
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --tp-size 8
```

**Custom Model (specify N and K):**
```bash
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 5120 --K 25600
```

## Parameters

- `--N`, `--K`: Weight matrix dimensions (N=output_dim, K=input_dim). If not specified, uses `--tp-size` for DeepSeek-V3
- `--tp-size`: Tensor parallelism size for DeepSeek-V3 (default: 8)
- `--input-type`: `fp8` or `int8` (default: fp8)
- `--block-n`, `--block-k`: Block quantization granularity (default: 128)
- `--batch-size`: Test single batch size (optional)

## How to Calculate N and K

For a linear layer `y = xW^T` where `x` is (M, K) and `W` is (N, K):
- **N**: Output features (weight matrix output dimension)
- **K**: Input features (weight matrix input dimension)

**Example: Qwen3-VL-32B** (hidden_size=5120, intermediate_size=25600, num_heads=64, num_kv_heads=8, head_dim=128) and TP=1
```bash
# QKV projection: Q(8192) + K(1024) + V(1024) = 10240
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 10240 --K 5120

# MLP gate+up (SwiGLU): 2 * intermediate_size = 51200
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 51200 --K 5120

# MLP down projection
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 5120 --K 25600

# O projection (if separate from QKV)
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 5120 --K 8192
```

If TP=8:

```bash
# QKV projection: Q(8192) + K(1024) + V(1024) = 10240 / TP=8
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 1280 --K 5120

# MLP gate+up (SwiGLU): 2 * intermediate_size = 51200 / TP=8
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 6400 --K 5120

# MLP down projection
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 5120 --K 3200

# O projection (if separate from QKV)
python benchmark/kernels/quantization/tuning_block_wise_kernel.py --N 5120 --K 1024
```

## Output

Generates JSON config files saved to `python/sglang/srt/layers/quantization/configs/`:
```
N={N},K={K},device_name={DEVICE},dtype=fp8_w8a8,block_shape=[128,128].json
```

Config maps batch size to optimal kernel parameters:
```json
{
    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, ...},
    "2048": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, ...}
}
```
