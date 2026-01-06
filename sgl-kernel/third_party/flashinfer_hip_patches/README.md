# FlashInfer AMD/HIP Fork

AMD MI300X compatible fork of [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

## Quick Start

```bash
# Install from AMD-compatible fork
pip install git+https://github.com/sunxxuns/flashinfer.git

# Or clone and install
git clone https://github.com/sunxxuns/flashinfer.git
cd flashinfer
pip install -e . -v
```

## Fork Repository

**https://github.com/sunxxuns/flashinfer**

The fork includes all HIP/ROCm patches directly in the codebase.

## Performance (FLUX.1-dev on MI300X)

| Kernel | Denoising (50 steps) | Per Step | Speedup |
|--------|---------------------|----------|---------|
| **FlashInfer** | 8.75s | 0.174s | **1.29x** |
| Triton fallback | 11.26s | 0.224s | baseline |

## Reproduce

```bash
# 1. Install FlashInfer AMD fork
pip install git+https://github.com/sunxxuns/flashinfer.git

# 2. Run with FlashInfer (default)
SGLANG_LOG_KERNEL=1 sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A beautiful sunset over mountains" \
  --save-output

# 3. Run WITHOUT FlashInfer (Triton fallback) for comparison
SGLANG_DISABLE_FLASHINFER=1 SGLANG_LOG_KERNEL=1 sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --prompt "A beautiful sunset over mountains" \
  --save-output
```

### Expected Output

With FlashInfer:
```
[FlashInfer] Using FlashInfer RMSNorm kernel on HIP
[FlashInfer] Using FlashInfer RoPE kernel
[DenoisingStage] average time per step: 0.174 seconds
```

Without FlashInfer:
```
FlashInfer disabled via SGLANG_DISABLE_FLASHINFER=1
[Triton] Using Triton RMSNorm fallback on HIP
FlashInfer not available, using Triton fallback for RoPE
[DenoisingStage] average time per step: 0.224 seconds
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_DISABLE_FLASHINFER` | `0` | Set to `1` to force Triton fallback |
| `SGLANG_LOG_KERNEL` | `0` | Set to `1` to log which kernels are used |

## What's Included

✅ Working on AMD MI300X (gfx942):
- Core math intrinsics (PTX → GCN translations)
- Vector types (half, bfloat16, **FP8 FNUZ**)
- JIT infrastructure for HIP/hipcc
- **RMSNorm/LayerNorm kernels**
- **RoPE kernels** (pos_enc.cuh)
- **FP8 support** (using FNUZ types for gfx942)

❌ Not portable (NVIDIA-specific):
- Hopper/Blackwell CUTLASS kernels
- Tensor core MMA operations
- Advanced TMA features
- cuBLAS/cuBLASLt GEMM operations

## FP8 Support on MI300X

MI300X (gfx942) uses **FNUZ FP8 format** (not OCP format):
- `HIP_FP8_TYPE_FNUZ=1`, `HIP_FP8_TYPE_OCP=0` on gfx942
- OCP types have `__host__`-only default constructors
- FNUZ types have `__host__ __device__` constructors

The fork maps CUDA FP8 types to HIP FNUZ types:
```cpp
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;  // Works on gfx942
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
```

## Syncing with Upstream

```bash
cd flashinfer
git remote add upstream https://github.com/flashinfer-ai/flashinfer.git
git fetch upstream
git merge upstream/main  # or rebase
# Resolve conflicts in HIP-specific code
git push origin main
```

## License

Apache 2.0 (same as FlashInfer)
