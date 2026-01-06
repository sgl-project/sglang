# FlashInfer HIP Patches

HIP/ROCm compatibility patches for [FlashInfer](https://github.com/flashinfer-ai/flashinfer).

## Quick Start

```bash
# Build and install FlashInfer with HIP support
./build_flashinfer_hip.sh --install

# Or just build wheel
./build_flashinfer_hip.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASHINFER_REPO` | github.com/flashinfer-ai/flashinfer | FlashInfer repository URL |
| `FLASHINFER_VERSION` | main | Branch/tag/commit to build |
| `AMDGPU_TARGET` | auto-detected | Target GPU arch (gfx942, gfx950) |
| `BUILD_DIR` | /tmp/flashinfer-hip-build | Build directory |

## Updating Patches

When FlashInfer upstream updates:

```bash
# Go to your patched FlashInfer repo
cd /path/to/flashinfer

# Rebase on upstream
git fetch origin
git rebase origin/main

# Regenerate patches
git format-patch origin/main -o /path/to/sglang/sgl-kernel/third_party/flashinfer_hip_patches/
```

## Patch Contents

The patches add `#ifdef __HIP_PLATFORM_AMD__` guards to provide:

- **PTX → GCN intrinsic translations** (exp2, log2, rcp, shuffles)
- **Type mappings** (cuda_fp16 → hip_fp16, bfloat16, etc.)
- **JIT compilation support** for HIP/hipcc
- **Memory operation replacements** (cp.async → regular loads)
- **TRT-LLM common utilities** (cooperative_groups, reduce kernels)

### Patch List

| Patch | Description |
|-------|-------------|
| 0001-0005 | Core FlashInfer HIP support (math, vec_dtypes, pos_enc) |
| 0006-feat-Add-TRT-LLM-norm-HIP-support | TRT-LLM norm kernel HIP compatibility |

## Scope

✅ Included:
- Core math intrinsics
- Vector types (half, bfloat16, **FP8 FNUZ**)
- JIT infrastructure
- Basic attention kernels
- **RMSNorm/LayerNorm kernels** (including fused variants)
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
- OCP types (`__hip_fp8_e4m3`) have `__host__`-only default constructors
- FNUZ types (`__hip_fp8_e4m3_fnuz`) have `__host__ __device__` constructors

The patches map CUDA FP8 types to HIP FNUZ types:
```cpp
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;  // Works on gfx942
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
```

Note: FNUZ and OCP have slightly different numeric representations, but
for inference workloads this difference is negligible.

## License

Apache 2.0 (same as FlashInfer)
