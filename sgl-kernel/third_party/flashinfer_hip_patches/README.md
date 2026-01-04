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
- Vector types (half, bfloat16)
- JIT infrastructure
- Basic attention kernels
- **RMSNorm/LayerNorm kernels** (non-quantized)
- **RoPE kernels** (pos_enc.cuh)

⚠️ In Progress (FP8 template specialization issues on hipcc):
- FP8 quantized norm kernels (RMSNormQuant, FusedAddRMSNormQuant)

❌ Not portable (NVIDIA-specific):
- Hopper/Blackwell CUTLASS kernels
- Tensor core MMA operations
- Advanced TMA features
- cuBLAS/cuBLASLt GEMM operations

## Known Issues

### FP8 Template Specialization
HIP compiler (hipcc) has issues matching type alias template specializations:
- `vec_t<__nv_fp8_e4m3, N>` specializations don't match when instantiated with the canonical type `__hip_fp8_e4m3`
- Workaround: Use non-quantized norm kernels on HIP

## License

Apache 2.0 (same as FlashInfer)
