# A100 GPU Optimization Guide — SGLang Diffusion JIT Kernels

Deep dive into A100-specific optimizations for diffusion model CUDA kernels in SGLang's JIT system.

> **Adapted from**: [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels)

---

## A100 Ampere Architecture Overview

| Component | A100 40GB | A100 80GB | Notes |
|-----------|-----------|-----------|-------|
| Compute Capability | sm_80 | sm_80 | Use `"-arch=sm_80"` in `extra_cuda_cflags` |
| SMs | 108 | 108 | Grid: aim for multiples of 108 |
| Shared Memory | 164 KB/SM | 164 KB/SM | Configurable: 48/96/164 KB |
| L2 Cache | 40 MB | 40 MB | Less than H100 (50 MB) |
| Memory Bandwidth | 1.55 TB/s | 2.0 TB/s | HBM2e |
| Max Threads/SM | 2048 | 2048 | Same as H100 |
| Tensor Cores | 3rd gen | 3rd gen | FP16, BF16, TF32, INT8, INT4 |

### A100 vs H100 Comparison

| Feature | A100 | H100 | Impact on JIT Kernels |
|---------|------|------|-----------------------|
| Memory BW | 2.0 TB/s | 3.35 TB/s | H100 ~67% faster for memory-bound ops |
| SMs | 108 | 132 | Adjust persistent kernel grid sizing |
| Shared Mem/SM | 164 KB | 192 KB | Reduce max tile sizes on A100 |
| L2 Cache | 40 MB | 50 MB | Attention tile reuse still works well |
| TMA | No | Yes | Can't use `cp.async.bulk` on A100 |
| FP8 | No | Yes | Use FP16/BF16 only on A100 |

---

## Memory Access Optimization

Same coalescing and vectorization rules as H100; lower bandwidth makes them even more critical.

### `AlignedVector` Vectorization (same pattern as H100)

```cpp
#include <sgl_kernel/vec.cuh>

constexpr int kVecN = 16 / sizeof(T);   // 8 for bf16/fp16, 4 for fp32
using vec_t = device::AlignedVector<T, kVecN>;

vec_t v;
v.load(src, vi);
// ... process elements ...
v.store(dst, vi);
```

**Expected A100 performance (BF16 RMSNorm):**

| Implementation | A100 (ms) | H100 (ms) | A100 Speedup |
|:---|:---:|:---:|:---:|
| Scalar loads | ~0.10 | 0.065 | 1.00x |
| `AlignedVector<bf16_t, 8>` | ~0.03 | 0.019 | ~3x |

**Target bandwidth**: 30–40% of A100's 2.0 TB/s = 600–800 GB/s.

### Shared Memory Configuration

```cpp
// A100 max: 164 KB/SM
cudaFuncSetAttribute(
    your_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    164 * 1024  // 164 KB max on A100
);
```

Attention tile sizes for A100:

```
BLOCK_SIZE_M = 128  (Q block)
BLOCK_SIZE_N = 64   (K,V block)
Tile = 128×64×2 = 16 KB (FP16) — fits in 164 KB shared mem
```

---

## Occupancy Tuning

**Grid sizing for A100 (108 SMs):**

```cpp
#include <sgl_kernel/runtime.cuh>

// Cap blocks to SM × occupancy (same pattern as H100)
static const uint32_t max_occ = host::runtime::get_blocks_per_sm(kernel, kBlockSize);
static const uint32_t num_sm  = host::runtime::get_sm_count(device.unwrap().device_id);
const uint32_t num_blocks = std::min(num_sm * max_occ, host::div_ceil(n, kBlockSize));
```

**Recommended block sizes (same as H100):**

| Kernel Type | Threads/Block | Notes |
|-------------|---------------|-------|
| Element-wise | 256 | High occupancy |
| Row reduction | 512 | Full reduction per row |
| Tiled/attention | 256 | Balance shared mem |

---

## A100-Specific Features

### Async Memory Copy (sm_80)

A100 introduced `cp.async` for overlapping compute and memory. Use this in custom kernels for prefetching:

```cuda
#if __CUDA_ARCH__ >= 800
// Async copy from global to shared (A100+)
__pipeline_memcpy_async(smem_ptr, global_ptr, bytes);
__pipeline_commit();
__pipeline_wait_prior(0);
#endif
```

### TF32 Mode (A100 specific)

Enables FP32-range with FP16-like throughput for GEMM. Enable in Python:

```python
# Enable TF32 for matmuls (A100+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TF32 is automatic for FP32 GEMMs via cuBLAS — no kernel changes needed.

### Structural Sparsity (2:4)

A100 tensor cores support 50% structured sparsity:

```python
from torch.sparse import to_sparse_semi_structured
sparse_weight = to_sparse_semi_structured(dense_weight)
# ~2x GEMM speedup for matmul with sparse weight
```

---

## JIT Compilation for A100

```python
return load_jit(
    "my_kernel",
    *args,
    cuda_files=["diffusion/my_kernel.cuh"],
    cuda_wrappers=[("my_kernel", f"my_kernel<{args}>")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_80",   # A100 only; omit for multi-arch
    ],
)
```

**Multi-arch (A100 + H100):**

```python
extra_cuda_cflags=[
    "-O3", "--use_fast_math",
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_90,code=sm_90",   # H100
]
```

Runtime arch guard (in Python wrapper):

```python
cap = torch.cuda.get_device_capability()
if cap < (8, 0):
    raise RuntimeError(f"This kernel requires sm_80 (A100) or later, got sm_{cap[0]}{cap[1]}")
```

---

## H100 → A100 Migration Checklist

When porting an H100-optimized kernel to A100:

| Item | H100 | A100 | Change Required |
|------|------|------|-----------------|
| Shared memory | 192 KB | 164 KB | Reduce `cudaFuncSetAttribute` size |
| Grid sizing | ×132 SMs | ×108 SMs | `get_sm_count()` handles automatically |
| TMA bulk copy | Available | **Not available** | Remove `cp.async.bulk`; use standard `__pipeline_memcpy_async` |
| FP8 | Available | **Not available** | Fall back to FP16/BF16 |
| PDL | Supported | Supported | `.enable_pdl(true)` works on sm_80 |
| Warp shuffles | Same | Same | No changes |
| `AlignedVector` | Same | Same | No changes |

**Conditional compilation:**

```cuda
#if __CUDA_ARCH__ >= 900
    // H100-only: TMA, FP8, thread block clusters
    #define USE_TMA 1
#elif __CUDA_ARCH__ >= 800
    // A100: cp.async, TF32, 2:4 sparsity
    #define USE_ASYNC_COPY 1
#endif
```

---

## Precision Notes

| Type | Available on A100 | Notes |
|------|-------------------|-------|
| FP16 | Yes | Good, watch overflow in attention |
| BF16 | Yes | Preferred for training and inference |
| TF32 | Yes (A100 specific) | Auto for FP32 GEMMs |
| FP8 | **No** | H100 only |

---

## Performance Profiling

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o a100_profile python scripts/bench_diffusion_rmsnorm.py

# Key metrics to watch:
# - Kernel duration
# - Memory transfer time
# - GPU idle time
# - Stream utilization
```

### NVIDIA Nsight Compute (ncu)

```bash
# Full metrics
ncu --set full -o a100_metrics.ncu-rep \
  python scripts/bench_diffusion_rmsnorm.py

# Specific metrics for bandwidth / occupancy checks
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python scripts/bench_diffusion_rmsnorm.py

# Key metrics for A100 diffusion kernels:
# - Achieved occupancy        (sm__warps_active.avg.pct_of_peak_sustained_active)
# - Memory throughput         (dram__throughput.avg.pct_of_peak_sustained_elapsed)
#   → Target: 30–40% of 2.0 TB/s (600–800 GB/s) for vectorized kernels
# - Compute throughput        (sm__throughput.avg.pct_of_peak_sustained_elapsed)
# - Warp stall reasons        (smsp__warp_issue_stalled_*.avg.pct_of_peak_sustained_active)
# - Kernel time               (gpu__time_duration.avg)
```

### Common A100 Performance Issues

1. **Memory bound below target**: `dram__throughput` < 30%
   - Fix: Use `AlignedVector<bf16_t, 8>` (128-bit vector loads)

2. **Low occupancy**: Grid too small for 108 SMs
   - Fix: Use `runtime::get_sm_count()` persistent kernel pattern

3. **No TF32 for FP32 GEMMs**: torch.backends.cuda.matmul.allow_tf32 not set
   - Fix: `torch.backends.cuda.matmul.allow_tf32 = True`

---

## Best Practices Summary (A100)

1. **Bandwidth**: Even more critical than H100 — profile with `ncu` first
2. **Vectorization**: `AlignedVector<bf16_t, 8>` gives ~3x over scalar
3. **TF32**: Enable for any FP32 matmul workload
4. **Shared memory**: Cap at 164 KB; use `cudaFuncSetAttribute`
5. **Grid sizing**: Multiples of 108 SMs via `runtime::get_sm_count`
6. **cp.async**: Use for prefetching in tiled kernels
7. **Multi-arch**: Build for both `sm_80` and `sm_90` to support both GPUs
8. **Same abstractions**: `AlignedVector`, `TensorMatcher`, `LaunchKernel` work identically

## Reference Benchmark Results (A100 80GB, BF16)

| Kernel | Shape | A100 (ms) | H100 (ms) | H100 Speedup |
|--------|-------|-----------|-----------|--------------|
| RMSNorm | [2, 1024, 2048] | ~0.08 | 0.054 | 1.5x |
| GEGLU | [2, 1024, 4096] | ~0.05 | 0.030 | 1.7x |
