# T4 GPU Optimization Guide — SGLang Diffusion JIT Kernels

T4 is a Turing architecture GPU (GCP n1+T4, AWS g4dn) commonly used for cloud inference.
Its key constraint for diffusion kernels: **no BF16 support** — FP16 only.

> **Adapted from**: [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels)
>
> If you use the FLUX `sglang generate` example below, export `HF_TOKEN` first. `black-forest-labs/FLUX.*` is a gated Hugging Face repo, and without a token the top-level CLI can fail before model loading.

---

## T4 Turing Architecture Overview

| Component | T4 | A100 | H100 |
|-----------|-----|------|------|
| Compute Capability | sm_75 | sm_80 | sm_90 |
| SMs | 40 | 108 | 132 |
| Shared Memory/SM | **64 KB** | 164 KB | 192 KB |
| L2 Cache | 4 MB | 40 MB | 50 MB |
| Memory Bandwidth | **320 GB/s** | 2.0 TB/s | 3.35 TB/s |
| Memory | 16 GB GDDR6 | 40–80 GB HBM2e | 80 GB HBM3 |
| Max Threads/SM | **1024** | 2048 | 2048 |
| BF16 Support | **No** | Yes | Yes |

### Critical T4 Constraints

1. **No BFloat16** — must use FP16 everywhere
2. **320 GB/s bandwidth** — ~10x lower than H100; vectorization is critical
3. **16 GB memory** — limits model size; use offloading
4. **64 KB shared memory/SM** — smaller attention tiles
5. **Max 1024 threads/SM** — half of A100/H100; affects occupancy calculations

---

## No BF16: Always Use FP16

This is the most impactful constraint. **Never use `bf16_t` or `__nv_bfloat16` on T4.**

**Python wrapper guard:**

```python
import torch
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

@cache_once
def _jit_rmsnorm_module(dtype: torch.dtype):
    # T4 (sm_75) does not support BF16
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0) and dtype == torch.bfloat16:
        raise RuntimeError(
            f"T4 (sm_75) does not support BF16. Use torch.float16 instead. "
            f"Got dtype={dtype}"
        )
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_rmsnorm",
        *args,
        cuda_files=["diffusion/rmsnorm.cuh"],
        cuda_wrappers=[("rmsnorm", f"rmsnorm<{args}>")],
    )
```

**Conditional type in kernel:**

```cuda
#if __CUDA_ARCH__ >= 800
    // A100/H100: BF16 available
    using DefaultHalf = bf16_t;
#else
    // T4/Turing: FP16 only
    using DefaultHalf = fp16_t;
#endif
```

**Runtime detection helper:**

```python
def get_diffusion_dtype() -> torch.dtype:
    """Return the appropriate half-precision dtype for the current GPU."""
    cap = torch.cuda.get_device_capability()
    if cap >= (8, 0):
        return torch.bfloat16   # A100/H100: prefer BF16
    else:
        return torch.float16    # T4/older: FP16 only
```

---

## Memory Access Optimization

With only 320 GB/s, **vectorization is more critical on T4 than on A100/H100**.

### `AlignedVector` (same abstraction, FP16 only)

```cpp
#include <sgl_kernel/vec.cuh>

// On T4, T must be fp16_t or fp32_t (NOT bf16_t)
constexpr int kVecN = 16 / sizeof(T);   // 8 for fp16, 4 for fp32
using vec_t = device::AlignedVector<T, kVecN>;
```

**Target bandwidth**: 40–50% of T4's 320 GB/s = 128–160 GB/s.

### Increase Arithmetic Intensity

With low bandwidth, fusing ops saves more on T4 than on H100:

```cpp
// BAD on T4: separate passes → 2× memory traffic
output1[i] = input[i] * scale;       // pass 1
output2[i] = output1[i] + bias;      // pass 2

// GOOD: fuse → single memory read, single write
float val = static_cast<float>(v[i]);
val = val * scale + bias;
val = device::math::max<float>(val, 0.f);  // ReLU
v[i] = static_cast<T>(val);
```

### Expected T4 Performance

| Kernel | T4 (ms) | A100 (ms) | H100 (ms) | T4 vs H100 |
|--------|---------|-----------|-----------|------------|
| RMSNorm [2, 1024, 2048] | ~0.5 | ~0.08 | 0.054 | ~9x slower |
| GEGLU [2, 1024, 4096] | ~0.3 | ~0.05 | 0.030 | ~10x slower |

---

## Shared Memory Configuration

T4 max: **64 KB/SM**. Use smaller tiles vs A100/H100.

```cpp
// T4: request max shared memory (64 KB)
cudaFuncSetAttribute(
    your_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    64 * 1024
);
```

**Attention tile sizes for T4** (halved vs H100):

```
H100/A100: BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 64
T4:        BLOCK_SIZE_M =  64, BLOCK_SIZE_N = 32   ← reduced for 64 KB limit
```

---

## Occupancy Tuning

T4 max: **1024 threads/SM** (vs 2048 on A100/H100). This halves max occupancy for a given block size.

**Block sizes for T4:**

| Kernel Type | Threads/Block | Notes |
|-------------|---------------|-------|
| Element-wise | 256 | Same as H100 |
| Row reduction | 256–512 | Avoid > 512 to fit multiple blocks/SM |
| Tiled/attention | 128–256 | Small tiles due to 64 KB shared mem |

**Grid sizing for T4 (40 SMs)** — `runtime::get_sm_count` handles this automatically:

```cpp
// get_sm_count() returns 40 on T4, 108 on A100, 132 on H100
const uint32_t num_sm = host::runtime::get_sm_count(device.unwrap().device_id);
```

---

## Numerical Stability with FP16

FP16 has a smaller dynamic range (±65504) vs BF16 (±3.39×10³⁸). Watch for overflow in attention:

```cuda
// Scale attention scores to prevent FP16 overflow
float scale_factor = 1.0f / sqrtf(static_cast<float>(head_dim));
// For very long sequences on T4, may need additional scaling:
// if (score * scale_factor > 65000.f) { /* clamp */ }
```

Always accumulate in FP32:

```cpp
float acc = 0.f;   // FP32 accumulation
for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
    vec_t v;
    v.load(src, vi);
    #pragma unroll
    for (int i = 0; i < kVecN; ++i) {
        float val = static_cast<float>(v[i]);   // fp16 → fp32
        acc += val * val;
    }
}
```

---

## Memory Management for 16 GB

T4's 16 GB requires careful planning for large diffusion models.

**sglang generate flags for T4:**

```bash
# Required for gated FLUX repos:
# export HF_TOKEN=<your_hf_token>

# Enable CPU offloading to fit within 16 GB
sglang generate \
  --model-path=black-forest-labs/FLUX.1-dev \
  --dit-cpu-offload true \        # DiT weights to CPU
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --width=512 --height=512 \      # Reduce resolution
  --num-inference-steps=20 \      # Fewer steps
  --seed=42
```

**Resolution recommendations for T4:**

| Model | H100/A100 | T4 |
|-------|-----------|-----|
| FLUX.1-dev | 1024×1024 | 512×512 |
| Wan2.2-TI2V-5B | 720P | 480P |
| FLUX.2-dev | 1024×1024 | 512×512 |

---

## JIT Compilation for T4

```python
return load_jit(
    "my_kernel",
    *args,
    cuda_files=["diffusion/my_kernel.cuh"],
    cuda_wrappers=[("my_kernel", f"my_kernel<{args}>")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_75",   # T4 only; omit for multi-arch
    ],
)
```

**Multi-arch (T4 + A100 + H100):**

```python
extra_cuda_cflags=[
    "-O3", "--use_fast_math",
    "-gencode=arch=compute_75,code=sm_75",   # T4
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_90,code=sm_90",   # H100
]
```

---

## H100/A100 → T4 Migration Checklist

| Item | H100/A100 | T4 | Action |
|------|-----------|-----|--------|
| BF16 | Available | **Not available** | Replace `bf16_t` with `fp16_t`; guard in Python wrapper |
| Shared memory | 164–192 KB | **64 KB** | Halve tile sizes |
| Grid sizing | ×108/132 SMs | ×40 SMs | `get_sm_count()` auto-handles |
| Max threads/SM | 2048 | **1024** | Don't exceed 512 threads/block |
| Memory | 40–80 GB | **16 GB** | Enable CPU offloading |
| cp.async | Available | No (Turing has limited async) | Remove async copy patterns |
| `AlignedVector` | Same | Same | No changes |
| `warp::reduce_sum` | Same | Same | No changes |

---

## Performance Profiling

### NVIDIA Nsight Systems (nsys)

```bash
nsys profile -o t4_profile python scripts/bench_diffusion_rmsnorm.py

# Key metrics to watch:
# - Kernel duration
# - Memory transfer time
# - GPU idle time
# - Stream utilization
```

### NVIDIA Nsight Compute (ncu)

```bash
# Full metrics
ncu --set full -o t4_metrics.ncu-rep \
  python scripts/bench_diffusion_rmsnorm.py

# Specific metrics — T4 is memory-bound; focus on dram throughput
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python scripts/bench_diffusion_rmsnorm.py

# Key metrics for T4 diffusion kernels:
# - Memory throughput     (dram__throughput.avg.pct_of_peak_sustained_elapsed)
#   → Target: 40–50% of 320 GB/s (128–160 GB/s) for vectorized kernels
# - SM utilization        (sm__throughput.avg.pct_of_peak_sustained_elapsed)
#   → Target high with only 40 SMs
# - Achieved occupancy    (sm__warps_active.avg.pct_of_peak_sustained_active)
#   → Max 1024 threads/SM on T4 — block size ≤ 512 for decent occupancy
# - Warp stall reasons    (smsp__warp_issue_stalled_*.avg.pct_of_peak_sustained_active)
```

### Common T4 Bottlenecks

1. **Memory Bandwidth** — 320 GB/s is the primary limit; if `dram__throughput` < 40% → use `AlignedVector`
2. **Limited Memory** — 16 GB; enable `--dit-cpu-offload`/`--vae-cpu-offload` as needed
3. **No BF16** — guard in Python wrapper; FP16 overflow risk in long-sequence attention
4. **Smaller tiles** — 64 KB shared memory; reduce `BLOCK_SIZE_M/N` vs H100

---

## Best Practices Summary (T4)

1. **No BF16**: Guard in Python wrapper, raise clear error
2. **Vectorization**: Even more critical at 320 GB/s — always use `AlignedVector`
3. **Tile sizes**: 64 KB shared memory limit → halve BLOCK_SIZE vs H100
4. **Block size**: Max 512 threads/block for decent occupancy (max 1024 threads/SM)
5. **Grid sizing**: 40 SMs — `runtime::get_sm_count()` auto-handles
6. **FP32 accumulation**: Always accumulate in FP32 to avoid FP16 overflow
7. **Memory**: Plan for 16 GB; use `--dit-cpu-offload`/`--vae-cpu-offload` as needed
8. **Fuse more**: Low bandwidth makes kernel fusion more impactful than on H100
9. **Multi-arch build**: Always build for `sm_75,sm_80,sm_90` together

## T4 Cloud Instance Quick Reference

| Provider | Instance | Notes |
|----------|----------|-------|
| GCP | n1-standard-4 + T4 | Most common inference setup |
| AWS | g4dn.xlarge | 1× T4, 16 GB |
| AWS | g4dn.12xlarge | 4× T4, 64 GB total |
| Azure | NC4as T4 v3 | 1× T4 |
