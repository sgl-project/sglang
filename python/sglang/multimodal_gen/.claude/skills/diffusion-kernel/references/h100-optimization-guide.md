# H100 GPU Optimization Guide — SGLang Diffusion JIT Kernels

Deep dive into H100-specific optimizations for diffusion model CUDA kernels, written for SGLang's JIT kernel system.

> **Adapted from**: [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels)

---

## H100 Hopper Architecture Overview

| Component | Specification | Optimization Implication |
|-----------|---------------|--------------------------|
| Compute Capability | sm_90 | Use `extra_cuda_cflags=["-arch=sm_90"]` in `load_jit` |
| SMs | 132 | Grid: aim for multiples of 132 |
| Shared Memory | 192 KB/SM | Configurable: 96/144/192 KB |
| L2 Cache | 50 MB | Tile K,V of attention to fit in L2 |
| Memory Bandwidth | 3.35 TB/s | BF16 vectorized: achieves ~38% (~1.27 TB/s) |
| Max Threads/SM | 2048 | Max 16 blocks of 128 threads per SM |
| Warp Size | 32 | All reductions use `warp::reduce_sum` |
| Registers | 64K 32-bit/SM | 255 per thread max |

### New Hopper Features (sm_90+)

1. **Thread Block Clusters** — groups cooperating via Distributed Shared Memory
2. **TMA (Tensor Memory Accelerator)** — hardware-accelerated bulk copies
3. **FP8 support** — native 8-bit floating point in tensor cores
4. **PDL (Programmatic Dependent Launch)** — enable with `.enable_pdl(true)` in `LaunchKernel`

Gate sm_90+ features with a runtime check before calling `load_jit`:

```python
if torch.cuda.get_device_capability()[0] < 9:
    raise RuntimeError("This kernel requires H100 (sm_90+)")
```

---

## Memory Hierarchy Optimization

### Coalesced Global Memory Access

```cpp
// GOOD: threads read consecutive addresses → 128-byte transaction per warp
uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
fp16_t val = src[idx];

// BAD: strided access → multiple transactions, lower effective bandwidth
uint32_t idx = threadIdx.x * stride;  // avoid stride > 1
```

**Transaction sizes**: 32 bytes minimum, 128 bytes optimal (full warp, FP32).

### Vectorized Memory Access with `AlignedVector`

SGLang's `AlignedVector<T, N>` provides 128-bit (16-byte) vector loads. Always use this instead of raw pointer reinterprets.

```cpp
#include <sgl_kernel/vec.cuh>

// 16 bytes per load: 8×bf16_t, 8×fp16_t, or 4×fp32_t
constexpr int kVecN = 16 / sizeof(T);
using vec_t = device::AlignedVector<T, kVecN>;

// Load
vec_t v;
v.load(src, vi);           // loads src[vi * kVecN .. vi * kVecN + kVecN - 1]

// Process
#pragma unroll
for (int i = 0; i < kVecN; ++i) {
    float val = static_cast<float>(v[i]);
    // ... compute ...
    v[i] = static_cast<T>(result);
}

// Store
v.store(dst, vi);
```

**RMSNorm benchmark (H100 80GB, BF16):**

| Implementation | Time (ms) | Speedup |
|:---|:---:|:---:|
| Scalar loads | 0.065 | 1.00x |
| `AlignedVector<bf16_t, 8>` | 0.019 | **3.37x** |

Bandwidth achieved: **~38% of 3.35 TB/s** = 1.27 TB/s.

### L2 Cache Utilization (50 MB)

For attention, tile K and V so they stay in L2 while Q iterates:

```
BLOCK_SIZE_M = 128  (Q block)
BLOCK_SIZE_N = 64   (K,V block)
With head_dim=64: tile = 128×64×2 = 16 KB (FP16), multiple tiles fit in L2
```

### Shared Memory Configuration

Request max shared memory for attention kernels:

```cpp
// In launcher (after selecting kernel function pointer):
cudaFuncSetAttribute(
    your_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    192 * 1024  // 192 KB max on H100
);
```

Shared memory has 32 banks (4 bytes/bank). Avoid conflicts with padding:

```cpp
__shared__ float data[32][33];  // 33 instead of 32 → no bank conflict
```

---

## Warp & CTA Reductions (SGLang Abstractions)

Use `sgl_kernel/warp.cuh` and `sgl_kernel/cta.cuh` — never raw `__shfl_xor_sync`.

```cpp
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/cta.cuh>

// Warp-level sum (uses __shfl_xor_sync internally)
float result = device::warp::reduce_sum<float>(partial);

// Warp-level max
float mx = device::warp::reduce_max<float>(val);

// CTA-wide max via shared memory
__shared__ float smem[32];
device::cta::reduce_max<float>(val, smem, -1e38f);
// smem[0] holds the result after __syncthreads()
```

**Block reduction pattern for RMSNorm:**

```cpp
// 1. Warp reduction
sum_sq = device::warp::reduce_sum<float>(sum_sq);

// 2. Write warp leaders to smem
__shared__ float smem_r[32];
if (threadIdx.x % 32 == 0) smem_r[threadIdx.x / 32] = sum_sq;
__syncthreads();

// 3. Final warp reduction over warp leaders
if (threadIdx.x < 32) {
    sum_sq = (threadIdx.x < blockDim.x / 32) ? smem_r[threadIdx.x] : 0.f;
    sum_sq = device::warp::reduce_sum<float>(sum_sq);
}
__syncthreads();
```

---

## Occupancy Tuning

```
Occupancy = Active Warps per SM / Max Warps per SM (64)

Limiting factors on H100:
  1. Registers: 65536 / (threads_per_block × regs_per_thread)
  2. Shared Memory: 192 KB / smem_per_block
  3. Threads: 2048 / threads_per_block
```

**Recommended block sizes:**

| Kernel Type | Threads/Block | Warps | Reasoning |
|-------------|---------------|-------|-----------|
| Element-wise (RoPE, GEGLU) | 256 | 8 | High occupancy, simple |
| Row reduction (RMSNorm, LayerNorm) | 256–512 | 8–16 | Enough threads for full reduction |
| Tiled (attention) | 256 | 8 | Balance shared mem and registers |

**Persistent kernel pattern** (cap grid to SM × occupancy):

```cpp
#include <sgl_kernel/runtime.cuh>

static const uint32_t max_occ = host::runtime::get_blocks_per_sm(kernel, kBlockSize);
static const uint32_t num_sm  = host::runtime::get_sm_count(device.unwrap().device_id);
const uint32_t num_blocks = std::min(num_sm * max_occ, host::div_ceil(n, kBlockSize));
host::LaunchKernel(num_blocks, kBlockSize, device.unwrap())(kernel, params);
```

---

## Precision and Numerical Stability

| Type | Exponent Bits | Mantissa Bits | Range | Use Case |
|------|--------------|---------------|-------|----------|
| FP16 | 5 | 10 | ±65504 | Inference; attention score overflow risk |
| BF16 | 8 | 7 | ±3.39×10³⁸ | Training/inference preferred; safer for attn |
| FP32 | 8 | 23 | ±3.39×10³⁸ | Accumulation only |

**Mixed precision pattern** (always accumulate in FP32):

```cpp
// Input via AlignedVector
vec_t v;
v.load(src, vi);
float acc = 0.f;
#pragma unroll
for (int i = 0; i < kVecN; ++i) {
    float val = static_cast<float>(v[i]);  // promote to FP32
    acc += val * val;
}
// Output
v[i] = static_cast<T>(fp32_result);       // demote back
```

---

## Diffusion-Specific Patterns

### DiT Block Operators

| Operator | Pattern | Key Constraint |
|----------|---------|----------------|
| **RMSNorm** | 2-pass row reduction | weight may be `None` |
| **AdaLN** | `norm(x) * (1 + scale) + shift` | fuse norm+scale+shift |
| **RoPE 3D** | `[B, t*h*w, heads, head_dim]` | layout: `seq = t*h*w` |
| **GEGLU** | `gelu(gate) * value`, input `[B,L,2H]` | don't use for LTX-Video (uses GELU) |
| **SiLU gate** | `x * sigmoid(x)` | fuse with MLP linear |

### Online Softmax (for custom attention)

```cuda
// Numerically stable without materializing full [seq×seq] score matrix
float row_max = -INFINITY, row_sum = 0.f;
for each K block:
    compute local_scores
    new_max = max(row_max, max(local_scores))
    rescale = exp(row_max - new_max)
    row_sum = row_sum * rescale + sum(exp(local_scores - new_max))
    out_acc = out_acc * rescale + softmax(local_scores) @ V_block
    row_max = new_max
```

---

## Profiling and Debugging

### NVIDIA Nsight Systems (nsys)

System-wide profiling to see kernel durations, memory transfers, and GPU idle time:

```bash
nsys profile -o profile_report python scripts/bench_diffusion_rmsnorm.py

# Key metrics to watch:
# - Kernel duration
# - Memory transfer time
# - GPU idle time
# - Stream utilization
```

For end-to-end denoise profiling via `sglang generate`, see `diffusion-benchmark-and-profile.md` (Level 2: nsys + gputrc2graph.py).

### NVIDIA Nsight Compute (ncu)

Detailed per-kernel analysis for tuning individual JIT CUDA kernels:

```bash
# Full metrics — use when you need everything (slow)
ncu --set full -o metrics.ncu-rep \
  python scripts/bench_diffusion_rmsnorm.py

# Specific metrics — use for targeted bandwidth / occupancy checks
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  python scripts/bench_diffusion_rmsnorm.py

# Key metrics for diffusion JIT kernels:
# - Achieved occupancy        (sm__warps_active.avg.pct_of_peak_sustained_active)
# - Memory throughput         (dram__throughput.avg.pct_of_peak_sustained_elapsed)
# - Compute throughput        (sm__throughput.avg.pct_of_peak_sustained_elapsed)
# - Warp stall reasons        (smsp__warp_issue_stalled_*.avg.pct_of_peak_sustained_active)
# - L1 cache hit rate         (l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum)
```

### Common Performance Issues

1. **Low occupancy**: Too many registers or shared memory per block
   - Check: `--ptxas-options=-v` in `extra_cuda_cflags` to see register count
   - Fix: Reduce `--maxrregcount=N`; use smaller block size

2. **Memory bound, low bandwidth**: Achieved < 30% of 3.35 TB/s
   - Check: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - Fix: Switch to `AlignedVector<T, 16/sizeof(T)>` for 128-bit vector loads

3. **Shared memory bank conflicts**: `l1tex__data_bank_conflicts_pipe_lmem_op_st.sum` is high
   - Fix: Add padding — `__shared__ float data[32][33]`

4. **Warp divergence**: Conditional branches splitting warps
   - Check: `smsp__warp_issue_stalled_branch.avg.pct_of_peak_sustained_active`
   - Fix: Restructure so elements with identical branches are in the same warp

5. **Too many small kernels**: High kernel launch overhead
   - Fix: Fuse operations (e.g., norm + scale + shift → AdaLN in one kernel)

---

## JIT Compilation Notes

SGLang's JIT compiles kernels on first use via `load_jit`. For H100-specific flags:

```python
return load_jit(
    "my_kernel",
    *args,
    cuda_files=["diffusion/my_kernel.cuh"],
    cuda_wrappers=[("my_kernel", f"my_kernel<{args}>")],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-arch=sm_90",          # H100 only; omit for multi-arch
        "--ptxas-options=-v",   # Remove after tuning
    ],
)
```

For multi-arch (H100 + A100):

```python
extra_cuda_cflags=[
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_80,code=sm_80",   # A100
    "-gencode=arch=compute_90,code=sm_90",   # H100
]
```

---

## Best Practices Summary

1. **Memory access**: Coalesce writes, align to 128-byte boundaries
2. **Vectorization**: Use `AlignedVector<T, 16/sizeof(T)>` for all element-wise loads/stores
3. **Reductions**: Use `warp::reduce_sum/max`, then shared memory pattern above
4. **Precision**: BF16 for I/O, FP32 for accumulation; use `static_cast<float>`
5. **Block size**: 256 threads default; 512 for reductions; tune with `runtime::get_blocks_per_sm`
6. **Grid sizing**: Multiples of 132 SMs; use persistent kernel pattern for small N
7. **Shared memory**: Add padding (`[32][33]`) to avoid bank conflicts
8. **Profile**: Run `ncu` before claiming a speedup; check dram throughput %
9. **Fuse**: Combine norm + scale + shift into a single pass to reduce memory traffic
10. **Abstractions**: Always use `TensorMatcher`, `AlignedVector`, `LaunchKernel` — never raw CUDA

## Reference Benchmark Results (H100 80GB, BF16)

| Kernel | Shape | Time (ms) |
|--------|-------|-----------|
| RMSNorm | [2, 1024, 2048] | 0.054 |
| GEGLU | [2, 1024, 4096] → [2, 1024, 2048] | 0.030 |
| RoPE 3D | [2, 480, 8, 64] | 1.670 |
| RMSNorm vectorized | [1, 1024, 2048] | 0.019 |
| RMSNorm vectorized | [4, 4096, 3072] | 0.157 |

> See `kernel-templates.md` for copy-paste ready sglang JIT kernel implementations.
