---
name: add-cuda-kernel
description: Step-by-step guide for adding a new JIT CUDA kernel to SGLang Diffusion. CUDA source files go in jit_kernel/csrc/diffusion/<op>.cuh; Python wrapper at jit_kernel/diffusion/<op>.py. Use when implementing optimized CUDA kernels for diffusion model operators (RMSNorm, RoPE, AdaLN, GEGLU, etc.) on NVIDIA GPUs (H100, A100). Covers kernel authoring with sglang abstractions, JIT compilation, Python wrapper, integration into the denoise stage, and benchmarking. Adapted from https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels.
---

# Adding a CUDA Kernel to SGLang Diffusion (JIT Style)

> **Origin**: This skill is adapted from the [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels), rewritten to follow SGLang's JIT compilation system and internal abstractions.
>
> **Run environment first**: before compiling, benchmarking, or profiling any kernel from this guide, use `scripts/diffusion_skill_env.py` (or the setup block in `diffusion-benchmark-and-profile.md`) to `cd` to the repo root resolved from `sglang.__file__`, verify write access, export `FLASHINFER_DISABLE_VERSION_CHECK=1`, and pick an idle GPU.
>
> **Extended references** (in this directory's `references/` and `scripts/`):
> - [references/kernel-templates.md](references/kernel-templates.md) — copy-paste ready templates for element-wise, row-reduction (RMSNorm), fused AdaLN
> - [references/troubleshooting.md](references/troubleshooting.md) — build errors, perf issues, integration pitfalls
> - [references/h100-optimization-guide.md](references/h100-optimization-guide.md) — H100 (sm_90) deep dive
> - [references/a100-optimization-guide.md](references/a100-optimization-guide.md) — A100 (sm_80) deep dive
> - [references/t4-optimization-guide.md](references/t4-optimization-guide.md) — T4 (sm_75, FP16 only) deep dive
> - [scripts/bench_diffusion_rmsnorm.py](scripts/bench_diffusion_rmsnorm.py) — RMSNorm micro-benchmark vs PyTorch
> - [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark with/without kernels

## When to Use CUDA vs Triton

| Scenario | Use |
|----------|-----|
| Fused elementwise / norm variants / RoPE | **Triton** (`add-triton-kernel.md`) — faster iteration |
| Bandwidth-bound reduction (RMSNorm, LayerNorm) requiring max vectorization | **CUDA** — full control over `__nv_bfloat162` / `float4` vectorization |
| Attention pattern or tile-based ops needing shared memory tuning | **CUDA** — warp-level primitives, shared memory layout |
| Prototype or NPU/CPU fallback needed | **Triton** — portable across backends |

For most diffusion-model elementwise ops, **start with Triton**. Switch to CUDA when profiling shows Triton can't reach hardware bandwidth limits.

## Directory Layout

```
python/sglang/jit_kernel/
├── csrc/
│   ├── diffusion/               # JIT CUDA source files for diffusion kernels (this skill)
│   │   ├── timestep_embedding.cuh   # existing example
│   │   ├── rmsnorm.cuh              # NEW: add here
│   │   └── adaln.cuh                # NEW: add here
│   └── elementwise/             # shared JIT CUDA csrc (non-diffusion)
├── diffusion/
│   ├── triton/                  # Triton kernels (scale_shift, norm, rope, ...)
│   ├── cutedsl/                 # CuTe DSL kernels
│   └── rmsnorm.py               # NEW: CUDA JIT Python wrapper (add here)
├── timestep_embedding.py        # existing CUDA diffusion kernel Python wrapper (legacy)
```

New diffusion CUDA kernel source files go into `python/sglang/jit_kernel/csrc/diffusion/<op_name>.cuh`.
The Python wrapper goes at `python/sglang/jit_kernel/diffusion/<op_name>.py`
(inside `diffusion/`, alongside the `triton/` and `cutedsl/` subdirectories).

---

## SGLang Kernel Abstractions (Required)

Always use these — do **not** use raw CUDA primitives directly.

```cpp
#include <sgl_kernel/tensor.h>    // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>    // fp16_t, bf16_t, fp32_t, dtype_trait, packed_t
#include <sgl_kernel/utils.h>     // RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>   // LaunchKernel, SGL_DEVICE, type aliases
#include <sgl_kernel/vec.cuh>     // AlignedVector<T, N> — 128-bit vector loads
#include <sgl_kernel/warp.cuh>    // warp::reduce_sum, warp::reduce_max
#include <sgl_kernel/math.cuh>    // device::math::rsqrt, sqrt, ...
#include <sgl_kernel/tile.cuh>    // tile::Memory (strided access pattern)
```

Key types: `fp16_t` = `__half`, `bf16_t` = `__nv_bfloat16`, `fp32_t` = `float`.
Packed variants: `fp16x2_t`, `bf16x2_t`. Use `packed_t<T>` for the 2-element alias.

---

## Step 1: Write the CUDA Kernel

Create `python/sglang/jit_kernel/csrc/diffusion/rmsnorm.cuh` (RMSNorm as example).

### 1a. Vectorized RMSNorm Kernel

```cpp
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// ---------------------------------------------------------------
// RMSNorm kernel: y = x / rms(x) * weight
// T      = fp16_t | bf16_t | fp32_t
// kVecN  = vectorized elements per load (8 for fp16/bf16, 4 for fp32)
// ---------------------------------------------------------------
template <typename T, int kVecN>
__global__ void rmsnorm_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const T* __restrict__ weight,        // may be nullptr if no affine weight
    uint32_t hidden_size,
    uint32_t n_vecs,                     // hidden_size / kVecN
    float eps)
{
    using vec_t = device::AlignedVector<T, kVecN>;

    const uint32_t row = blockIdx.x;
    const T* row_src = src + row * hidden_size;
    T*       row_dst = dst + row * hidden_size;

    // --- Pass 1: accumulate sum of squares (vectorized) ---
    float sum_sq = 0.f;
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
        vec_t v;
        v.load(row_src, vi);
        #pragma unroll
        for (int i = 0; i < kVecN; ++i) {
            float val = static_cast<float>(v[i]);
            sum_sq += val * val;
        }
    }

    // --- Warp reduction ---
    sum_sq = device::warp::reduce_sum<float>(sum_sq);

    // --- Block reduction via shared memory ---
    __shared__ float smem[32];
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = sum_sq;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.f;
        sum_sq = device::warp::reduce_sum<float>(sum_sq);
    }
    __syncthreads();

    const float rms_inv = device::math::rsqrt<float>(sum_sq / static_cast<float>(hidden_size) + eps);

    // --- Pass 2: normalize + apply weight (vectorized) ---
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
        vec_t v_in, v_w, v_out;
        v_in.load(row_src, vi);
        if (weight != nullptr) {
            v_w.load(weight, vi);
        }
        #pragma unroll
        for (int i = 0; i < kVecN; ++i) {
            float val = static_cast<float>(v_in[i]) * rms_inv;
            if (weight != nullptr) {
                val *= static_cast<float>(v_w[i]);
            }
            v_out[i] = static_cast<T>(val);
        }
        v_out.store(row_dst, vi);
    }
}

// ---------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------
template <typename T>
void rmsnorm(
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView weight,          // pass empty / nullptr for no-weight case
    float eps)
{
    using namespace host;

    // Validate
    SymbolicSize B{"batch_tokens"}, H{"hidden_size"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({B, H})
        .with_dtype<T>()
        .with_device(device)
        .verify(dst)
        .verify(src);

    const uint32_t num_rows   = static_cast<uint32_t>(B.unwrap());
    const uint32_t hidden     = static_cast<uint32_t>(H.unwrap());
    const DLDevice dev        = device.unwrap();

    RuntimeCheck(hidden % (16 / sizeof(T)) == 0,
        "rmsnorm: hidden_size must be divisible by vector width, got ", hidden);

    constexpr int kVecN    = 16 / sizeof(T);   // 128-bit vector: 8×fp16/bf16, 4×fp32
    const uint32_t n_vecs  = hidden / kVecN;

    // Thread count: enough warps to cover n_vecs, max 512 threads
    uint32_t threads = std::min(n_vecs, 512u);
    threads = (threads + 31) / 32 * 32;   // round up to warp boundary

    const T* w_ptr = (weight.data_ptr() != nullptr)
        ? static_cast<const T*>(weight.data_ptr()) : nullptr;

    LaunchKernel(num_rows, threads, dev)(
        rmsnorm_kernel<T, kVecN>,
        static_cast<T*>(dst.data_ptr()),
        static_cast<const T*>(src.data_ptr()),
        w_ptr,
        hidden,
        n_vecs,
        eps);
}

}  // namespace
```

---

## Step 2: Python Wrapper

Create `python/sglang/jit_kernel/diffusion/rmsnorm.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rmsnorm_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_rmsnorm",
        *args,
        cuda_files=["diffusion/rmsnorm.cuh"],    # relative to csrc/
        cuda_wrappers=[("rmsnorm", f"rmsnorm<{args}>")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def diffusion_rmsnorm(
    src: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    RMSNorm for diffusion DiT layers.

    y = x / rms(x) * weight   (weight=None → no affine scaling)

    Supported dtypes: float16, bfloat16, float32.
    hidden_size must be divisible by 8 (fp16/bf16) or 4 (fp32).
    """
    assert src.is_cuda, "src must be a CUDA tensor"
    assert src.dtype in (torch.float16, torch.bfloat16, torch.float32)

    if out is None:
        out = torch.empty_like(src)

    # Pass a zero-sized tensor when weight is absent (launcher checks data_ptr == nullptr)
    w = weight if weight is not None else torch.empty(0, dtype=src.dtype, device=src.device)

    module = _jit_rmsnorm_module(src.dtype)
    module.rmsnorm(out, src, w, eps)
    return out
```

**Key rules for the wrapper:**
- Use `cache_once` — never `functools.lru_cache` (breaks `torch.compile`)
- First arg(s) to `load_jit` form the unique build cache key
- `cuda_files` are relative to `python/sglang/jit_kernel/csrc/`
- `cuda_wrappers`: `(python_name, cpp_template_instantiation)`

---

## Step 3: Integrate into Denoising Stage

The kernel replaces a slow operator inside the DiT forward pass. Find the correct module in:

```
python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
python/sglang/multimodal_gen/runtime/models/dits/<model>.py
```

**Pattern — monkey-patch the DiT block's RMSNorm:**

```python
from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm

def _patch_rmsnorm(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in ("RMSNorm", "LlamaRMSNorm") or "RMSNorm" in cls_name:
            eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
            has_weight = hasattr(module, "weight") and module.weight is not None

            if has_weight:
                def _make_fwd(mod, epsilon):
                    def forward(x):
                        return diffusion_rmsnorm(x, weight=mod.weight, eps=epsilon)
                    return forward
                module.forward = _make_fwd(module, eps)
            else:
                def _make_fwd_noweight(epsilon):
                    def forward(x):
                        return diffusion_rmsnorm(x, weight=None, eps=epsilon)
                    return forward
                module.forward = _make_fwd_noweight(eps)
```

**Critical:** inject kernels **before** `torch.compile` and before any CPU offload is enabled.

---

## Step 4: Key Kernel Patterns Reference

### Diffusion-Specific Operators

| Operator | Kernel Pattern | Notes |
|----------|---------------|-------|
| **RMSNorm** | 2-pass row reduction + vectorized normalize | Weight may be `None` (`elementwise_affine=False`) |
| **AdaLN modulation** | `y = norm(x) * (1 + scale) + shift` | Fuse norm + scale + shift in one pass |
| **RoPE 3D** | Read `(t, h, w)` cos/sin tables, apply to `(q, k)` | Layout: `[batch, t*h*w, heads, head_dim]` |
| **GEGLU** | Split last dim → `gate * silu(linear)` | Input `[B, L, 2*H]` → output `[B, L, H]` |
| **SiLU gate** | `out = a * sigmoid(a)` fused | Avoid separate elementwise ops |

### Vectorized Memory Access

```cpp
// BF16: 8 elements × 2 bytes = 16 bytes per vector load (AlignedVector<bf16_t, 8>)
// FP16: 8 elements × 2 bytes = 16 bytes (AlignedVector<fp16_t, 8>)
// FP32: 4 elements × 4 bytes = 16 bytes (AlignedVector<fp32_t, 4>)
constexpr int kVecN = 16 / sizeof(T);
using vec_t = device::AlignedVector<T, kVecN>;
```

### Warp / Block Reductions

```cpp
// Warp reduction (within 32 threads)
float result = device::warp::reduce_sum<float>(partial);

// Block reduction via shared memory (see rmsnorm example above)
__shared__ float smem[32];
// ... write warp-leaders into smem, sync, reduce again
```

### Thread Configuration

```cpp
// Element-wise (RoPE, GEGLU, SiLU): simple 1D grid
constexpr uint32_t kBlock = 256;
uint32_t grid = host::div_ceil(total_elements, kBlock);
LaunchKernel(grid, kBlock, dev)(kernel, ...);

// Row reduction (RMSNorm, LayerNorm): one block per row
uint32_t threads = std::min(hidden_size / kVecN, 512u);
threads = (threads + 31) / 32 * 32;
LaunchKernel(num_rows, threads, dev)(kernel, ...);
```

---

## Step 5: GPU Architecture Targets

| GPU | Compute Cap | Memory BW | BF16 | Key Note |
|-----|------------|-----------|------|----------|
| H100 | sm_90 | 3.35 TB/s | Yes | Primary target; 132 SMs, 192 KB shared mem/SM |
| A100 | sm_80 | 2.0 TB/s  | Yes | 108 SMs, 164 KB shared mem/SM |
| T4   | sm_75 | 320 GB/s  | **No** | FP16 only; no `__nv_bfloat16` |

If kernel requires SM90+ features (e.g., TMA, wgmma), raise a clear error:

```python
if torch.cuda.get_device_capability()[0] < 9:
    raise RuntimeError("This kernel requires SM90 (H100/Hopper) or later")
```

**Grid sizing for H100** (132 SMs): aim for grid multiples of 132 for good occupancy.

---

## Step 6: Tests

Create `python/sglang/jit_kernel/tests/test_diffusion_rmsnorm.py`:

```python
import pytest
import torch
from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 2048), (4, 3072), (16, 4096)])
@pytest.mark.parametrize("has_weight", [True, False])
def test_rmsnorm_correctness(dtype, shape, has_weight):
    batch, hidden = shape
    src = torch.randn(batch, hidden, dtype=dtype, device="cuda")
    weight = torch.randn(hidden, dtype=dtype, device="cuda") if has_weight else None

    out_jit = diffusion_rmsnorm(src, weight=weight, eps=1e-6)

    # Reference: torch.nn.functional
    ref = torch.nn.functional.rms_norm(
        src.float(), (hidden,), weight.float() if weight is not None else None, eps=1e-6
    ).to(dtype)

    tol = {"rtol": 1e-2, "atol": 1e-2} if dtype != torch.float32 else {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(out_jit, ref, **tol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

---

## Step 7: Benchmark

Create `python/sglang/jit_kernel/benchmark/bench_diffusion_rmsnorm.py`:

```python
import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark
from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm

SHAPES = [(4096, 2048), (4096, 3072), (4096, 4096)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden"],
        x_vals=[s[1] for s in SHAPES],
        line_arg="provider",
        line_vals=["jit_cuda", "torch"],
        line_names=["SGLang JIT CUDA", "PyTorch rms_norm"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="diffusion-rmsnorm",
        args={},
    )
)
def benchmark(hidden: int, provider: str):
    src = torch.randn(4096, hidden, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    w   = torch.ones(hidden, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    if provider == "jit_cuda":
        fn = lambda: diffusion_rmsnorm(src, weight=w, eps=1e-6)
    else:
        fn = lambda: torch.nn.functional.rms_norm(src, (hidden,), w, eps=1e-6)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
```

---

## Step 8: Profile with Nsight Compute (required)

After correctness + benchmarking, you must collect **Nsight Compute (ncu)** data to validate:

- Whether the kernel reaches reasonable bandwidth/throughput (avoid false positives where it is “faster” but under-utilizes hardware)
- Whether there are clear occupancy / register / shared memory limiters

Use the canonical docs in this directory (do not duplicate CLI details across multiple skills):

- `diffusion-benchmark-and-profile.md` → Step 3.5 (ncu workflow, including CUDA graph profiling)
- `nsight-profiler.md` (metrics interpretation: bandwidth / occupancy / roofline / stall reasons)

---

## Common Pitfalls

| Issue | Fix |
|-------|-----|
| `RMSNorm weight is None` | Use `type(module).__name__` check; pass `None` weight explicitly |
| `isinstance(m, torch.nn.RMSNorm)` misses diffusers variants | Use `"RMSNorm" in type(m).__name__` |
| Kernel patched after `torch.compile` | Inject **before** any compile call |
| Kernel patched after `enable_model_cpu_offload()` | Inject **before** CPU offload |
| `hidden_size` not divisible by `kVecN` | Add `RuntimeCheck(hidden % kVecN == 0, ...)` in launcher |
| `torch.compile` fails with custom CUDA kernel | Register as `@torch.library.custom_op` or use Triton instead |
| T4 GPU with BF16 kernel | Gate on compute capability; T4 is `sm_75`, no native BF16 |

---

## Summary of Files

```
python/sglang/jit_kernel/csrc/diffusion/
└── rmsnorm.cuh                                  # NEW: JIT CUDA kernel source

python/sglang/jit_kernel/diffusion/
└── rmsnorm.py                                   # NEW: Python wrapper + load_jit

python/sglang/jit_kernel/tests/
└── test_diffusion_rmsnorm.py                    # NEW: correctness tests

python/sglang/jit_kernel/benchmark/
└── bench_diffusion_rmsnorm.py                   # NEW: benchmark
```

---

## References

### This Skill's Extended Docs (references/ and scripts/)

| File | Contents |
|------|----------|
| [references/kernel-templates.md](references/kernel-templates.md) | Copy-paste templates: element-wise, RMSNorm, AdaLN, Python wrapper, test, benchmark |
| [references/troubleshooting.md](references/troubleshooting.md) | Build errors, perf issues, torch.compile compatibility, debugging checklist |
| [references/h100-optimization-guide.md](references/h100-optimization-guide.md) | H100 (sm_90): memory hierarchy, warp reductions, occupancy, vectorization benchmarks |
| [references/a100-optimization-guide.md](references/a100-optimization-guide.md) | A100 (sm_80): cp.async, TF32, 2:4 sparsity, H100→A100 migration checklist |
| [references/t4-optimization-guide.md](references/t4-optimization-guide.md) | T4 (sm_75): FP16 only, low bandwidth, tile size limits, memory constraints |
| [scripts/bench_diffusion_rmsnorm.py](scripts/bench_diffusion_rmsnorm.py) | Micro-benchmark: JIT CUDA RMSNorm vs PyTorch, correctness check, bandwidth analysis |
| [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) | End-to-end: `sglang generate` baseline vs custom kernels, comparison table |

### SGLang Internals

- **JIT system**: `add-jit-kernel` skill (`sglang/.claude/skills/add-jit-kernel/SKILL.md`)
- **JIT utils**: `python/sglang/jit_kernel/utils.py` — `cache_once`, `load_jit`, `make_cpp_args`
- **Abstractions**: `python/sglang/jit_kernel/include/sgl_kernel/` — `tensor.h`, `utils.cuh`, `vec.cuh`, `warp.cuh`, `math.cuh`, `tile.cuh`
- **Real csrc examples**: `python/sglang/jit_kernel/csrc/elementwise/rmsnorm.cuh`, `python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh`

### Other Diffusion Kernel Skills (this directory)

- **Triton alternative**: `add-triton-kernel.md` — prefer Triton unless bandwidth analysis shows CUDA needed
- **Existing fused kernels**: `use-efficient-diffusion-kernels.md` — check here first before writing new kernels
- **Profiling**: `diffusion-benchmark-and-profile.md` — workflow to identify bottleneck before implementing
- **Nsight Compute deep dive**: `nsight-profiler.md` — full guide: occupancy analysis, roofline model, warp efficiency, kernel comparison

### External

- [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels) — original source adapted for this skill
