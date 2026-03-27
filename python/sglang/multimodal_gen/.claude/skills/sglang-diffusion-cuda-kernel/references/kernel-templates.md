# CUDA Kernel Templates — SGLang Diffusion JIT Style

Copy-paste ready templates for JIT CUDA kernels in `python/sglang/jit_kernel/csrc/diffusion/`.
All templates use SGLang's internal abstractions; no raw CUDA headers needed.

> **Adapted from**: [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels)

---

## Prerequisite: Standard Includes

Every kernel file in `csrc/diffusion/` starts with:

```cpp
#include <sgl_kernel/tensor.h>    // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>    // fp16_t, bf16_t, fp32_t, dtype_trait, packed_t
#include <sgl_kernel/utils.h>     // RuntimeCheck, Panic, div_ceil
#include <sgl_kernel/utils.cuh>   // LaunchKernel, SGL_DEVICE, type aliases
#include <sgl_kernel/vec.cuh>     // AlignedVector<T, N>
#include <sgl_kernel/warp.cuh>    // warp::reduce_sum, warp::reduce_max
#include <sgl_kernel/math.cuh>    // device::math::rsqrt, sqrt, ...
#include <sgl_kernel/tile.cuh>    // tile::Memory (strided access pattern)

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
```

**Key type aliases** (from `utils.cuh`):
- `fp16_t` = `__half`, `fp16x2_t` = `__half2`
- `bf16_t` = `__nv_bfloat16`, `bf16x2_t` = `__nv_bfloat162`
- `fp32_t` = `float`, `fp32x2_t` = `float2`
- `SGL_DEVICE` = `__forceinline__ __device__`

---

## Template 1: Element-wise Operation

Use for ops that process elements independently: RoPE, SiLU, GEGLU, scale+bias.

### `.cuh` file: `csrc/diffusion/silu_gate.cuh`

```cpp
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/math.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// SiLU gate: out[i] = x[i] * sigmoid(x[i])
// Input layout: [B, L, hidden]
template <typename T, int kVecN>
__global__ void silu_gate_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    uint32_t n_vecs,
    uint32_t n_remainder,
    uint32_t n_total)
{
    using vec_t = device::AlignedVector<T, kVecN>;

    const uint32_t stride = blockDim.x * gridDim.x;

    // --- vectorized body ---
    for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x; vi < n_vecs; vi += stride) {
        vec_t v;
        v.load(src, vi);
        #pragma unroll
        for (int i = 0; i < kVecN; ++i) {
            float val = static_cast<float>(v[i]);
            float sig = 1.f / (1.f + device::math::exp<float>(-val));
            v[i] = static_cast<T>(val * sig);
        }
        v.store(dst, vi);
    }

    // --- scalar tail (for sizes not divisible by kVecN) ---
    const uint32_t base = n_vecs * kVecN;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_remainder; i += stride) {
        float val = static_cast<float>(src[base + i]);
        float sig = 1.f / (1.f + device::math::exp<float>(-val));
        dst[base + i] = static_cast<T>(val * sig);
    }
}

template <typename T>
void silu_gate(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
    using namespace host;

    SymbolicSize N{"num_elements"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({N})
        .with_dtype<T>()
        .with_device(device)
        .verify(dst)
        .verify(src);

    const uint32_t n      = static_cast<uint32_t>(N.unwrap());
    const DLDevice dev    = device.unwrap();
    RuntimeCheck(n > 0, "silu_gate: num_elements must be > 0");

    constexpr int kVecN      = 16 / sizeof(T);   // 128-bit vector load
    const uint32_t n_vecs    = n / kVecN;
    const uint32_t n_rem     = n % kVecN;

    constexpr uint32_t kBlock = 256;
    const uint32_t grid       = div_ceil(std::max(n_vecs, n_rem), kBlock);

    LaunchKernel(grid, kBlock, dev)(
        silu_gate_kernel<T, kVecN>,
        static_cast<T*>(dst.data_ptr()),
        static_cast<const T*>(src.data_ptr()),
        n_vecs, n_rem, n);
}

}  // namespace
```

### Python wrapper: `diffusion/silu_gate.py`

```python
from __future__ import annotations
import torch
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

@cache_once
def _jit_silu_gate_module(dtype: torch.dtype):
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_silu_gate",
        *args,
        cuda_files=["diffusion/silu_gate.cuh"],
        cuda_wrappers=[("silu_gate", f"silu_gate<{args}>")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )

def diffusion_silu_gate(src: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    assert src.is_cuda and src.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if out is None:
        out = torch.empty_like(src)
    module = _jit_silu_gate_module(src.dtype)
    module.silu_gate(out, src)
    return out
```

---

## Template 2: Row-wise Reduction (RMSNorm / LayerNorm)

Use for ops that reduce across the last dimension of each row.

### `.cuh` file: `csrc/diffusion/rmsnorm.cuh`

```cpp
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/math.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// RMSNorm: y = x / rms(x) * weight
// One block per row; vectorized loads/stores; warp + shared-mem reduction
template <typename T, int kVecN>
__global__ void rmsnorm_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const T* __restrict__ weight,   // nullptr if no affine weight
    uint32_t hidden,
    uint32_t n_vecs,
    float eps)
{
    using vec_t = device::AlignedVector<T, kVecN>;

    const uint32_t row     = blockIdx.x;
    const T* row_src       = src + row * hidden;
    T*       row_dst       = dst + row * hidden;

    // Pass 1: sum of squares
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

    // Warp + block reduction
    sum_sq = device::warp::reduce_sum<float>(sum_sq);
    __shared__ float smem[32];
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.f;
        sum_sq = device::warp::reduce_sum<float>(sum_sq);
    }
    __syncthreads();

    const float rms_inv = device::math::rsqrt<float>(sum_sq / static_cast<float>(hidden) + eps);

    // Pass 2: normalize + optional weight
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
        vec_t v_in, v_out;
        v_in.load(row_src, vi);
        if (weight != nullptr) {
            vec_t v_w;
            v_w.load(weight, vi);
            #pragma unroll
            for (int i = 0; i < kVecN; ++i)
                v_out[i] = static_cast<T>(static_cast<float>(v_in[i]) * rms_inv
                                         * static_cast<float>(v_w[i]));
        } else {
            #pragma unroll
            for (int i = 0; i < kVecN; ++i)
                v_out[i] = static_cast<T>(static_cast<float>(v_in[i]) * rms_inv);
        }
        v_out.store(row_dst, vi);
    }
}

template <typename T>
void rmsnorm(
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView weight,   // data_ptr == nullptr → no weight
    float eps)
{
    using namespace host;

    SymbolicSize B{"batch_tokens"}, H{"hidden_size"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({B, H})
        .with_dtype<T>()
        .with_device(device)
        .verify(dst)
        .verify(src);

    const uint32_t num_rows = static_cast<uint32_t>(B.unwrap());
    const uint32_t hidden   = static_cast<uint32_t>(H.unwrap());
    const DLDevice dev      = device.unwrap();

    constexpr int kVecN = 16 / sizeof(T);
    RuntimeCheck(hidden % kVecN == 0,
        "rmsnorm: hidden_size (", hidden, ") must be divisible by ", kVecN);
    const uint32_t n_vecs = hidden / kVecN;

    uint32_t threads = std::min(n_vecs, 512u);
    threads = (threads + 31) / 32 * 32;

    const T* w_ptr = (weight.data_ptr() != nullptr)
        ? static_cast<const T*>(weight.data_ptr()) : nullptr;

    LaunchKernel(num_rows, threads, dev)(
        rmsnorm_kernel<T, kVecN>,
        static_cast<T*>(dst.data_ptr()),
        static_cast<const T*>(src.data_ptr()),
        w_ptr, hidden, n_vecs, eps);
}

}  // namespace
```

---

## Template 3: Fused Row-Reduction + Element-wise (AdaLN)

Combines RMSNorm + AdaLN modulation into one pass: `y = norm(x) * (1 + scale) + shift`.

### `.cuh` file: `csrc/diffusion/adaln.cuh`

```cpp
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>
#include <sgl_kernel/math.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// AdaLN: y = norm(x) * (1 + scale) + shift
// scale, shift: [batch, hidden] (one per row)
template <typename T, int kVecN>
__global__ void adaln_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const T* __restrict__ weight,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    uint32_t hidden,
    uint32_t n_vecs,
    float eps)
{
    using vec_t = device::AlignedVector<T, kVecN>;

    const uint32_t row     = blockIdx.x;
    const T* row_src       = src   + row * hidden;
    const T* row_scale     = scale + row * hidden;
    const T* row_shift     = shift + row * hidden;
    T*       row_dst       = dst   + row * hidden;

    // Pass 1: compute RMS
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
    sum_sq = device::warp::reduce_sum<float>(sum_sq);
    __shared__ float smem[32];
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = sum_sq;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum_sq = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.f;
        sum_sq = device::warp::reduce_sum<float>(sum_sq);
    }
    __syncthreads();
    const float rms_inv = device::math::rsqrt<float>(sum_sq / static_cast<float>(hidden) + eps);

    // Pass 2: normalize + modulate
    for (uint32_t vi = threadIdx.x; vi < n_vecs; vi += blockDim.x) {
        vec_t v_in, v_w, v_sc, v_sh, v_out;
        v_in.load(row_src, vi);
        v_w.load(weight,   vi);
        v_sc.load(row_scale, vi);
        v_sh.load(row_shift, vi);
        #pragma unroll
        for (int i = 0; i < kVecN; ++i) {
            float x  = static_cast<float>(v_in[i]) * rms_inv * static_cast<float>(v_w[i]);
            float sc = static_cast<float>(v_sc[i]);
            float sh = static_cast<float>(v_sh[i]);
            v_out[i] = static_cast<T>(x * (1.f + sc) + sh);
        }
        v_out.store(row_dst, vi);
    }
}

template <typename T>
void adaln(
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scale,
    tvm::ffi::TensorView shift,
    float eps)
{
    using namespace host;

    SymbolicSize B{"batch_tokens"}, H{"hidden_size"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();

    TensorMatcher({B, H})
        .with_dtype<T>()
        .with_device(device)
        .verify(dst).verify(src).verify(weight).verify(scale).verify(shift);

    const uint32_t num_rows = static_cast<uint32_t>(B.unwrap());
    const uint32_t hidden   = static_cast<uint32_t>(H.unwrap());
    const DLDevice dev      = device.unwrap();

    constexpr int kVecN = 16 / sizeof(T);
    RuntimeCheck(hidden % kVecN == 0, "adaln: hidden_size must be divisible by ", kVecN);
    const uint32_t n_vecs = hidden / kVecN;

    uint32_t threads = std::min(n_vecs, 512u);
    threads = (threads + 31) / 32 * 32;

    LaunchKernel(num_rows, threads, dev)(
        adaln_kernel<T, kVecN>,
        static_cast<T*>(dst.data_ptr()),
        static_cast<const T*>(src.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        static_cast<const T*>(scale.data_ptr()),
        static_cast<const T*>(shift.data_ptr()),
        hidden, n_vecs, eps);
}

}  // namespace
```

---

## Template 4: Python Wrapper (generic pattern)

File location: `python/sglang/jit_kernel/diffusion/<op>.py`

```python
from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    """Cache key: dtype (and any other template params you need)."""
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_your_op",           # unique build cache key
        *args,
        cuda_files=["diffusion/your_op.cuh"],  # relative to csrc/
        cuda_wrappers=[("your_op", f"your_op<{args}>")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def diffusion_your_op(
    src: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Your op description.

    Supported dtypes: float16, bfloat16, float32.
    """
    assert src.is_cuda, "src must be a CUDA tensor"
    assert src.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"Unsupported dtype {src.dtype}"
    )
    if out is None:
        out = torch.empty_like(src)

    module = _jit_module(src.dtype)
    module.your_op(out, src)
    return out
```

**`make_cpp_args` conversion table:**

| `torch.dtype` | C++ type |
|---------------|----------|
| `torch.float16` | `fp16_t` |
| `torch.bfloat16` | `bf16_t` |
| `torch.float32` | `fp32_t` |

---

## Template 5: Correctness Test

```python
# python/sglang/jit_kernel/tests/test_diffusion_<op>.py
import pytest
import torch
from sglang.jit_kernel.diffusion.<op> import diffusion_<op>


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 2048), (4, 3072), (16, 4096)])
def test_<op>_correctness(dtype, shape):
    src = torch.randn(*shape, dtype=dtype, device="cuda")

    out_jit = diffusion_<op>(src)
    ref     = reference_<op>(src.float()).to(dtype)  # reference in fp32

    tol = {"rtol": 1e-2, "atol": 1e-2} if dtype != torch.float32 else {"rtol": 1e-5, "atol": 1e-6}
    torch.testing.assert_close(out_jit, ref, **tol)


def test_<op>_out_param():
    src = torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(src)
    result = diffusion_<op>(src, out=out)
    assert result is out


def test_<op>_cpu_error():
    src = torch.randn(128, dtype=torch.float16)  # CPU tensor
    with pytest.raises(AssertionError):
        diffusion_<op>(src)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
```

---

## Template 6: Benchmark

```python
# python/sglang/jit_kernel/benchmark/bench_diffusion_<op>.py
import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark
from sglang.jit_kernel.diffusion.<op> import diffusion_<op>

SHAPES = [(4096, 2048), (4096, 3072), (4096, 4096)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden"],
        x_vals=[s[1] for s in SHAPES],
        line_arg="provider",
        line_vals=["jit_cuda", "torch"],
        line_names=["SGLang JIT CUDA", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="diffusion-<op>",
        args={},
    )
)
def benchmark(hidden: int, provider: str):
    src = torch.randn(4096, hidden, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    if provider == "jit_cuda":
        fn = lambda: diffusion_<op>(src)
    else:
        fn = lambda: reference_<op>(src)  # torch baseline

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
```

---

## Summary of New Files per Kernel

```
python/sglang/jit_kernel/csrc/diffusion/
└── <op>.cuh                               # CUDA kernel + launcher

python/sglang/jit_kernel/diffusion/
└── <op>.py                                # Python wrapper (load_jit + cache_once)

python/sglang/jit_kernel/tests/
└── test_diffusion_<op>.py                 # correctness tests

python/sglang/jit_kernel/benchmark/
└── bench_diffusion_<op>.py                # triton.testing benchmark
```

> See `scripts/bench_diffusion_rmsnorm.py` and `scripts/bench_diffusion_denoise.py` for full runnable examples.
