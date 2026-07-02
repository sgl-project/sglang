---
name: add-jit-kernel
description: Step-by-step tutorial for adding a new lightweight JIT CUDA kernel to sglang's jit_kernel module
---

# Tutorial: Adding a New JIT Kernel to SGLang

This tutorial walks through adding a simple element-wise scale operation as a JIT kernel. We'll implement `scale(x, factor) = x * factor` to demonstrate the complete workflow.

## Goal

Add a new operation that scales each element of a tensor by a scalar factor:

- Input: tensor `x` (CUDA) and scalar `factor` (float, passed at runtime)
- Output: `x * factor` (element-wise), allocated internally
- Supported dtypes: **FP16 (`torch.float16`), BF16 (`torch.bfloat16`), FP32 (`torch.float32`)**

## When to use JIT vs AOT (`sgl-kernel`)

- **JIT (`jit_kernel`)**: prefer this first for kernels that do **not** depend on CUTLASS or another large C++ project. It is the default choice for lightweight kernels that benefit from rapid iteration and first-use compilation.
- **AOT (`sgl-kernel`)**: prefer this when the kernel **does** depend on CUTLASS or another large C++ project, or when it should live in `sgl-kernel/` and participate in the wheel build / torch op registration flow.
- **Exception**: kernels that depend on `flashinfer`, or on CUTLASS that is already provided through `flashinfer`, can still be implemented as `jit_kernel`.

---

## Common Abstractions in `python/sglang/jit_kernel/include/sgl_kernel/`

**Always prefer these abstractions over raw CUDA primitives.** They provide safety, readability, and consistency with the rest of the codebase.

**Important include rule:** for every `#include <sgl_kernel/...>` line, add a short trailing comment explaining why that header is included (for example `// For TensorMatcher, SymbolicSize, SymbolicDevice`). This matches the current JIT kernel style and keeps include usage self-documenting.

### `utils.h` — Host-side utilities

```cpp
#include <sgl_kernel/utils.h>
```

- **`host::RuntimeCheck(cond, args...)`** — Assert a condition at runtime; throws `PanicError` with file/line info on failure. Prefer this over bare `assert`.
- **`host::Panic(args...)`** — Unconditionally throw a `PanicError` with a descriptive message.
- **`host::div_ceil(a, b)`** — Integer ceiling division `(a + b - 1) / b`.
- **`host::irange(n)`** / **`host::irange(start, end)`** — Range views for cleaner loops.
- **`host::pointer::offset(ptr, offsets...)`** — Byte-safe pointer arithmetic on `void*`. Use this instead of raw casts.

### `utils.cuh` — Device-side utilities + `LaunchKernel`

```cpp
#include <sgl_kernel/utils.cuh>
```

- **Type aliases**: `fp16_t`, `bf16_t`, `fp32_t`, `fp8_e4m3_t`, `fp8_e5m2_t` and their packed variants `fp16x2_t`, `bf16x2_t`, `fp32x2_t`, etc.
- **`SGL_DEVICE`** — Expands to `__forceinline__ __device__`. Use on all device functions.
- **`device::kWarpThreads`** — Constant `32`.
- **`device::load_as<T>(ptr, offset)`** / **`device::store_as<T>(ptr, val, offset)`** — Type-safe loads/stores from `void*`.
- **`device::pointer::offset(ptr, offsets...)`** — Pointer arithmetic on device.
- **`host::LaunchKernel(grid, block, device_or_stream [, smem])`** — RAII kernel launcher that:
  - Resolves the CUDA stream from a `DLDevice` via TVM-FFI automatically.
  - Checks the CUDA error with file/line info after launch via `operator()(kernel, args...)`.
  - Supports `.enable_pdl(bool)` for PDL (Programmatic Dependent Launch, SM90+).
- **`host::RuntimeDeviceCheck(cudaError_t)`** — Check a CUDA error; throw on failure.

### `tensor.h` — Tensor validation (`TensorMatcher`, Symbolic types)

```cpp
#include <sgl_kernel/tensor.h>
```

This is the **primary validation API** for all kernel launchers. Use it to validate every `tvm::ffi::TensorView` argument.

- **`host::SymbolicSize{"name"}`** — A named symbolic dimension. Call `.set_value(n)` to pin it, `.unwrap()` to extract after verification.
- **`host::SymbolicDType`** — Symbolic dtype. Use `.set_options<Ts...>()` to restrict allowed types.
- **`host::SymbolicDevice`** — Symbolic device. Use `.set_options<kDLCUDA>()` to restrict to CUDA.
- **`host::TensorMatcher({dims...})`** — Fluent builder for tensor validation:
  - `.with_dtype<T>()` — require a specific C++ type (e.g. `fp16_t`)
  - `.with_dtype<T1, T2, ...>()` — allow a set of types
  - `.with_device<kDLCUDA>(device_sym)` — require CUDA and bind the checked device to a `SymbolicDevice`
  - `.with_strides({strides...})` — validate strides (omit to require contiguous)
  - `.verify(tensor_view)` — execute the check; throws `PanicError` with full context on failure; **chainable** (`verify(a).verify(b)` to check multiple tensors with the same shape)

**Typical pattern:**
```cpp
auto N = SymbolicSize{"num_elements"};
auto device = SymbolicDevice{};
device.set_options<kDLCUDA>();
TensorMatcher({N})  //
    .with_dtype<fp16_t>()
    .with_device<kDLCUDA>(device)
    .verify(dst)
    .verify(src);  // same shape, dtype, device as dst
const size_t n = N.unwrap();
const DLDevice dev = device.unwrap();
```

### `type.cuh` — `dtype_trait<T>` and `packed_t<T>`

```cpp
#include <sgl_kernel/type.cuh>
```

- **`dtype_trait<T>`** — Static trait struct for each scalar type. Provides:
  - `dtype_trait<T>::from(value)` — convert from another type (e.g. `fp32_t` → `fp16_t`)
  - `dtype_trait<T>::abs/sqrt/rsqrt/exp/sin/cos(x)` — type-dispatched unary math (primarily for `fp32_t`)
  - `dtype_trait<T>::max/min(x, y)` — type-dispatched binary math (primarily for `fp32_t`)
- **`packed_t<T>`** — Two-element packed alias: `packed_t<fp16_t>` = `fp16x2_t`, `packed_t<bf16_t>` = `bf16x2_t`, `packed_t<fp32_t>` = `fp32x2_t`. Use for vectorized loads/stores.
- **`device::cast<To, From>(value)`** — Type-safe cast using `dtype_trait`, e.g. `cast<fp32x2_t, fp16x2_t>(v)`.

### `vec.cuh` — Vectorized memory access (`AlignedVector`)

```cpp
#include <sgl_kernel/vec.cuh>
```

- **`device::AlignedVector<T, N>`** — Aligned storage for N elements of type T. N must be a power of two, `sizeof(T)*N <= 32`. Enables vectorized loads/stores for bandwidth efficiency. In terms of API/codegen constraints, the upper bound is 256-bit; in practice, 128-bit is the portable default, while 256-bit vectorization is typically only viable on `SM100+` and should be gated by an architecture check when needed.
  - `.load(ptr, offset)` — vectorized load from `ptr[offset]`
  - `.store(ptr, offset)` — vectorized store to `ptr[offset]`
  - `.fill(value)` — fill all lanes
  - `operator[](i)` — element access

### `tile.cuh` — `tile::Memory` (strided memory access pattern)

```cpp
#include <sgl_kernel/tile.cuh>
```

- `tile::Memory<T>` is fundamentally a **1D cooperative accessor** over a contiguous region.
- **`device::tile::Memory<T>::cta(blockDim.x)`** — Creates a tile accessor where each thread handles `tid = threadIdx.x` with stride `tsize` (for `cta(blockDim.x)`, this is `blockDim.x`). Common for loops over a 1D array.
- **`.load(ptr, offset)`** — loads `ptr[tid + offset * tsize]`
- **`.store(ptr, val, offset)`** — stores to `ptr[tid + offset * tsize]`
- **`.in_bound(n, offset)`** — boundary check

For a **2D tile**, either flatten `(row, col)` into a linear tile index first, or compute the address manually with `ptr[row * stride + col]` using your thread/block coordinates.

### `math.cuh` — Device math (`device::math::`)

```cpp
#include <sgl_kernel/math.cuh>
```

- `device::math::max/min<T>(a, b)` — type-dispatched binary math via `dtype_trait`
- `device::math::abs/sqrt/rsqrt/exp/sin/cos<T>(x)` — type-dispatched unary math via `dtype_trait`

### `warp.cuh` — Warp-level primitives

```cpp
#include <sgl_kernel/warp.cuh>
```

- `device::warp::reduce_sum<T>(value)` — warp-level sum reduction via `__shfl_xor_sync`
- `device::warp::reduce_max<T>(value)` — warp-level max reduction

### `cta.cuh` — CTA-level primitives

```cpp
#include <sgl_kernel/cta.cuh>
```

- `device::cta::reduce_max<T>(value, smem, min_value)` — CTA-wide max using shared memory + warp reduction. Caller is responsible for a `__syncthreads()` after if the result in `smem[0]` is needed.

### `atomic.cuh` — Atomic operations

```cpp
#include <sgl_kernel/atomic.cuh>
```

- `device::atomic::max(float* addr, float value)` — float atomic max (handles negative values correctly via bit tricks).

### `runtime.cuh` — Occupancy and device info

```cpp
#include <sgl_kernel/runtime.cuh>
```

- `host::runtime::get_blocks_per_sm(kernel, block_dim)` — max active blocks per SM (occupancy)
- `host::runtime::get_sm_count(device_id)` — number of SMs on the device
- `host::runtime::get_cc_major(device_id)` — compute capability major version

**Persistent kernel pattern** (cap blocks to SM count × occupancy):
```cpp
static const uint32_t max_occ = runtime::get_blocks_per_sm(kernel, kBlockSize);
static const uint32_t num_sm  = runtime::get_sm_count(device.unwrap().device_id);
const auto num_blocks = std::min(num_sm * max_occ, div_ceil(n, kBlockSize));
LaunchKernel(num_blocks, kBlockSize, device.unwrap())(kernel, params);
```

---

## Step 0 (optional): Generate a `.clangd` config for better IDE support

```bash
python -m sglang.jit_kernel -h  # for verbose help info about clangd configuration
python -m sglang.jit_kernel
python -m sglang.jit_kernel --dep cutlass flashinfer  # with cutlass/flashinfer dependency
```

---

## Step 1: Implement the CUDA kernel in `jit_kernel/csrc/`

Create `python/sglang/jit_kernel/csrc/elementwise/scale.cuh`.

The implementation fully uses the project abstractions described above:

```cpp
// NOTE: Comments for headers are not common in practice.
// It is only shown here for tutorial purposes to highlight the key abstractions.
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/type.cuh>   // For dtype_trait, fp16_t, bf16_t, fp32_t
#include <sgl_kernel/utils.h>    // For RuntimeCheck, div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

namespace {

// ----------------------------------------------------------------
// Kernel: element-wise scale using vectorized 128-bit loads/stores
// T       = fp16_t | bf16_t | fp32_t
// kVecN   = number of elements per vector load (e.g. 8 for fp16)
// factor  = runtime scale factor
// ----------------------------------------------------------------
template <typename T, int kVecN, bool kUsePDL>
__global__ void scale_kernel(T* __restrict__ dst,
                              const T* __restrict__ src,
                              float factor,
                              uint32_t n_total) {
  using vec_t = device::AlignedVector<T, kVecN>;
  const uint32_t n_vecs = n_total / kVecN;

  // If using PDL, wait for primary kernel before any global memory load.
  // This is NOT a synchronization point, which means some threads can early exit before this.
  device::PDLWaitPrimary<kUsePDL>();

  // --- vectorised body ---
  const uint32_t vec_stride = blockDim.x * gridDim.x;
  for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x;
       vi < n_vecs;
       vi += vec_stride) {
    vec_t v;
    v.load(src, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      v[i] = static_cast<T>(static_cast<float>(v[i]) * factor);
    }
    v.store(dst, vi);
  }

  // --- scalar tail ---
  const uint32_t base = n_vecs * kVecN;
  const uint32_t scalar_stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       base + i < n_total;
       i += scalar_stride) {
    dst[base + i] = static_cast<T>(static_cast<float>(src[base + i]) * factor);
  }

  // If using PDL, signal for the secondary kernel to start after all threads have finished
  // This is NOT a synchronization point, which means some threads can early exit before this.
  device::PDLTriggerSecondary<kUsePDL>();
}

// ----------------------------------------------------------------
// Launcher: validates tensors, selects vector width, launches kernel
// ----------------------------------------------------------------
template <typename T, bool kUsePDL>
void scale(tvm::ffi::TensorView dst, tvm::ffi::TensorView src, float factor) {
  using namespace host;

  // 1. Validate input tensors with TensorMatcher
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<T>()
      .with_device<kDLCUDA>(device_)
      .verify(dst)
      .verify(src);  // same shape / dtype / device as dst

  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(n > 0, "scale: num_elements must be > 0, got ", n);

  // 2. Choose vector width for 128-bit loads (16 bytes)
  //    fp16/bf16: 8 elements x 2 bytes = 16 bytes
  //    fp32:      4 elements x 4 bytes = 16 bytes
  // We encourage using `device::kMaxVecBytes`, which will change according to
  // the target architecture and can enable 256-bit vectorization on SM100+ if desired.
  // But 128-bit is more commonly adapted for better compatibility,
  // so it's still ok to hardcode 16 here just for simplicity.
  constexpr int kVecN = 16 / sizeof(T);
  const uint32_t n_work_items = div_ceil(n, static_cast<uint32_t>(kVecN));

  // 3. Launch
  constexpr uint32_t kBlockSize = 256;
  const uint32_t grid = div_ceil(n_work_items, kBlockSize);

  // PDL feature is 100% optional. Without `enable_pdl`, the code should still be correct.
  // Try to enable it if profiling shows that it can benefit the performance of this kernel.
  LaunchKernel(grid, kBlockSize, device).enable_pdl(kUsePDL)(
      scale_kernel<T, kVecN, kUsePDL>,
      static_cast<T*>(dst.data_ptr()),
      static_cast<const T*>(src.data_ptr()),
      factor,
      n);
}

}  // namespace
```

**Key points:**

- Include headers from `sgl_kernel/` — **not** raw CUDA headers for anything already covered
- Add a short trailing `// For ...` explanation to every `#include <sgl_kernel/...>` line
- Use `TensorMatcher` for all tensor validation; never manually check shape/dtype/device
- Use `AlignedVector` for vectorised 128-bit loads/stores — significant bandwidth win
- Use `LaunchKernel` — it resolves the stream and checks errors automatically
- Use `RuntimeCheck` for runtime assertions with useful error messages
- Prefer passing runtime scalars like `factor` directly unless compile-time specialisation is genuinely required
- `fp16_t` / `bf16_t` / `fp32_t` are the project's type aliases (from `utils.cuh`)
- `device::cast<To, From>` or `dtype_trait<T>::from(val)` for cross-type conversions
- `device::math::` functions for device math instead of bare `__` intrinsics if possible.
- Try to use `PDL` feature. In some cases, this will benefit the performance.

---

## Step 2: Add the Python wrapper in `jit_kernel/`

Create `python/sglang/jit_kernel/scale.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_scale_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT scale module for a given dtype."""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "scale",
        *args,
        cuda_files=["elementwise/scale.cuh"],
        cuda_wrappers=[("scale", f"scale<{args}>")],
    )


def scale(src: torch.Tensor, factor: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """
    Element-wise scale: dst = src * factor.

    Supported dtypes: torch.float16, torch.bfloat16, torch.float32.

    Parameters
    ----------
    src    : CUDA tensor (FP16 / BF16 / FP32)
    factor : scale factor
    out    : optional pre-allocated output tensor (same shape/dtype as src)

    Returns
    -------
    Scaled tensor (dst = src * factor).
    """
    # DO NOT add too much proactive validation here.
    # Keep the Python wrapper thin, only enforce the preconditions
    # that the current JIT/FFI path (C++ side) does not reject on its own.
    if src.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {src.dtype}. Supported: float16, bfloat16, float32"
        )
    if out is None:
        out = torch.empty_like(src)

    module = _jit_scale_module(src.dtype)
    module.scale(out, src, factor)
    return out
```

**Key points:**

- Use `cache_once` — **not** `functools.lru_cache` (incompatible with `torch.compile`)
- `load_jit` first arg(s) form the unique build marker; same marker = same cached binary
- Only include compile-time specialisation knobs in the build marker; runtime values like `factor` should stay runtime unless the kernel truly needs templating
- `cuda_wrappers`: `(export_name, kernel_symbol)` — `export_name` is called from Python
- `make_cpp_args(dtype, ...)` converts `torch.dtype` to C++ type alias:
- `is_arch_support_pdl()` checks if the current architecture supports PDL, which is typically passed as a template argument to the kernel.
- Keep Python launchers thin, but still validate the basic invariants (`is_cuda`, supported dtype, `out` metadata). In the current JIT/FFI path, invalid tensors are not always rejected safely before launch

| `torch.dtype`      | C++ type   |
|--------------------|------------|
| `torch.float16`    | `fp16_t`   |
| `torch.bfloat16`   | `bf16_t`   |
| `torch.float32`    | `fp32_t`   |

---

## Step 3 (optional): Tune JIT build flags

If your kernel uses some math functions like `expf` or `sinf`, consider enabling `--use_fast_math` for better performance (with a potential precision tradeoff):

```python
return load_jit(
    "scale",
    *args,
    cuda_files=["elementwise/scale.cuh"],
    cuda_wrappers=[("scale", f"scale<{args}>")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

If your kernel requires SM90+, raise a clear Python error before calling `load_jit`:

```python
if torch.cuda.get_device_capability()[0] < 9:
    raise RuntimeError("This kernel requires SM90 (Hopper) or later")
```

---

## Step 4: Write tests (required)

JIT kernel correctness tests and benchmarks live under `test/registered/jit/` and `test/registered/jit/benchmark/` (NOT inside the `sglang` package -- a `register_*_ci(...)` call anywhere under `python/sglang/` is rejected by the `check-no-registered-tests-in-package` pre-commit hook). Only their test-only helpers (e.g. `benchmark/marker.py`) stay alongside the kernel source under `python/sglang/jit_kernel/` and are imported by absolute path. **CI does not run `pytest` in those directories directly.** The unified runner `test/run_suite.py` discovers every `test_*.py` and `bench_*.py` under `test/registered/`, collects `register_*_ci(...)` calls by **statically parsing each file's AST**, and executes the selected suite. Every test file must register at least one CUDA entry or the collector fails its sanity check.

- **PR / per-commit CUDA suites** (see `test/run_suite.py` → `PER_COMMIT_SUITES`): JIT unit tests use `base-b-kernel-unit-test-1-gpu-large` on H100 and `base-b-kernel-unit-test-4-gpu-b200` on B200/SM100 paths (see `.github/workflows/pr-test-jit-kernel.yml`). Multi-GPU JIT tests use `base-b-kernel-unit-test-8-gpu-h200`.
- **Nightly kernel suite**: `nightly-kernel-1-gpu` with `--nightly` — typically used with `SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1` in CI for expanded parameter grids (see `python/sglang/jit_kernel/utils.py` → `should_run_full_tests` / `get_ci_test_range`). Wired in `.github/workflows/nightly-test-nvidia.yml` (e.g. `python3 run_suite.py --hw cuda --suite nightly-kernel-1-gpu --nightly --continue-on-error`).

Registration pattern (module level, **literal** `est_time`, `stage`, and `runner_config` values — required for AST parsing):

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Optional B200/SM100 registration for tests that cover Blackwell-specific code paths
# register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# Optional second registration: same file also listed under the nightly kernel suite
# (nightly suites use the legacy single-string suite=, not stage/runner_config)
# register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)
```

CI generates the suite name as `{stage}-test-{runner_config}`, so `stage="base-b-kernel-unit", runner_config="1-gpu-large"` becomes the `base-b-kernel-unit-test-1-gpu-large` suite you pass to `run_suite.py` below — don't put the `-test-` infix in `register_cuda_ci`. The single-string `suite=` form is only for nightly/stress/weekly suites.

Keep `est_time`, `stage`, `runner_config`, and `suite` as literal values. `run_suite.py` collects them from the file AST, so computed values and helper wrappers can break CI discovery.

Use `register_cuda_ci(..., disabled="reason")` if the file must stay in-tree but should be skipped in CI (e.g. multi-GPU only).

**Run like CI** (from repo root):

```bash
(cd test && python3 run_suite.py --hw cuda --suite base-b-kernel-unit-test-1-gpu-large)
# For B200/SM100-specific coverage:
(cd test && python3 run_suite.py --hw cuda --suite base-b-kernel-unit-test-4-gpu-b200)
```

For fast iteration you can still run `pytest` on a single file locally; CI coverage is via `run_suite.py`.

Create `test/registered/jit/test_scale.py`:

```python
import pytest
import torch
from sglang.jit_kernel.scale import scale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [1, 127, 128, 1024, 4097])  # cover tail remainder
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, 3.0])
def test_scale_correctness(dtype, size, factor):
    src = torch.randn(size, dtype=dtype, device="cuda")
    out = scale(src, factor)
    expected = src * factor

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scale_out_param(dtype):
    src = torch.randn(1024, dtype=dtype, device="cuda")
    out = torch.empty_like(src)
    result = scale(src, 2.0, out=out)
    assert result is out
    torch.testing.assert_close(out, src * 2.0, rtol=1e-2, atol=1e-2)


def test_scale_cpu_error():
    src = torch.randn(128, dtype=torch.float16)  # CPU tensor
    with pytest.raises(RuntimeError, match="CUDA"):
        scale(src, 2.0)


def test_scale_unsupported_dtype():
    src = torch.randint(0, 10, (128,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        scale(src, 2.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
```

---

## Step 5: Add a benchmark (required)

Benchmarks are `bench_*.py` files under `test/registered/jit/benchmark/`. They are picked up by the same `run_suite.py` machinery as unit tests. Register them for **`base-b-kernel-benchmark-test-1-gpu-large`** (PR JIT benchmark job: `python3 run_suite.py --hw cuda --suite base-b-kernel-benchmark-test-1-gpu-large`).

Benchmarks use the project's own `marker` framework (in `python/sglang/jit_kernel/benchmark/marker.py`) — **do not** use `triton.testing.perf_report` / `triton.testing.do_bench` directly. The marker framework provides (public names: `benchmark`, `parametrize`, `do_bench`, `skip`, `BenchResult`, `BenchSkip`):

- **`@marker.benchmark(line_arg, line_vals, *, unit="us")`** — the **innermost** decorator (bottom of the stack, directly above `def benchmark`). Declares the column axis: each value in `line_vals` becomes a result column, and `line_arg` is the parameter name passed into the benchmark function. `unit` is one of `"us" | "ms" | "s"`.
- **`@marker.parametrize(names, vals, ci_vals=None)`** — stackable decorator that adds a row axis (pytest-style). Each `@parametrize` adds one (or more, correlated) parameter the benchmark is swept over (Cartesian product across all `parametrize` decorators). `names` may be a single name (`"size"`) or a comma-separated correlated tuple axis (`"h,d"`, with `vals` then a list of tuples like `[(1, 64), (2, 128)]`). Pass the optional third `ci_vals` for a smaller sweep that is auto-selected under `is_in_ci()` — this is the built-in CI-shrinking mechanism, so you usually don't need `get_benchmark_range` for swept axes.
- **`marker.do_bench(fn, *, input_args=(), input_kwargs={}, ...)`** — runs `fn` under CUDA graph (default) or a naive loop, returns a `BenchResult`. Key knobs:
  - `memory_args`: defaults to `"all"` (footprint derived from all input args/kwargs). Pass an explicit tuple of tensors (e.g. `(k, v, indices)`) to count only the inputs the kernel actually touches.
  - `memory_output`: defaults to `"out"` — re-runs `fn` once to capture its **returned** tensor and counts it. For in-place kernels (which return `None`), pass the written tensors explicitly (e.g. `memory_output=(k, v)`); the re-run is then skipped. Set to `None` to count no output.
  - Together `memory_args` + `memory_output` give the GB/s column; with both defaults a function `out = f(src)` already reports `bytes(src) + bytes(out)`.
  - `graph_clone_args` / `graph_clone_kwargs`: which inputs to clone per CUDA-graph iteration to defeat L2 cache reuse. Defaults to `"all"` — pass an iterable of indices/keys to limit to the *read* args (writes don't need cloning).
  - `use_cuda_graph=False` for kernels that can't be captured.
  - `metrics=(0.5, "avg")` controls reported quantiles (the first metric becomes the table latency column).
  - `disable_log_bandwidth` (defaults from `SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH=1`) skips the bandwidth column entirely.
- **`utils.create_random(*shape)` / `utils.create_empty(*shape)`** — shorthand for `torch.randn` / `torch.empty` with `DEFAULT_DTYPE` (`bfloat16`) and `DEFAULT_DEVICE` (`"cuda"`). Override via the `dtype=` / `device=` kwargs.
- **`utils.get_benchmark_range(full_range, ci_range)`** — returns the smaller `ci_range` under CI (`is_in_ci()`), the `full_range` locally. Still available for the `benchmark(...)` column axis (which has no `ci_vals`); for `parametrize` row axes prefer the built-in `ci_vals` argument.

Create `test/registered/jit/benchmark/bench_scale.py`:

```python
import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.scale import scale as jit_scale
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")


@torch.compile()
def torch_impl_scale(src: torch.Tensor, factor: float) -> torch.Tensor:
    return src * factor


FN_MAP = {
    "jit": jit_scale,
    "torch": torch_impl_scale,
}


# `parametrize(name, full_vals, ci_vals)`: the 3rd arg is the smaller sweep
# auto-selected under CI; the full range runs locally.
@marker.parametrize("size", [2**n for n in range(10, 20)], [4096, 65536])  # 1K … 512K
@marker.benchmark("impl", ["jit", "torch"])
def benchmark(size: int, impl: str):
    src = create_random(size)
    factor = 2.0
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(src, factor),
        # `src` is read -> clone it per iter to avoid L2 reuse; factor is a scalar.
        graph_clone_args=(0,),
        # Defaults already report bandwidth: memory_args="all" counts src,
        # memory_output="out" counts the returned tensor -> bytes(src)+bytes(out).
    )


if __name__ == "__main__":
    benchmark.run()
```

**Key points:**

- The `line_arg` name passed to `benchmark` (`"impl"` here) must match a parameter on `benchmark(...)`; same for every `parametrize` name (`"size"`).
- Stack `@parametrize` once per swept axis. The required `@marker.benchmark` is the **innermost** decorator (bottom of the stack, directly above the function) — `@parametrize` rows go above it.
- Prefer `create_random` / `create_empty` from `utils.py` over open-coding `torch.randn(..., dtype=..., device=...)`.
- The GB/s column appears by default (`memory_args="all"` + `memory_output="out"`). For memory-bound kernels it's the most informative number; scope `memory_args` / `memory_output` to the tensors actually touched if the defaults over- or under-count. For compute-bound kernels where bandwidth is misleading, set `SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH=1` (or `disable_log_bandwidth=True`).
- For in-place kernels (which return `None`), pass the written tensors via `memory_output=(...)` since the `"out"` default would capture nothing.
- Tune `graph_clone_args` / `graph_clone_kwargs` to all the arguments that might be read by the kernel. We can only skip cloning for write-only args. For in-place modified args, we still need to clone them to get accurate timing (reusing the same buffer keeps it L2-hot and skews results).
- Call `benchmark.run()` (no `print_data=` kwarg — the marker framework prints directly).

Run locally:

```bash
python test/registered/jit/benchmark/bench_scale.py
```

Run the benchmark suite the way CI does:

```bash
cd test && python3 run_suite.py --hw cuda --suite base-b-kernel-benchmark-test-1-gpu-large
```

---

## Troubleshooting

- **`No CI registry found in ...` from `run_suite.py`**: add a module-level `register_cuda_ci(...)` with literal `est_time`, `stage`, and `runner_config` (and optional `nightly=True`); starred args and non-literal values break AST collection
- **JIT compilation fails**: ensure the `.cuh` file is under `python/sglang/jit_kernel/csrc/`; reduce template argument combinations
- **CUDA crash / illegal memory access**: `CUDA_LAUNCH_BLOCKING=1`; `compute-sanitizer --tool memcheck python ...`
- **Unstable benchmark results**: `marker.do_bench` uses CUDA-graph-based timing by default; set `use_cuda_graph=False` only if the kernel can't be captured. Make sure `graph_clone_args` covers every *read* tensor — reusing a single buffer keeps it L2-hot and skews results
- **Missing GB/s column**: the column is on by default; check that `SGLANG_KERNEL_DISABLE_LOG_BANDWIDTH` is not `1` and `disable_log_bandwidth` is not `True`. For in-place kernels (return `None`) the `memory_output="out"` default counts nothing — pass the written tensors via `memory_output=(...)`

---

## References

- `docs/developer_guide/development_jit_kernel_guide.md`
- `test/run_suite.py` — suite names, discovery of `test/registered/`, execution entrypoint for CI
- `python/sglang/test/ci/ci_register.py` — `register_cuda_ci` and AST registration rules
- `python/sglang/jit_kernel/utils.py` — `cache_once`, `load_jit`, `make_cpp_args`, `should_run_full_tests`, `get_ci_test_range`
- `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` — `TensorMatcher`, `SymbolicSize/DType/Device`
- `python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` — type aliases, `LaunchKernel`, `SGL_DEVICE`
- `python/sglang/jit_kernel/include/sgl_kernel/vec.cuh` — `AlignedVector`
- `python/sglang/jit_kernel/include/sgl_kernel/tile.cuh` — `tile::Memory`
- `python/sglang/jit_kernel/include/sgl_kernel/type.cuh` — `dtype_trait`, `packed_t`, `device::cast`
- `python/sglang/jit_kernel/include/sgl_kernel/math.cuh` — `device::math::`
- `python/sglang/jit_kernel/include/sgl_kernel/warp.cuh` — `warp::reduce_sum/max`
- `python/sglang/jit_kernel/include/sgl_kernel/cta.cuh` — `cta::reduce_max`
- `python/sglang/jit_kernel/include/sgl_kernel/atomic.cuh` — `atomic::max`
- `python/sglang/jit_kernel/include/sgl_kernel/runtime.cuh` — occupancy / SM count helpers
- `python/sglang/jit_kernel/csrc/add_constant.cuh` — minimal runnable reference
- `python/sglang/jit_kernel/csrc/elementwise/rmsnorm.cuh` — real example using `TensorMatcher` + `LaunchKernel` + `tile::Memory`
- `python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh` — real example using `runtime::get_blocks_per_sm` + persistent kernel pattern
- `python/sglang/jit_kernel/benchmark/marker.py` — `benchmark`, `parametrize`, `do_bench`, `BenchResult`
- `python/sglang/jit_kernel/benchmark/utils.py` — `create_random` / `create_empty` / `get_benchmark_range` helpers and `DEFAULT_DTYPE` / `DEFAULT_DEVICE`
- `test/registered/jit/benchmark/bench_qknorm.py` — real example: multi-axis `parametrize` (with `ci_vals`) + in-place `memory_output`
- `test/registered/jit/benchmark/bench_store_cache.py` — real example: scoped `memory_args` / `memory_output` + selective `graph_clone_args`

## Summary of Files Created

```
python/sglang/jit_kernel/csrc/elementwise/scale.cuh   # NEW: CUDA kernel
python/sglang/jit_kernel/scale.py                     # NEW: Python wrapper
test/registered/jit/test_scale.py                     # NEW: Tests
test/registered/jit/benchmark/bench_scale.py          # NEW: Benchmark
```
