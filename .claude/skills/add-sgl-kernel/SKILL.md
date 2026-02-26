---
name: add-sgl-kernel
description: Step-by-step tutorial for adding a heavyweight AOT CUDA/C++ kernel to sgl-kernel (including tests & benchmarks)
---

# Tutorial: Adding a New Kernel to `sgl-kernel` (AOT / Heavyweight)

This tutorial walks through adding a simple element-wise scale operation as an AOT kernel. We'll implement `scale(x, factor) = x * factor` to demonstrate the complete workflow.

## Goal

Add a new operation that scales each element of a tensor by a scalar factor:

- Input: tensor `x` (CUDA) and scalar `factor` (float)
- Output: `x * factor` (element-wise, in-place or into pre-allocated `out`)
- Supported dtypes: **FP16 (`torch.float16`), BF16 (`torch.bfloat16`), FP32 (`torch.float32`)**
  - Dispatched via `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` macro (defined in `sgl-kernel/include/utils.h`)

## Two rules of thumb (must follow)

1. **Heavyweight kernels go to `sgl-kernel`.** If it depends on CUTLASS / FlashInfer / DeepGEMM (or similarly heavy stacks), implement it in `sgl-kernel/`.
2. **Lightweight kernels go to `python/sglang/jit_kernel`.** If it is small, has few dependencies, and benefits from rapid iteration, implement it as a JIT kernel instead.

In addition, every new kernel must ship with:

- **Tests** (pytest)
- **A benchmark script** (triton.testing)

---

## Repository integration map

You will typically touch these files/areas:

- Implementation: `sgl-kernel/csrc/elementwise/scale.cu` (pick the right subdirectory)
- Public declarations: `sgl-kernel/include/sgl_kernel_ops.h`
- Torch extension registration: `sgl-kernel/csrc/common_extension.cc`
- Build: `sgl-kernel/CMakeLists.txt` (`set(SOURCES ...)`)
- Python API: `sgl-kernel/python/sgl_kernel/` and `sgl-kernel/python/sgl_kernel/__init__.py`
- Tests: `sgl-kernel/tests/test_scale.py`
- Benchmarks: `sgl-kernel/benchmark/bench_scale.py`

---

## Step 1: Implement the kernel in `csrc/`

Pick the right subdirectory:

- `csrc/elementwise/` — for element-wise ops (our example)
- `csrc/gemm/`, `csrc/attention/`, `csrc/moe/` — for other categories

Create `sgl-kernel/csrc/elementwise/scale.cu`:

```cpp
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"  // DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16

// scale_kernel: out[i] = input[i] * factor
// Supports float, half (__half), __nv_bfloat16 via template T
template <typename T>
__global__ void scale_kernel(T* __restrict__ out,
                              const T* __restrict__ input,
                              float factor,
                              int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(static_cast<float>(input[idx]) * factor);
  }
}

void scale(at::Tensor& out, const at::Tensor& input, double factor) {
  TORCH_CHECK(input.is_cuda(),       "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(out.is_cuda(),         "out must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(),   "out must be contiguous");
  TORCH_CHECK(out.sizes() == input.sizes(),  "out and input must have the same shape");
  TORCH_CHECK(out.scalar_type() == input.scalar_type(),
              "out and input must have the same dtype");

  const int64_t n = input.numel();
  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  // Dispatches over float, float16, bfloat16
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    scale_kernel<c_type><<<blocks, threads, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()),
        static_cast<const c_type*>(input.data_ptr()),
        static_cast<float>(factor),
        n);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess,
                "scale_kernel launch failed: ", cudaGetErrorString(status));
    return true;
  });
}
```

**Key points:**

- Use `at::Tensor` (PyTorch tensors), `TORCH_CHECK` for validation, `at::cuda::getCurrentCUDAStream()` for stream
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` covers `float`, `half` (FP16), `__nv_bfloat16` (BF16)
- Add device error checking after every kernel launch
- If a kernel only works on certain architectures, enforce that with `TORCH_CHECK` and skip logic in tests

---

## Step 2: Add a C++ declaration in `include/sgl_kernel_ops.h`

Edit `sgl-kernel/include/sgl_kernel_ops.h`, add to the elementwise section:

```cpp
void scale(at::Tensor& out, const at::Tensor& input, double factor);
```

---

## Step 3: Register the op in `csrc/common_extension.cc`

Edit `sgl-kernel/csrc/common_extension.cc`, inside `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)`:

```cpp
// From csrc/elementwise
m.def("scale(Tensor! out, Tensor input, float factor) -> ()");
m.impl("scale", torch::kCUDA, &scale);
```

**Key points:**

- `Tensor!` means in-place / mutable output argument
- The schema is important for `torch.compile` and for consistent call signatures
- If your underlying C++ API uses `float` but PyTorch bindings expect `double`, the implicit cast is fine for scalars; use shims if needed for other types

---

## Step 4: Add the new source file to `CMakeLists.txt`

Edit `sgl-kernel/CMakeLists.txt`, add to `set(SOURCES ...)`:

```cmake
csrc/elementwise/scale.cu
```

**Key points:**

- Keep the list **alphabetically sorted** (the file explicitly requires this)
- If the kernel has arch constraints, reflect that in tests/benchmarks via skip logic

---

## Step 5: Expose a Python API under `sgl-kernel/python/sgl_kernel/`

In `sgl-kernel/python/sgl_kernel/__init__.py`, add:

```python
from torch.ops import sgl_kernel as _ops

def scale(out: torch.Tensor, input: torch.Tensor, factor: float) -> None:
    """
    Element-wise scale: out = input * factor (in-place into out).

    Supported dtypes: torch.float16, torch.bfloat16, torch.float32.

    Parameters
    ----------
    out    : pre-allocated CUDA output tensor (same shape/dtype as input)
    input  : CUDA input tensor
    factor : scale factor (float)
    """
    _ops.scale(out, input, factor)
```

Or export it from the existing module organisation — follow the pattern already used by similar ops in `__init__.py`.

---

## Step 6: Write tests (required)

Create `sgl-kernel/tests/test_scale.py`:

```python
import pytest
import torch
import sgl_kernel


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [128, 1024, 4096, 65536])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_scale_correctness(dtype, size, factor):
    input = torch.randn(size, dtype=dtype, device="cuda")
    out   = torch.empty_like(input)

    sgl_kernel.scale(out, input, factor)

    expected = input * factor
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_scale_shape_mismatch():
    input = torch.randn(128, dtype=torch.float16, device="cuda")
    out   = torch.empty(256, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        sgl_kernel.scale(out, input, 2.0)


def test_scale_cpu_input():
    input = torch.randn(128, dtype=torch.float16)  # CPU
    out   = torch.empty_like(input)
    with pytest.raises(RuntimeError, match="CUDA"):
        sgl_kernel.scale(out, input, 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
```

Run:

```bash
pytest sgl-kernel/tests/test_scale.py -q
```

---

## Step 7: Add a benchmark (required)

Create `sgl-kernel/benchmark/bench_scale.py`:

```python
import itertools
import os

import torch
import triton
import triton.testing

import sgl_kernel

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

dtypes  = [torch.float16] if IS_CI else [torch.float16, torch.bfloat16, torch.float32]
sizes   = [4096] if IS_CI else [2**n for n in range(10, 20)]  # 1K … 512K
factors = [2.0]

configs = list(itertools.product(dtypes, sizes))


def torch_scale(input: torch.Tensor, factor: float) -> torch.Tensor:
    return input * factor


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dtype", "size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sglang", "torch"],
        line_names=["SGL Kernel", "PyTorch"],
        styles=[("green", "-"), ("red", "--")],
        ylabel="µs (median)",
        plot_name="scale-performance",
        args={},
    )
)
def benchmark(dtype, size, provider):
    input  = torch.randn(size, dtype=dtype, device="cuda")
    out    = torch.empty_like(input)
    factor = 2.0

    if provider == "sglang":
        fn = lambda: sgl_kernel.scale(out, input, factor)
    else:
        fn = lambda: torch_scale(input, factor)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
```

Run:

```bash
python sgl-kernel/benchmark/bench_scale.py
```

---

## Step 8: Build and validate

Build:

```bash
cd sgl-kernel
make build -j16
```

If you need to limit host resource usage:

```bash
cd sgl-kernel
make build -j1 MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

Validate:

```bash
pytest sgl-kernel/tests/test_scale.py -q
python sgl-kernel/benchmark/bench_scale.py
```

---

## Troubleshooting

- **Async CUDA errors**: `CUDA_LAUNCH_BLOCKING=1`
- **Memory errors**: `compute-sanitizer --tool memcheck python ...`
- **Build is too slow / OOM**: reduce `MAX_JOBS` and `SGL_KERNEL_COMPILE_THREADS`
- **Binary bloat**: use `sgl-kernel/analyze_whl_kernel_sizes.py`
- **CMake sources list**: if your `.cu` file is missing from `SOURCES`, the symbol will be undefined at link time

---

## References

- `sgl-kernel/README.md`
- `sgl-kernel/include/sgl_kernel_ops.h`
- `sgl-kernel/csrc/common_extension.cc`
- `sgl-kernel/CMakeLists.txt`
- `sgl-kernel/include/utils.h` — `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16` macro and friends
- `sgl-kernel/csrc/elementwise/activation.cu` — reference for the FP16/BF16/FP32 dispatch pattern

## Summary of Files Created/Modified

```
sgl-kernel/csrc/elementwise/scale.cu          # NEW: CUDA kernel + launcher
sgl-kernel/include/sgl_kernel_ops.h           # MODIFIED: C++ declaration
sgl-kernel/csrc/common_extension.cc           # MODIFIED: schema + dispatch registration
sgl-kernel/CMakeLists.txt                     # MODIFIED: add source file (alphabetical)
sgl-kernel/python/sgl_kernel/__init__.py      # MODIFIED: export Python API
sgl-kernel/tests/test_scale.py                # NEW: tests
sgl-kernel/benchmark/bench_scale.py           # NEW: benchmark
```
