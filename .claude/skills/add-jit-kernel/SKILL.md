---
name: add-jit-kernel
description: Step-by-step tutorial for adding a lightweight JIT CUDA/C++ kernel to python/sglang/jit_kernel (including tests & benchmarks)
---

# Tutorial: Adding a New Kernel to `python/sglang/jit_kernel` (JIT / Lightweight)

This SKILL is a step-by-step guide for adding a **lightweight** CUDA/C++ kernel to `python/sglang/jit_kernel/`.

Typical characteristics:

- Few dependencies (usually tvm-ffi + a small subset of `sgl_kernel` utility headers)
- Compiled at runtime (JIT), optimized for fast iteration
- Avoids pulling heavyweight third-party/template code into AOT builds

## Two rules of thumb (must follow)

1. **Heavyweight kernels go to `sgl-kernel`.** If it depends on CUTLASS / FlashInfer / DeepGEMM (or similarly heavy stacks), implement it in `sgl-kernel/`.
2. **Lightweight kernels go to `jit_kernel`.** If it is small and can be compiled independently, implement it here.

## Stop and use `sgl-kernel` instead (important)

Do **not** add a new kernel under `jit_kernel` if any of the following applies:

- It directly depends on CUTLASS / FlashInfer (or other heavyweight third-party stacks)
- It requires complex link-time integration, large template instantiations, or AOT-style packaging

In addition, every new JIT kernel must ship with:

- **Tests** (pytest)
- **A benchmark script** (triton.testing)

---

## Goal

Add a new JIT kernel end-to-end, including:

- CUDA/C++ implementation in `jit_kernel/csrc`
- A Python wrapper that compiles + loads the JIT module via tvm-ffi
- Correctness tests
- A reproducible benchmark (with CI-friendly ranges)

---

## Repository integration map

You will typically touch these files/areas:

- Implementation: `python/sglang/jit_kernel/csrc/`
- Reusable headers: `python/sglang/jit_kernel/include/`
- Python API: `python/sglang/jit_kernel/<op>.py`
- JIT build + cache utilities: `python/sglang/jit_kernel/utils.py`
- Tests: `python/sglang/jit_kernel/tests/test_<op>.py`
- Benchmarks: `python/sglang/jit_kernel/benchmark/bench_<op>.py`
- Benchmark helpers: `python/sglang/jit_kernel/benchmark/utils.py`

---

## tvm-ffi primer (practical, as used in this repo)

This repository uses tvm-ffi primarily as a **stable C++ ABI** and a set of **lightweight container types** to move data between Python and C++ with minimal overhead.

### Core types you will see in JIT kernels

- `tvm::ffi::TensorView`
  - A **non-owning view** of a tensor (backed by DLPack) that enables zero-copy interop.
  - Use it for most tensor arguments in kernel entrypoints.
  - You typically inspect/validate:
    - Shape/strides: `dim()`, `shape()`, `strides()`, `size(i)`, `stride(i)`, `is_contiguous()`
    - Dtype/device: `dtype()`, `device()`
    - Raw pointer: `data_ptr()` (then cast after dtype checks)

- `tvm::ffi::Optional<T>`
  - Used for optional tensor arguments, e.g. `tvm::ffi::Optional<tvm::ffi::TensorView>`.
  - Always check `has_value()` before using it.

### Containers you may want (even if not widely used here yet)

- `tvm::ffi::Array<T>`, `tvm::ffi::Tuple<...>`
  - Useful for passing small structured metadata without inventing ad-hoc pointer conventions.

### STL support

tvm-ffi has optional headers to interop with parts of the C++ standard library (review mentions `extra/stl.h`). This repo currently mostly relies on `TensorView` + `Optional` for kernel interfaces.

### Source of truth in `sglang`

The most reliable documentation for how tvm-ffi is used in `sglang` is the code under:

- `python/sglang/jit_kernel/include/`

In particular:

- `python/sglang/jit_kernel/include/sgl_kernel/tensor.h`
  - `host::TensorMatcher` for validating shapes/strides/dtypes/devices
  - Symbolic helper types used across many kernels:
    - `host::SymbolicSize` / `host::SymbolicDType` / `host::SymbolicDevice`
    - Typical pattern: declare symbols, validate with `TensorMatcher(...).verify(...)`, then `unwrap()` the resolved values for launch configuration

---

## Step 0 (optional): Generate a `.clangd` config for better IDE support

Because JIT kernels compile at runtime, there is no static `compile_commands.json`.

Run from your working directory (typically the repository root):

```bash
python -m sglang.jit_kernel
```

This will generate a `.clangd` file (and will not overwrite an existing one).

---

## Step 1: Implement the CUDA/C++ kernel in `jit_kernel/csrc/`

1. Create a new source file:

- `python/sglang/jit_kernel/csrc/<op>.cuh` (common pattern)

2. Use the project’s recommended utilities.

Notes:

- Prefer reading and reusing the actual helper code in `python/sglang/jit_kernel/include/`.
- If you find a missing helper that would be reusable across kernels, add it under `python/sglang/jit_kernel/include/`.

- Use `tvm::ffi::TensorView` for tensor arguments (PyTorch tensors are passed through tvm-ffi)
- Validate inputs with `TensorMatcher` (shape/stride/dtype/device)
- Use `RuntimeCheck` / `RuntimeDeviceCheck` for readable runtime validation
- Launch kernels via `LaunchKernel` (stream/device resolution)

**Key points:**

- Be explicit about contiguity/stride assumptions.
- Make failures readable. A crash is not an error message.

---

## Step 2: Add the Python wrapper (compile + load with `load_jit`)

Create:

- `python/sglang/jit_kernel/<op>.py`

### 2.1 Use `cache_once` for module caching

Use `sglang.jit_kernel.utils.cache_once` (do **not** use `functools.lru_cache`).

Reason: `functools.lru_cache` is not compatible with `torch.compile` in this codebase.

### 2.2 Build and load the module with `load_jit`

`load_jit` compiles a tvm-ffi module from C++/CUDA sources and returns a module object.

Key fields:

- `*args: str`: a unique marker for the build (different kernels / different template args must produce different markers)
- `cpp_files` / `cuda_files`: filenames under `jit_kernel/csrc/`
- `cpp_wrappers` / `cuda_wrappers`: list of `(export_name, kernel_symbol)`
  - `export_name` is how you call it from Python: `module.export_name(...)`
  - `kernel_symbol` is the C++ symbol name (can include template args)

### 2.3 Template arguments (if needed)

Use `make_cpp_args(...)` to convert Python values (int/float/bool/torch.dtype) into C++ template arguments.

### 2.4 Destination-passing style (recommended)

Prefer APIs that accept preallocated outputs (e.g. `out=` / `output=`) to avoid allocations in hot paths.

---

## Step 3 (optional): Tune JIT build flags

`load_jit` supports:

- `extra_cflags`, `extra_cuda_cflags`, `extra_ldflags`
- `extra_include_paths`
- `build_directory`

**CUDA arch list:**

`load_jit` sets `TVM_FFI_CUDA_ARCH_LIST` automatically if it is not already present.

If your kernel has hard arch requirements (e.g. SM90+ only), enforce that:

- In Python wrapper (raise a clear error)
- In tests/benchmarks (skip or return NaN for unsupported providers)

---

## Step 4: Write tests (required)

Create:

- `python/sglang/jit_kernel/tests/test_<op>.py`

**Recommended test patterns:**

- Compare against a reference implementation (PyTorch or math definition)
- If a corresponding op exists in `sgl-kernel` (AOT) or FlashInfer, add a cross-implementation equivalence test

**Minimum coverage:**

- Shapes: typical + edge cases
- Dtypes: the dtypes you claim to support
- Correctness: `torch.testing.assert_close` with appropriate tolerances
- Failure modes: invalid dtype/shape/device should fail clearly (or be skipped)

Run:

```bash
pytest python/sglang/jit_kernel/tests/test_<op>.py -q
```

---

## Step 5: Add a benchmark (required)

Create:

- `python/sglang/jit_kernel/benchmark/bench_<op>.py`

Use the shared helpers:

- `python/sglang/jit_kernel/benchmark/utils.py`
  - `is_in_ci()`
  - `get_benchmark_range(...)`
  - `run_benchmark(fn)` (uses `triton.testing.do_bench_cudagraph` and returns microseconds)

**Minimum benchmark requirements:**

- At least two providers/variants:
  - Your JIT kernel
  - A baseline (FlashInfer / `sgl-kernel` AOT / PyTorch / `torch.compile`)
- CI-friendly reduced ranges (guard with `is_in_ci()` or env vars)
- Use `triton.testing.Benchmark` + `triton.testing.perf_report`

Run:

```bash
python python/sglang/jit_kernel/benchmark/bench_<op>.py
```

---

## Troubleshooting

- **JIT compilation fails**:
  - Ensure the file is under `python/sglang/jit_kernel/csrc/`
  - Reduce template argument combinations to minimize compilation scope

- **CUDA crash / illegal memory access**:
  - `CUDA_LAUNCH_BLOCKING=1`
  - `compute-sanitizer --tool memcheck python ...`

- **Unstable benchmark results**:
  - Use CUDA-graph-based benchmarking (`run_benchmark` does this by default)
  - Fix input distributions and shapes

---

## References

- `docs/developer_guide/development_jit_kernel_guide.md`
- `python/sglang/jit_kernel/utils.py` (`cache_once`, `load_jit`, wrappers, CUDA arch list)
- `python/sglang/jit_kernel/include/sgl_kernel/tensor.h` (`TensorMatcher` and symbolic size/dtype/device helpers)
- Existing kernels that are good references for utility usage:
  - `python/sglang/jit_kernel/per_tensor_quant_fp8.py` + `python/sglang/jit_kernel/csrc/gemm/per_tensor_quant_fp8.cuh`
  - `python/sglang/jit_kernel/norm.py` + `python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh`
  - `python/sglang/jit_kernel/csrc/elementwise/qknorm_across_heads.cuh`
- `python/sglang/jit_kernel/tests/test_add_constant.py` (minimal runnable example)
- `python/sglang/jit_kernel/benchmark/utils.py` (benchmark helpers)
