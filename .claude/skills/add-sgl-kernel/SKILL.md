---
name: add-sgl-kernel
description: Step-by-step tutorial for adding a heavyweight AOT CUDA/C++ kernel to sgl-kernel (including tests & benchmarks)
---

# Tutorial: Adding a New Kernel to `sgl-kernel` (AOT / Heavyweight)

This SKILL is a step-by-step guide for adding a **heavyweight** CUDA/C++ kernel to `sgl-kernel/`.

Typical characteristics:

- Depends on heavyweight components such as CUTLASS / FlashInfer / DeepGEMM / sgl-attn
- Needs AOT build and distribution (wheel / torch extension), so build time, link flags, CUDA arch targets, and binary size matter
- Exposed as a stable `sgl_kernel` API and used by higher-level code (including `torch.compile`)

## Two rules of thumb (must follow)

1. **Heavyweight kernels go to `sgl-kernel`.** If it depends on CUTLASS/FlashInfer/DeepGEMM (or similarly heavy stacks), implement it in `sgl-kernel/`.
2. **Lightweight kernels go to `python/sglang/jit_kernel`.** If it is small, has few dependencies, and benefits from rapid iteration, implement it as a JIT kernel instead.

In addition, every new kernel must ship with:

- **Tests** (pytest)
- **A benchmark script** (triton.testing)

---

## Goal

Add a new kernel end-to-end, including:

- CUDA/C++ implementation
- Torch library registration (`m.def` schema + `m.impl` dispatch)
- Build system integration (CMake sources list)
- Python-facing API
- Correctness tests and performance benchmarks

---

## Repository integration map

You will typically touch these files/areas:

- Implementation: `sgl-kernel/csrc/...`
- Public declarations: `sgl-kernel/include/sgl_kernel_ops.h`
- Torch extension registration: `sgl-kernel/csrc/common_extension.cc`
- Build: `sgl-kernel/CMakeLists.txt` (`set(SOURCES ...)`)
- Python API: `sgl-kernel/python/sgl_kernel/...` and `sgl-kernel/python/sgl_kernel/__init__.py`
- Tests: `sgl-kernel/tests/test_<op>.py`
- Benchmarks: `sgl-kernel/benchmark/bench_<op>.py`

---

## Step 1: Implement the kernel in `csrc/`

1. Pick the right subdirectory:

- `csrc/elementwise/`
- `csrc/gemm/`
- `csrc/attention/`
- `csrc/moe/`

2. Implementation requirements:

- Clearly define dtype/shape/stride/contiguity assumptions
- If assumptions are violated, fail fast with a readable error (e.g. `TORCH_CHECK(...)`)
- After kernel launch, perform device error checking (follow existing project conventions)

**Key points:**

- Prefer explicit validation over "it probably works".
- If a kernel only works on certain architectures, make that restriction explicit (error/skip behavior).

---

## Step 2: Add a C++ declaration in `include/sgl_kernel_ops.h`

Edit:

- `sgl-kernel/include/sgl_kernel_ops.h`

Add your function declaration in the appropriate section.

---

## Step 3: Register the op in `csrc/common_extension.cc` (schema + dispatch)

Edit:

- `sgl-kernel/csrc/common_extension.cc`

Inside `TORCH_LIBRARY_FRAGMENT(sgl_kernel, m)`:

1. Add `m.def(...)` with a **schema**.
2. Add `m.impl(...)` for CUDA dispatch.

**Key points:**

- The schema is important for `torch.compile` and for consistent call signatures.
- If your underlying C++ API uses native types (e.g. `int`, `float`), but PyTorch bindings expect `int64_t` / `double`, use the project’s recommended shim approach (see `sgl-kernel/README.md`).

---

## Step 4: Add the new source file to `CMakeLists.txt`

Edit:

- `sgl-kernel/CMakeLists.txt`

Add your new `.cu` / `.cc` file to the `set(SOURCES ...)` list.

**Key points:**

- Keep the list **alphabetically sorted** (the file explicitly requires this).
- If your kernel has arch constraints, reflect that in tests/benchmarks via skip logic.

---

## Step 5: Expose a Python API under `sgl-kernel/python/sgl_kernel/`

Goal: users can call `sgl_kernel.<op>(...)`.

- Add/extend a Python wrapper under `sgl-kernel/python/sgl_kernel/` (follow existing module organization).
- Export it from `sgl-kernel/python/sgl_kernel/__init__.py`.

---

## Step 6: Write tests (required)

Create:

- `sgl-kernel/tests/test_<op>.py`

**Minimum coverage:**

- **Shapes**: typical + edge cases
- **Dtypes**: whatever the kernel claims to support
- **Correctness**: compare with a reference implementation (PyTorch / FlashInfer / another stable backend)
- **Negative cases**: unsupported dtype/shape/arch should either raise a clear error or be explicitly skipped

**Skipping by architecture:**

- Use `@pytest.mark.skipif(..., reason="...")` when compute capability requirements apply.

Run:

```bash
pytest sgl-kernel/tests/test_<op>.py -q
```

---

## Step 7: Add a benchmark (required)

Create:

- `sgl-kernel/benchmark/bench_<op>.py`

Follow the repository convention:

- Use `triton.testing.Benchmark` + `triton.testing.perf_report`
- Prefer `triton.testing.do_bench_cudagraph` for timing

**Minimum benchmark requirements:**

- At least two providers/variants:
  - Your `sgl_kernel` implementation
  - A baseline (PyTorch / `torch.compile` / Triton / FlashInfer)
- Quantiles output (median/min/max)
- CI-friendly ranges controlled by `CI` / `GITHUB_ACTIONS`

Run:

```bash
python sgl-kernel/benchmark/bench_<op>.py
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

- Tests: `pytest sgl-kernel/tests/test_<op>.py -q`
- Benchmark: `python sgl-kernel/benchmark/bench_<op>.py`

---

## Troubleshooting

- **Async CUDA errors**: `CUDA_LAUNCH_BLOCKING=1`
- **Memory errors**: `compute-sanitizer --tool memcheck python ...`
- **Build is too slow / OOM**: reduce `MAX_JOBS` and `SGL_KERNEL_COMPILE_THREADS`
- **Binary bloat**: use `sgl-kernel/analyze_whl_kernel_sizes.py`

---

## References

- `sgl-kernel/README.md`
- `sgl-kernel/include/sgl_kernel_ops.h`
- `sgl-kernel/csrc/common_extension.cc`
- `sgl-kernel/CMakeLists.txt`
