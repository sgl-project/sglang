---
paths:
  - "**/bench_*.py"
  - "**/benchmark/**/*.py"
  - "benchmark/kernels/**"
---

# Kernel benchmarks

How a benchmark is written is up to you. The requirement is only that the numbers are valid.

## Tools

In SGLang we recommend `marker.do_bench` (`python/sglang/kernels/jit/benchmark/marker.py`). It works for any callable, not only JIT kernels, and it measures with a cold L2 by default. It is not mandatory. CUPTI (e.g. CUDA 13 `cupti-python`), the torch profiler / `bench_kineto`, Triton's `do_bench` / proton, `flashinfer.testing`, Nsight Systems / Nsight Compute, or a hand-written loop are all fine. Each pitfall below must be handled, whichever tool you use.

## L2 cache reuse

Timing the same kernel on the same inputs in a loop measures an L2-hot kernel, not the one serving runs. `triton.testing.do_bench_cudagraph` with fixed arguments and bare `for` loops between two events both do this. Measure cold L2 by flushing between calls or by rotating enough input copies to exceed L2 several times over. Hot L2 is acceptable only when the working set far exceeds L2 or when hot L2 is the real serving pattern, and the reason must be stated.

Host-memory (zero-copy) loads do not go through device L2. Peer-GPU (NVLink) reads still see reuse (on remote GPU).
