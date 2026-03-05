"""Benchmark JIT vs AOT nvfp4 scaled GEMM kernel.

Usage:
    python bench_nvfp4_scaled_mm.py              # CI mode (small range)
    SGLANG_JIT_KERNEL_BENCHMARK_FULL=1 python bench_nvfp4_scaled_mm.py
"""

import itertools
from typing import Tuple

import torch
import triton
import triton.testing
from sgl_kernel import cutlass_scaled_fp4_mm as aot_cutlass_scaled_fp4_mm
from sgl_kernel import scaled_fp4_quant

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.nvfp4_scaled_mm import (
    cutlass_scaled_fp4_mm as jit_cutlass_scaled_fp4_mm,
)

M_RANGE = get_benchmark_range(
    full_range=[1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ci_range=[128],
)
N_RANGE = get_benchmark_range(
    full_range=[4096, 8192, 14336],
    ci_range=[4096],
)
K_RANGE = get_benchmark_range(
    full_range=[4096, 8192, 14336],
    ci_range=[4096],
)

LINE_VALS = ["aot", "jit"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel"]
STYLES = [("orange", "-"), ("blue", "--")]
X_NAMES = ["m", "n", "k"]
CONFIGS = list(itertools.product(M_RANGE, N_RANGE, K_RANGE))


def _make_inputs(m: int, n: int, k: int, out_dtype: torch.dtype):
    """Create FP4 inputs for benchmarking."""
    a_fp16 = torch.randn(m, k, dtype=torch.float16, device=DEFAULT_DEVICE)
    b_fp16 = torch.randn(n, k, dtype=torch.float16, device=DEFAULT_DEVICE)
    a_global_scale = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=DEFAULT_DEVICE)
    b_global_scale = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=DEFAULT_DEVICE)
    a_fp4, a_sf = scaled_fp4_quant(a_fp16, a_global_scale)
    b_fp4, b_sf = scaled_fp4_quant(b_fp16, b_global_scale)
    alpha = torch.tensor(
        [1.0 / (a_global_scale.item() * b_global_scale.item())],
        dtype=torch.float32,
        device=DEFAULT_DEVICE,
    )
    return a_fp4, b_fp4, a_sf, b_sf, alpha


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=X_NAMES,
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="nvfp4-scaled-mm-performance",
        args={},
    )
)
def benchmark(m: int, n: int, k: int, provider: str) -> Tuple[float, float, float]:
    out_dtype = torch.bfloat16
    a_fp4, b_fp4, a_sf, b_sf, alpha = _make_inputs(m, n, k, out_dtype)
    torch.cuda.synchronize()

    if provider == "aot":

        def fn():
            aot_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype)

    elif provider == "jit":

        def fn():
            jit_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, out_dtype)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    import torch

    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        print("Skipping: nvfp4 GEMM requires Blackwell GPU (SM100+)")
    else:
        # Warm up JIT compilation
        print("Warming up JIT compilation...")
        m, n, k = 128, 4096, 4096
        a_fp4, b_fp4, a_sf, b_sf, alpha = _make_inputs(m, n, k, torch.bfloat16)
        jit_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)
        torch.cuda.synchronize()
        print("JIT compilation done. Running benchmark...")
        benchmark.run(print_data=True)
