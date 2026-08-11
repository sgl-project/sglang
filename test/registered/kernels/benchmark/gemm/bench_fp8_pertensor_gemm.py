from __future__ import annotations

import functools
import statistics
import sys

import torch
import triton
from flashinfer import bmm_fp8
from flashinfer.testing.utils import bench_gpu_time_with_cupti

from sglang.kernels.jit.benchmark.utils import get_benchmark_range
from sglang.kernels.ops.gemm.fp8_pertensor_gemm import fp8_pertensor_scaled_mm
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


@functools.lru_cache(maxsize=None)
def _make_inputs(m: int, n: int, k: int, device: str = "cuda"):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(m, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(n, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_nk = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    a_scale = torch.tensor([0.01], device=device, dtype=torch.float32)
    b_scale = torch.tensor([0.02], device=device, dtype=torch.float32)
    return a_fp8, b_nk, a_scale, b_scale


shape_range = get_benchmark_range(
    full_range=[
        (m, n, k)
        for n, k in ((16384, 5120), (5120, 6144), (14336, 5120))
        for m in (4, 8, 16, 24, 32, 48, 60)
    ],
    ci_range=[(16, 16384, 5120), (24, 5120, 6144)],
)


def _bench_us(fn):
    times_ms = bench_gpu_time_with_cupti(
        fn=fn,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    times_us = sorted(1000.0 * t for t in times_ms)
    return statistics.median(times_us), times_us[0], times_us[-1]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n", "k"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=["jit", "flashinfer_cublas"],
        line_names=["JIT SM120 Per-Tensor FP8 GEMM", "FlashInfer cuBLAS bmm_fp8"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="fp8-pertensor-gemm-performance",
        args={},
    )
)
def benchmark(m, n, k, provider):
    a_fp8, b_nk, a_scale, b_scale = _make_inputs(m, n, k)

    if provider == "jit":
        fn = lambda: fp8_pertensor_scaled_mm(a_fp8, b_nk, a_scale, b_scale)
    elif provider == "flashinfer_cublas":
        b_kn = b_nk.t()
        fn = lambda: bmm_fp8(
            a_fp8.unsqueeze(0),
            b_kn.unsqueeze(0),
            a_scale,
            b_scale,
            torch.bfloat16,
            backend="cublas",
        ).view(m, n)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _bench_us(fn)


if __name__ == "__main__":
    if not is_sm120_supported():
        print(
            "[skip] fp8_pertensor_scaled_mm benchmark requires SM120 with CUDA 12.8+."
        )
        sys.exit(0)
    benchmark.run(print_data=True)
