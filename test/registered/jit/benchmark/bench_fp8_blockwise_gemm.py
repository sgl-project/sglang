from __future__ import annotations

import sys

import torch
import triton

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-small",
)

_SM120_SUPPORTED = is_sm120_supported()


def _make_inputs(m: int, n: int, k: int, device: str = "cuda"):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(m, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(n, k, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()

    scale_a = torch.randn((m, k // 128), device=device, dtype=torch.float32) * 0.001
    scale_b = (
        torch.randn((k // 128, n // 128), device=device, dtype=torch.float32) * 0.001
    )
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    return a_fp8, b_fp8, scale_a, scale_b


def _torch_ref(a_fp8, b_fp8, scale_a, scale_b):
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    sa = group_broadcast(scale_a, a_fp8.shape)
    sb = group_broadcast(scale_b, b_fp8.shape)
    return torch.mm(sa * a_fp8.to(torch.float32), sb * b_fp8.to(torch.float32)).to(
        torch.bfloat16
    )


shape_range = get_benchmark_range(
    full_range=[
        (16, 4096, 4096),  # swapAB tile N=32
        (64, 4096, 4096),  # swapAB tile N=64
        (128, 4096, 4096),  # non-swap 128
        (512, 4096, 4096),
        (1024, 8192, 4096),
    ],
    ci_range=[(16, 4096, 4096), (128, 4096, 4096)],
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n", "k"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=["jit", "torch_ref"],
        line_names=["JIT FP8 Blockwise GEMM", "Torch Ref"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="us",
        plot_name="fp8-blockwise-scaled-mm-performance",
        args={},
    )
)
def benchmark(m, n, k, provider):
    a_fp8, b_fp8, scale_a, scale_b = _make_inputs(m, n, k)

    if provider == "jit":
        fn = lambda: fp8_blockwise_scaled_mm(
            a_fp8, b_fp8, scale_a, scale_b, out_dtype=torch.bfloat16
        )
    elif provider == "torch_ref":
        fn = lambda: _torch_ref(a_fp8, b_fp8, scale_a, scale_b)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    if not _SM120_SUPPORTED:
        print(
            "[skip] fp8_blockwise_scaled_mm benchmark requires SM120 with CUDA 12.8+."
        )
        sys.exit(0)
    benchmark.run(print_data=True)
