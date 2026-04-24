import itertools

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.per_tensor_absmax_fp8 import per_tensor_absmax_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-benchmark-1-gpu-large")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

BATCH_SIZE_LIST = get_benchmark_range(
    full_range=[1, 16, 64, 256, 1024, 4096],
    ci_range=[16, 256],
)
HIDDEN_DIM_LIST = get_benchmark_range(
    full_range=[2048, 4096, 7168, 14336],
    ci_range=[4096],
)

configs = list(itertools.product(BATCH_SIZE_LIST, HIDDEN_DIM_LIST))


def torch_absmax_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(x)).float().div_(FP8_MAX).view(1)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "hidden_dim"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit", "torch"],
        line_names=["SGL JIT Kernel", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="per-tensor-absmax-fp8-performance",
        args={},
    )
)
def benchmark(batch_size: int, hidden_dim: int, provider: str):
    device = torch.device(DEFAULT_DEVICE)
    x = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16, device=device)

    if provider == "jit":
        scale = torch.zeros(1, dtype=torch.float32, device=device)

        def fn():
            scale.zero_()
            per_tensor_absmax_fp8(x, scale)

    else:

        def fn():
            torch_absmax_scale(x)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
