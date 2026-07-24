import itertools
import math

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.new_gelu import new_gelu as jit_new_gelu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-benchmark-1-gpu-large")

SIZE_LIST = get_benchmark_range(
    full_range=[2**n for n in range(10, 18)],
    ci_range=[4096, 65536],
)

DTYPE_LIST = get_benchmark_range(
    full_range=[torch.float16, torch.bfloat16, torch.float32],
    ci_range=[torch.bfloat16],
)

configs = list(itertools.product(SIZE_LIST, DTYPE_LIST))


def _torch_new_gelu(x: torch.Tensor) -> torch.Tensor:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size", "dtype"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["jit", "torch"],
        line_names=["SGL JIT Kernel", "PyTorch"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="us",
        plot_name="new-gelu-performance",
        args={},
    )
)
def benchmark(size: int, dtype: torch.dtype, provider: str):
    x = torch.randn(size, dtype=dtype, device=DEFAULT_DEVICE)

    if provider == "jit":
        fn = lambda: jit_new_gelu(x)
    else:
        fn = lambda: _torch_new_gelu(x)

    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
