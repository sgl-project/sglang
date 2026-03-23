import itertools

import torch
import triton
import triton.testing
from flashinfer.norm import fused_add_rmsnorm as fi_fused_add_rmsnorm
from flashinfer.norm import rmsnorm as fi_rmsnorm

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.utils import is_in_ci

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"

# JIT rmsnorm: hidden_size in {64,128,256} or (multiple of 256, <=8192)
# JIT fused_add_rmsnorm: hidden_size % 8 == 0, <=8192
# Use multiples of 256 <=8192 to satisfy both kernels
if IS_CI:
    BS_LIST = [16]
    HIDDEN_SIZE_LIST = [512, 2048]
else:
    BS_LIST = [2**n for n in range(0, 14)]
    HIDDEN_SIZE_LIST = [1536, 3072, 4096, 5120, 8192]

LINE_VALS = ["jit", "flashinfer"]
LINE_NAMES = ["SGL JIT Kernel", "FlashInfer"]
STYLES = [("blue", "--"), ("green", "-.")]

configs = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={},
    )
)
def benchmark_rmsnorm(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": lambda: jit_rmsnorm(input.clone(), weight),
        "flashinfer": lambda: fi_rmsnorm(input.clone(), weight, out=input.clone()),
    }
    fn = FN_MAP[provider]
    return run_benchmark(fn)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark_fused_add_rmsnorm(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    residual = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": lambda: jit_fused_add_rmsnorm(
            input.clone(), residual.clone(), weight, torch.finfo(DTYPE).eps
        ),
        "flashinfer": lambda: fi_fused_add_rmsnorm(
            input.clone(), residual.clone(), weight, eps=torch.finfo(DTYPE).eps
        ),
    }
    fn = FN_MAP[provider]
    return run_benchmark(fn)


if __name__ == "__main__":
    print("Benchmarking rmsnorm...")
    benchmark_rmsnorm.run(print_data=True)

    print("Benchmarking fused_add_rmsnorm...")
    benchmark_fused_add_rmsnorm.run(print_data=True)
