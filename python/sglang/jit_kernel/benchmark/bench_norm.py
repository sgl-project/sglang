import itertools

import torch
import triton
import triton.testing
from flashinfer.norm import fused_add_rmsnorm as fi_fused_add_rmsnorm
from flashinfer.norm import rmsnorm as fi_rmsnorm

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-benchmark-1-gpu-large")


DTYPE = torch.bfloat16
DEVICE = "cuda"

BS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(0, 14)],
    ci_range=[16, 32],
)
HIDDEN_SIZE_LIST = get_benchmark_range(
    full_range=sorted([1536, *range(1024, 8192 + 1, 1024)]),
    ci_range=[512, 2048],
)

LINE_VALS = ["flashinfer", "jit"]
LINE_NAMES = ["FlashInfer", "SGL JIT Kernel"]
STYLES = [("blue", "--"), ("green", "-.")]
NUM_LAYERS = 4  # avoid L2 effect

configs_0 = list(itertools.product(HIDDEN_SIZE_LIST + [16384], BS_LIST))
configs_1 = list(itertools.product(HIDDEN_SIZE_LIST, BS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs_0,
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
    input = torch.randn(
        (NUM_LAYERS, batch_size, hidden_size), dtype=DTYPE, device=DEVICE
    )
    weight = torch.randn((NUM_LAYERS, hidden_size), dtype=DTYPE, device=DEVICE)
    FN_MAP = {"jit": jit_rmsnorm, "flashinfer": fi_rmsnorm}

    def f():
        fn = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            fn(input[i], weight[i], out=input[i])

    return run_benchmark(f, scale=NUM_LAYERS)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size"],
        x_vals=configs_1,
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
    input = torch.randn(
        (NUM_LAYERS, batch_size, hidden_size), dtype=DTYPE, device=DEVICE
    )
    residual = torch.randn_like(input)
    weight = torch.randn((NUM_LAYERS, hidden_size), dtype=DTYPE, device=DEVICE)
    FN_MAP = {"jit": jit_fused_add_rmsnorm, "flashinfer": fi_fused_add_rmsnorm}

    def f():
        fn = FN_MAP[provider]
        for i in range(NUM_LAYERS):
            fn(input[i], residual[i], weight[i])

    return run_benchmark(f, scale=NUM_LAYERS)


if __name__ == "__main__":
    print("Benchmarking rmsnorm...")
    benchmark_rmsnorm.run(print_data=True)

    print("Benchmarking fused_add_rmsnorm...")
    benchmark_fused_add_rmsnorm.run(print_data=True)
