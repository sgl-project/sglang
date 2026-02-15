import itertools

import torch
import triton
import triton.testing
from flashinfer import rmsnorm as fi_rmsnorm
from sgl_kernel import rmsnorm

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.jit_kernel.norm import rmsnorm as jit_rmsnorm


def sglang_aot_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    rmsnorm(input, weight, out=input)


def sglang_jit_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    jit_rmsnorm(input, weight, output=input)


def flashinfer_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    fi_rmsnorm(input, weight, out=input)


@torch.compile()
def torch_impl_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    mean = input.float().pow(2).mean(dim=-1, keepdim=True)
    norm = (mean + eps).rsqrt()
    input.copy_(input.float() * norm * weight.float())


BS_LIST = get_benchmark_range(
    full_range=[2**n for n in range(0, 14)],
    ci_range=[16],
)
HIDDEN_SIZE_LIST = get_benchmark_range(
    full_range=[1536, 3072, 4096, 5120, 8192],
    ci_range=[512, 2048],
)

LINE_VALS = ["aot", "jit", "fi", "torch"]
LINE_NAMES = ["SGL AOT Kernel", "SGL JIT Kernel", "FlashInfer", "PyTorch"]
STYLES = [("orange", "-"), ("blue", "--"), ("green", "-."), ("red", ":")]

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
def benchmark(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn(
        (batch_size, hidden_size), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE
    )
    weight = torch.randn(hidden_size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "aot": sglang_aot_rmsnorm,
        "jit": sglang_jit_rmsnorm,
        "fi": flashinfer_rmsnorm,
        "torch": torch_impl_rmsnorm,
    }
    fn = lambda: FN_MAP[provider](input.clone(), weight)
    return run_benchmark(fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
