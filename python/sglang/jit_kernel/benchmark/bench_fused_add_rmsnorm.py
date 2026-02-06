import itertools

import torch
import triton
import triton.testing
from flashinfer import fused_add_rmsnorm as fi_fused_add_rmsnorm

from sglang.jit_kernel.benchmark.utils import is_in_ci
from sglang.jit_kernel.norm import fused_add_rmsnorm as jit_fused_add_rmsnorm

IS_CI = is_in_ci()


def sglang_jit_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    jit_fused_add_rmsnorm(input, residual, weight, eps)


def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    fi_fused_add_rmsnorm(input, residual, weight, eps=eps)


DTYPE = torch.bfloat16
DEVICE = "cuda"

if IS_CI:
    BS_LIST = [16]
    HIDDEN_SIZE_LIST = [512, 2048]
else:
    BS_LIST = [2**n for n in range(0, 14)]
    HIDDEN_SIZE_LIST = [1536, 3072, 4096, 5120, 8192]

LINE_VALS = ["jit", "fi"]
LINE_NAMES = ["SGL JIT Kernel", "FlashInfer"]
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
        plot_name="fused-add-rmsnorm-performance",
        args={},
    )
)
def benchmark(hidden_size: int, batch_size: int, provider: str):
    input = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    residual = torch.randn((batch_size, hidden_size), dtype=DTYPE, device=DEVICE)
    weight = torch.randn(hidden_size, dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": sglang_jit_fused_add_rmsnorm,
        "fi": flashinfer_fused_add_rmsnorm,
    }
    fn = lambda: FN_MAP[provider](
        input.clone(), residual.clone(), weight, torch.finfo(torch.bfloat16).eps
    )
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)  # type: ignore
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    benchmark.run(print_data=True)
