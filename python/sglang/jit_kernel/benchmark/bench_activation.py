import itertools

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import gelu_and_mul as gelu_and_mul_aot
from sgl_kernel import gelu_tanh_and_mul as gelu_tanh_and_mul_aot
from sgl_kernel import silu_and_mul as silu_and_mul_aot

from sglang.jit_kernel.activation import gelu_and_mul as gelu_and_mul_jit
from sglang.jit_kernel.activation import gelu_tanh_and_mul as gelu_tanh_and_mul_jit
from sglang.jit_kernel.activation import silu_and_mul as silu_and_mul_jit
from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-benchmark-1-gpu-large")


@torch.compile
def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.silu(lhs) * rhs


@torch.compile
def gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.gelu(lhs, approximate="none") * rhs


@torch.compile
def gelu_tanh_and_mul(input: torch.Tensor) -> torch.Tensor:
    lhs, rhs = input.split(input.shape[-1] // 2, dim=-1)
    return F.gelu(lhs, approximate="tanh") * rhs


OPS = {
    "silu": (silu_and_mul_aot, silu_and_mul_jit, silu_and_mul),
    "gelu": (gelu_and_mul_aot, gelu_and_mul_jit, gelu_and_mul),
    "gelu_tanh": (gelu_tanh_and_mul_aot, gelu_tanh_and_mul_jit, gelu_tanh_and_mul),
}
BS_LIST = get_benchmark_range(full_range=[2**x for x in range(0, 15)], ci_range=[8])
DIM_LIST = get_benchmark_range(full_range=[1024, 4096, 6144, 8192], ci_range=[4096])
CONFIGS = list(itertools.product(OPS, DIM_LIST, BS_LIST))
NUM_LAYERS = 4  # to eliminate L2 effect


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["op_name", "dim", "batch_size"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["aot", "jit", "torch"],
        line_names=["AOT (sgl-kernel)", "JIT (jit_kernel)", "torch.compile"],
        styles=[("blue", "--"), ("orange", "-"), ("green", "-")],
        ylabel="us",
        plot_name="activation-aot-vs-jit",
        args={},
    )
)
def benchmark(op_name: str, dim: int, batch_size: int, provider: str):
    x = torch.randn(
        NUM_LAYERS,
        batch_size,
        2 * dim,
        dtype=DEFAULT_DTYPE,
        device=DEFAULT_DEVICE,
    )
    aot_op, jit_op, torch_op = OPS[op_name]
    fn = {"aot": aot_op, "jit": jit_op, "torch": torch_op}[provider]

    def f():
        for i in range(NUM_LAYERS):
            fn(x[i])

    return run_benchmark(f, scale=NUM_LAYERS)


if __name__ == "__main__":
    benchmark.run(print_data=True)
