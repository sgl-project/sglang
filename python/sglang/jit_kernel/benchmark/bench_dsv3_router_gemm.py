"""Benchmark for DeepSeek V3 router GEMM (JIT kernel vs torch.nn.functional.linear).

Run on a Hopper (SM90+) GPU:
    python -m sglang.jit_kernel.benchmark.bench_dsv3_router_gemm
"""

import itertools

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.jit_kernel.dsv3_router_gemm import (
    can_use_dsv3_router_gemm,
    dsv3_router_gemm,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=5, suite="stage-b-kernel-benchmark-1-gpu-large")

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"
HIDDEN_DIM = 7168

# num_tokens: 1-16 (the kernel's sweet spot)
# num_experts: 256 (DeepSeek-V3) or 384 (Kimi-K2)
if IS_CI:
    NUM_TOKENS_LIST = [1, 8, 16]
    NUM_EXPERTS_LIST = [256]
else:
    NUM_TOKENS_LIST = list(range(1, 17))
    NUM_EXPERTS_LIST = [256, 384]

LINE_VALS = ["jit", "torch"]
LINE_NAMES = ["SGL JIT Kernel", "torch F.linear"]
STYLES = [("blue", "--"), ("green", "-.")]

configs = list(itertools.product(NUM_EXPERTS_LIST, NUM_TOKENS_LIST))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts", "num_tokens"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsv3-router-gemm-bf16-output",
        args={"out_dtype": torch.bfloat16},
    )
)
def benchmark_bf16_output(
    num_experts: int, num_tokens: int, provider: str, out_dtype: torch.dtype
):
    mat_a = torch.randn((num_tokens, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((num_experts, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": lambda: dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype),
        "torch": lambda: F.linear(mat_a, mat_b).to(out_dtype),
    }
    return run_benchmark(FN_MAP[provider])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts", "num_tokens"],
        x_vals=configs,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsv3-router-gemm-float32-output",
        args={"out_dtype": torch.float32},
    )
)
def benchmark_float32_output(
    num_experts: int, num_tokens: int, provider: str, out_dtype: torch.dtype
):
    mat_a = torch.randn((num_tokens, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((num_experts, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
    FN_MAP = {
        "jit": lambda: dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype),
        "torch": lambda: F.linear(mat_a, mat_b).to(out_dtype),
    }
    return run_benchmark(FN_MAP[provider])


if __name__ == "__main__":
    if not can_use_dsv3_router_gemm(256, HIDDEN_DIM):
        print(
            "dsv3_router_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        print("Benchmarking dsv3_router_gemm (bfloat16 output)...")
        benchmark_bf16_output.run(print_data=True)

        print("Benchmarking dsv3_router_gemm (float32 output)...")
        benchmark_float32_output.run(print_data=True)
