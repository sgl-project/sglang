"""Benchmark for DeepSeek V3 router GEMM (JIT kernel vs torch.nn.functional.linear).

Run on a Hopper (SM90+) GPU:
    python -m sglang.jit_kernel.benchmark.bench_dsv3_router_gemm
"""

import itertools

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import dsv3_router_gemm as sgl_kernel_dsv3_router_gemm

from sglang.jit_kernel.benchmark.utils import run_benchmark
from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-benchmark-1-gpu-large")

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"

if IS_CI:
    NUM_TOKENS_LIST = [1, 8, 16]
    NUM_EXPERTS_LIST = [256]
    HIDDEN_DIM_LIST = [7168]
else:
    NUM_TOKENS_LIST = list(range(1, 17))
    NUM_EXPERTS_LIST = [256, 384]
    HIDDEN_DIM_LIST = [6144, 7168]

# sgl_kernel AOT kernel is specialized for hidden_dim=7168 only.
SGL_KERNEL_HIDDEN_DIM = 7168

LINE_VALS = ["jit", "sgl_kernel", "torch"]
LINE_NAMES = ["SGL JIT Kernel", "sgl_kernel AOT", "torch F.linear"]
STYLES = [("blue", "--"), ("red", ":"), ("green", "-.")]

configs = list(itertools.product(NUM_EXPERTS_LIST, HIDDEN_DIM_LIST, NUM_TOKENS_LIST))


def _bench(num_experts, hidden_dim, num_tokens, provider, out_dtype):
    if provider == "sgl_kernel" and hidden_dim != SGL_KERNEL_HIDDEN_DIM:
        return float("nan"), float("nan"), float("nan")
    mat_a = torch.randn((num_tokens, hidden_dim), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((num_experts, hidden_dim), dtype=DTYPE, device=DEVICE)
    fn_map = {
        "jit": lambda: dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype),
        "sgl_kernel": lambda: sgl_kernel_dsv3_router_gemm(
            mat_a, mat_b, out_dtype=out_dtype
        ),
        "torch": lambda: F.linear(mat_a, mat_b).to(out_dtype),
    }
    return run_benchmark(fn_map[provider])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts", "hidden_dim", "num_tokens"],
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
def benchmark_bf16_output(num_experts, hidden_dim, num_tokens, provider, out_dtype):
    return _bench(num_experts, hidden_dim, num_tokens, provider, out_dtype)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_experts", "hidden_dim", "num_tokens"],
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
def benchmark_float32_output(num_experts, hidden_dim, num_tokens, provider, out_dtype):
    return _bench(num_experts, hidden_dim, num_tokens, provider, out_dtype)


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_router_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        print("Benchmarking dsv3_router_gemm (bfloat16 output)...")
        benchmark_bf16_output.run(print_data=True)

        print("Benchmarking dsv3_router_gemm (float32 output)...")
        benchmark_float32_output.run(print_data=True)
