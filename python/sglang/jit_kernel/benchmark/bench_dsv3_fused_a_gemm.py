"""Benchmark for DeepSeek V3 fused QKV-A GEMM (JIT kernel vs sgl_kernel AOT vs torch).

Run on a Hopper (SM90+) GPU:
    python -m sglang.jit_kernel.benchmark.bench_dsv3_fused_a_gemm
"""

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import dsv3_fused_a_gemm as sgl_kernel_dsv3_fused_a_gemm

from sglang.jit_kernel.benchmark.utils import run_benchmark_cupti
from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-benchmark-1-gpu-large")

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"
HD_IN = 7168
HD_OUT = 2112

NUM_TOKENS_LIST = [1, 8, 16] if IS_CI else list(range(1, 17))

LINE_VALS = ["jit", "sgl_kernel", "torch"]
LINE_NAMES = ["SGL JIT Kernel", "sgl_kernel AOT", "torch F.linear"]
STYLES = [("blue", "--"), ("red", ":"), ("green", "-.")]


def _bench(num_tokens, provider):
    mat_a = torch.randn((num_tokens, HD_IN), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((HD_OUT, HD_IN), dtype=DTYPE, device=DEVICE).transpose(0, 1)
    fn_map = {
        "jit": lambda: dsv3_fused_a_gemm(mat_a, mat_b),
        "sgl_kernel": lambda: sgl_kernel_dsv3_fused_a_gemm(mat_a, mat_b),
        "torch": lambda: F.linear(mat_a, mat_b.T),
    }
    return run_benchmark_cupti(fn_map[provider])


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=NUM_TOKENS_LIST,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsv3-fused-a-gemm-bf16",
        args={},
    )
)
def benchmark(num_tokens, provider):
    return _bench(num_tokens, provider)


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_fused_a_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        benchmark.run(print_data=True)
