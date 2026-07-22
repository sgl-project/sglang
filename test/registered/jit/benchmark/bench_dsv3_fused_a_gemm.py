"""Benchmark for DeepSeek V3 fused QKV-A GEMM: CuTe DSL vs CUDA JIT vs torch.

Run on SM90+ (Hopper or later):
    python test/registered/jit/benchmark/bench_dsv3_fused_a_gemm.py
"""

import torch
import torch.nn.functional as F
import triton.testing

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm as cutedsl_dsv3_fused_a_gemm,
)
from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"
HD_OUT = 2112
HD_IN_LIST = [6144, 7168]

NUM_TOKENS_LIST = [1, 8, 16] if IS_CI else list(range(1, 17))

LINE_VALS = ["cutedsl", "jit", "torch"]
LINE_NAMES = ["CuTe DSL", "CUDA JIT", "torch F.linear"]
STYLES = [("blue", "-"), ("orange", "--"), ("green", "-.")]


def _median_us(fn, *args) -> float:
    result = marker.do_bench(
        fn,
        input_args=args,
        use_cuda_graph=True,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6


def _bench(num_tokens, provider, hd_in):
    mat_a = torch.randn((num_tokens, hd_in), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((HD_OUT, hd_in), dtype=DTYPE, device=DEVICE).transpose(0, 1)
    fn_map = {
        "cutedsl": cutedsl_dsv3_fused_a_gemm,
        "jit": dsv3_fused_a_gemm,
        "torch": lambda a, b: F.linear(a, b.T),
    }
    return _median_us(fn_map[provider], mat_a, mat_b)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=NUM_TOKENS_LIST,
            line_arg="provider",
            line_vals=LINE_VALS,
            line_names=LINE_NAMES,
            styles=STYLES,
            ylabel="us",
            plot_name=f"dsv3-fused-a-gemm-bf16-K{hd_in}-N{HD_OUT}",
            args={"hd_in": hd_in},
        )
        for hd_in in HD_IN_LIST
    ]
)
def benchmark(num_tokens, provider, hd_in):
    return _bench(num_tokens, provider, hd_in)


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_fused_a_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        benchmark.run(print_data=True)
