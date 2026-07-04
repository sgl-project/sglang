"""Benchmark for DeepSeek V3 fused QKV-A GEMM: CuTe DSL vs CUDA JIT vs
sgl_kernel AOT vs torch.

Run on SM90+ (Hopper or later):
    python test/registered/jit/benchmark/bench_dsv3_fused_a_gemm.py
"""

import torch
import torch.nn.functional as F
from sgl_kernel import dsv3_fused_a_gemm as sgl_kernel_dsv3_fused_a_gemm

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm as cutedsl_dsv3_fused_a_gemm,
)
from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DTYPE = torch.bfloat16
DEVICE = "cuda"
HD_OUT = 2112
HD_IN_LIST = [6144, 7168]

AOT_HD_IN = 7168
HAS_AOT = not is_sm120_supported()

NUM_TOKENS_LIST = list(range(1, 17))
NUM_TOKENS_CI_LIST = [1, 8, 16]

LINE_VALS = ["cutedsl", "jit", "sgl_kernel", "torch"]


@marker.parametrize("hd_in", HD_IN_LIST)
@marker.parametrize("num_tokens", NUM_TOKENS_LIST, NUM_TOKENS_CI_LIST)
@marker.benchmark("provider", LINE_VALS)
def benchmark(num_tokens: int, provider: str, hd_in: int) -> marker.BenchResult:
    if provider == "sgl_kernel" and not (HAS_AOT and hd_in == AOT_HD_IN):
        marker.skip("sgl_kernel AOT is only available for the supported HD_IN")

    mat_a = torch.randn((num_tokens, hd_in), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((HD_OUT, hd_in), dtype=DTYPE, device=DEVICE).transpose(0, 1)
    fn_map = {
        "cutedsl": cutedsl_dsv3_fused_a_gemm,
        "jit": dsv3_fused_a_gemm,
        "sgl_kernel": sgl_kernel_dsv3_fused_a_gemm,
        "torch": lambda a, b: F.linear(a, b.T),
    }
    return marker.do_bench(
        fn_map[provider],
        input_args=(mat_a, mat_b),
        use_cuda_graph=True,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_fused_a_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        benchmark.run()
