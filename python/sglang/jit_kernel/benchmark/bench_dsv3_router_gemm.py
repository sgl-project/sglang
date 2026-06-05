"""Benchmark for DeepSeek V3 router GEMM (JIT kernel vs sgl_kernel AOT vs torch).

Run on a Hopper (SM90+) GPU:
    python -m sglang.jit_kernel.benchmark.bench_dsv3_router_gemm
"""

import torch
import torch.nn.functional as F

try:
    from sgl_kernel import dsv3_router_gemm as sgl_kernel_dsv3_router_gemm
except ImportError:
    sgl_kernel_dsv3_router_gemm = None

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-benchmark-1-gpu-large")

# sgl_kernel AOT kernel is specialized for hidden_dim=7168 only.
SGL_KERNEL_HIDDEN_DIM = 7168


def _torch(mat_a, mat_b, out_dtype):
    return F.linear(mat_a, mat_b).to(out_dtype)


FN_MAP = {
    "jit": dsv3_router_gemm,
    "sgl_kernel": sgl_kernel_dsv3_router_gemm,
    "torch": _torch,
}


@marker.parametrize("num_experts", [256, 384], [256])
@marker.parametrize("hidden_dim", [6144, 7168], [7168])
@marker.parametrize("num_tokens", list(range(1, 17)), [1, 8, 16])
@marker.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@marker.benchmark("provider", ["jit", "sgl_kernel", "torch"])
def benchmark(num_experts, hidden_dim, num_tokens, out_dtype, provider):
    if provider == "sgl_kernel":
        if sgl_kernel_dsv3_router_gemm is None:
            marker.skip("sgl_kernel dsv3_router_gemm not available in this build")
        if hidden_dim != SGL_KERNEL_HIDDEN_DIM:
            marker.skip("sgl_kernel AOT only supports hidden_dim=7168")
    mat_a = create_random(num_tokens, hidden_dim)
    mat_b = create_random(num_experts, hidden_dim)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(mat_a, mat_b),
        input_kwargs={"out_dtype": out_dtype},
    )


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_router_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        benchmark.run()
