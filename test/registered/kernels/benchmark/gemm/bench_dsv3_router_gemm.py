"""Benchmark for the FlashInfer router GEMM adapter versus torch.

Run on a supported NVIDIA GPU:
    python test/registered/kernels/benchmark/gemm/bench_dsv3_router_gemm.py
"""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm import dsv3_router_gemm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

SUPPORTED_DEVICE_SMS = {90, 100, 103, 107}
ROUTER_GEMM_CONFIGS = [
    (128, 7168, torch.bfloat16),
    (256, 6144, torch.float32),
    (256, 7168, torch.bfloat16),
    (256, 7168, torch.float32),
    (384, 7168, torch.bfloat16),
    (384, 7168, torch.float32),
    (896, 7168, torch.bfloat16),
    (896, 7168, torch.float32),
]


def _torch(mat_a, mat_b, out_dtype):
    return F.linear(mat_a, mat_b).to(out_dtype)


FN_MAP = {
    "flashinfer": dsv3_router_gemm,
    "torch": _torch,
}


@marker.parametrize(
    "num_experts,hidden_dim,out_dtype",
    ROUTER_GEMM_CONFIGS,
    [
        (256, 7168, torch.float32),
        (384, 7168, torch.float32),
        (896, 7168, torch.float32),
    ],
)
@marker.parametrize("num_tokens", list(range(1, 17)), [1, 8, 16])
@marker.benchmark("provider", ["flashinfer", "torch"])
def benchmark(num_experts, hidden_dim, num_tokens, out_dtype, provider):
    mat_a = create_random(num_tokens, hidden_dim)
    mat_b = create_random(num_experts, hidden_dim)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(mat_a, mat_b),
        input_kwargs={"out_dtype": out_dtype},
    )


if __name__ == "__main__":
    arch = get_jit_cuda_arch()
    device_sm = arch.major * 10 + arch.minor
    if is_hip_runtime() or device_sm not in SUPPORTED_DEVICE_SMS:
        print(
            "FlashInfer router GEMM requires SM90, SM100, SM103, or SM107. "
            "Skipping benchmark."
        )
    else:
        benchmark.run()
