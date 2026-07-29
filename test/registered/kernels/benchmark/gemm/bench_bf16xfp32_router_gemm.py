"""Benchmark for the HPC-Ops bf16xfp32 router GEMM (HPC-Ops vs cublas fp32).

`linear_bf16_fp32` computes `x[m, k](bf16) @ w[n, k](fp32)^T`. The `hpc`
provider requires HPC-Ops (https://github.com/Tencent/hpc-ops) installed and
a Hopper GPU (sm90a); it decomposes the fp32 weight into two cached bf16
halves and runs both bf16 GEMMs fused on tensor cores. The `cublas` provider
upcasts the activation to fp32 and is what every model uses without HPC-Ops.

Shapes are the LongCat-Flash router shapes: (hidden_size, n_routed_experts +
zero_experts) = (6144, 768) for Chat and (3072, 384) for Lite.

Run on a Hopper (SM90) GPU with HPC-Ops installed:
    python -m sglang.kernels.jit.benchmark.bench_bf16xfp32_router_gemm
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.attention.dsv4.gemm import (
    _linear_bf16_fp32_hpc,
    mark_hpc_bf16xfp32_gemm_enabled,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

mark_hpc_bf16xfp32_gemm_enabled()


def _cublas_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(x.float(), w.t())


def _hpc(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = _linear_bf16_fp32_hpc(x, w, min_m=1)
    if out is None:
        marker.skip("HPC-Ops is not installed, or this GPU is not Hopper (sm90a)")
    return out


FN_MAP = {
    "cublas": _cublas_fp32,
    "hpc": _hpc,
}


@marker.parametrize(
    "m", [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], [1, 64, 8192]
)
@marker.parametrize("n,k", [(768, 6144), (384, 3072)], [(768, 6144)])
@marker.benchmark("provider", ["cublas", "hpc"])
def benchmark(m, n, k, provider):
    x = create_random(m, k, dtype=torch.bfloat16)
    w = create_random(n, k, dtype=torch.float32)
    return marker.do_bench(FN_MAP[provider], input_args=(x, w))


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major != 9:
        print(
            "The HPC-Ops bf16xfp32 GEMM requires a Hopper (sm90a) CUDA GPU. Skipping."
        )
    else:
        benchmark.run()
