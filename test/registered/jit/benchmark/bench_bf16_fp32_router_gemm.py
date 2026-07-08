"""Benchmark BF16 activation x FP32 router GEMM.

Compares cublas against the in-tree JIT bf16xfp32 kernel on an SM90 GPU:
    python test/registered/jit/benchmark/bench_bf16_fp32_router_gemm.py
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.dsv4.gemm import _linear_bf16_fp32_jit
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _cublas_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.mm(x.float(), w.t())


def _jit_bf16xfp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = _linear_bf16_fp32_jit(x, w, min_m=1)
    if out is None:
        marker.skip("JIT bf16xfp32 path is unavailable for this shape/device")
    return out


FN_MAP = {
    "cublas_fp32": _cublas_fp32,
    "jit_bf16xfp32": _jit_bf16xfp32,
}


@marker.parametrize("m", [1, 2, 4, 8, 16, 48, 64, 96, 208, 512, 1024], [1, 8, 64])
@marker.parametrize("n", [192, 256, 384, 512], [256])
@marker.parametrize("k", [4096, 7168], [4096])
@marker.benchmark("provider", ["cublas_fp32", "jit_bf16xfp32"])
def benchmark(m, n, k, provider):
    x = create_random(m, k, dtype=torch.bfloat16)
    w = create_random(n, k, dtype=torch.float32)
    return marker.do_bench(FN_MAP[provider], input_args=(x, w))


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major != 9:
        print("The JIT bf16xfp32 benchmark requires an SM90 CUDA GPU.")
    else:
        benchmark.run()
