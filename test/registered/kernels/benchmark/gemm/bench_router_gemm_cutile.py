"""Compare prepared-weight cuTile and cuBLAS router GEMMs with cast + TF32.

Run this file on Blackwell. Weight preparation is outside the timed region;
split-K reduction is included. Latencies use rotating inputs and CUDA graphs.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.gemm.router_gemm_cutile import router_gemm_bf16x3_cutile
from sglang.kernels.ops.gemm.router_gemv import split_fp32_to_bf16x3
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _cast_tf32(x, w):
    return x.float() @ w.T


def _cublas(x, parts):
    n = parts.shape[0] // 3
    return torch.mm(x, parts.T, out_dtype=torch.float32).view(-1, 3, n).sum(1)


@marker.parametrize(
    "m",
    [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    [1, 32, 1024, 16384],
)
@marker.parametrize("n,k", [(256, 5120), (128, 6144)], [(256, 5120)])
@marker.benchmark(
    "provider", ["cast_tf32", "tiny_fp32", "cublas_bf16x3", "cutile_bf16x3"]
)
def benchmark(m, n, k, provider):
    if provider == "tiny_fp32" and m > 32:
        marker.skip("Tiny GEMM comparison is limited to M <= 32.")
    x = create_random(m, k, dtype=torch.bfloat16)
    w = create_random(n, k, dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if provider == "cast_tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            return marker.do_bench(_cast_tf32, input_args=(x, w))
        if provider == "tiny_fp32":
            return marker.do_bench(
                tiny_gemm_bf16,
                input_args=(x, w),
                input_kwargs={
                    "out_dtype": torch.float32,
                    "max_m": 16 if m <= 16 else 32,
                },
            )
        parts = split_fp32_to_bf16x3(w)
        fn = _cublas if provider == "cublas_bf16x3" else router_gemm_bf16x3_cutile
        return marker.do_bench(fn, input_args=(x, parts))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("Blackwell required.")
    else:
        benchmark.run()
