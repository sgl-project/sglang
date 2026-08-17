"""Benchmark for the tiny GEMM (JIT kernel vs torch).

Run on a Hopper (SM90+) GPU:
    python -m sglang.kernels.jit.benchmark.bench_tiny_gemm
"""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _torch(x, w, out_dtype):
    return F.linear(x, w).to(out_dtype)


def _jit(x, w, out_dtype):
    return tiny_gemm_bf16(x, w, out_dtype=out_dtype, max_m=16)


FN_MAP = {"jit": _jit, "torch": _torch}

SHAPES = [(256, 7168), (384, 7168), (256, 4096), (896, 7168), (144, 7168), (1536, 128)]


@marker.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@marker.parametrize("shape", SHAPES, [(384, 7168)])
@marker.parametrize("num_tokens", list(range(1, 17)), [1, 8, 16])
@marker.benchmark("provider", ["jit", "torch"])
def benchmark(shape, num_tokens, out_dtype, provider):
    n, k = shape
    x = create_random(num_tokens, k)
    w = create_random(n, k)
    return marker.do_bench(
        FN_MAP[provider],
        input_args=(x, w),
        input_kwargs={"out_dtype": out_dtype},
    )


if __name__ == "__main__":
    benchmark.run()
