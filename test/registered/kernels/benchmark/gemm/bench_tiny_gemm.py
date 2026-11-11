"""Benchmark for the tiny GEMM: both JIT templates (SIMT / MMA), the dispatched
entry point, and torch.

Run on a Hopper (SM90+) GPU:
    python -m sglang.kernels.jit.benchmark.bench_tiny_gemm
"""

import functools
import itertools

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.jit.utils import parallel_compile
from sglang.kernels.ops.gemm import tiny_gemm as tg
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _torch(x, w, out_dtype):
    return F.linear(x, w).to(out_dtype)


def _jit(x, w, out_dtype):
    return tg.tiny_gemm_bf16(x, w, out_dtype=out_dtype)


def _supported(impl, n, k):
    if impl == "simt":
        return tg._supports_simt(n, k)
    return tg._supports_mma(n, k, torch.bfloat16)


def _template(impl):
    def run(x, w, out_dtype):
        n, k = w.shape
        if not _supported(impl, n, k):
            return None
        out = torch.empty((x.shape[0], n), dtype=out_dtype, device=x.device)
        tg._get_module(n, k, w.dtype, out_dtype, impl == "mma").run(x, w, out)
        return out

    return run


FN_MAP = {
    "jit": _jit,
    "simt": _template("simt"),
    "mma": _template("mma"),
    "torch": _torch,
}

SHAPES = [
    (256, 7168),
    (384, 7168),
    (256, 6144),
    (256, 4096),
    (384, 5120),
    (896, 7168),
    (144, 7168),
    (128, 2048),
    (128, 4096),
    (512, 2048),
    (512, 4096),
    (256, 3072),
    (160, 6144),
    (512, 2560),
    (32, 2880),
    (1536, 128),
    (128, 512),
    (32, 5120),
]
CI_SHAPES = [(384, 7168)]
OUT_DTYPES = [torch.bfloat16, torch.float32]


def _build_all():
    """Build every module the benchmark reaches in parallel, before timing."""
    builders = []
    for (n, k), out_dtype in itertools.product(
        CI_SHAPES if is_in_ci() else SHAPES, OUT_DTYPES
    ):
        for impl in ("simt", "mma"):
            if _supported(impl, n, k):
                builders.append(
                    functools.partial(
                        tg._get_module, n, k, torch.bfloat16, out_dtype, impl == "mma"
                    )
                )
    parallel_compile(builders)


@marker.parametrize("out_dtype", OUT_DTYPES)
@marker.parametrize("shape", SHAPES, CI_SHAPES)
@marker.parametrize("num_tokens", [1, 2, 4, 8, 16, 32], [1, 8, 32])
@marker.benchmark("provider", ["jit", "simt", "mma", "torch"])
def benchmark(shape, num_tokens, out_dtype, provider):
    n, k = shape
    x = create_random(num_tokens, k)
    w = create_random(n, k)
    fn = FN_MAP[provider]
    if fn(x, w, out_dtype) is None:
        marker.skip(f"{provider} does not support N={n}, K={k}")
    return marker.do_bench(
        fn,
        input_args=(x, w),
        input_kwargs={"out_dtype": out_dtype},
    )


if __name__ == "__main__":
    _build_all()
    benchmark.run()
