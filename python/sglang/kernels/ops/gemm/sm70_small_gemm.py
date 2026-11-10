"""FP16 projections for two- to four-token verification rows."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

# (rows, output features, input features) -> (threads, lanes per output).
# Shapes where cuBLAS is faster deliberately stay on the existing path.
_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
    (2, 16384, 2560): (128, 32),
    (4, 16384, 2560): (128, 32),
}


def shape_supported(x, weight):
    return (
        envs.SGLANG_SM70_MTP_SMALL_GEMM.get()
        and (x.shape[0], *weight.shape) in _CONFIGS
    )


@cache_once
def _module(rows, threads, lanes):
    return load_jit(
        "sm70_small_gemm",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear(x, weight):
    threads, lanes = _CONFIGS[(x.shape[0], *weight.shape)]
    return linear_with(x, weight, threads, lanes)


def linear_with(x, weight, threads, lanes):
    """x has 2-4 rows; threads per block and lanes per output are explicit."""
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _module(x.shape[0], threads, lanes).run(x, weight, out)
    return out
