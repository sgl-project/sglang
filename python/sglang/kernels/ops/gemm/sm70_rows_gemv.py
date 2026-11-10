"""Row-exact batched FP16/FP32 GEMV for GLM-5.3 decode and MTP verify on SM70.

One config per weight shape, never per row count, so a verify row is bitwise
the decode output for the same input row.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

# (batch, output features, input features, weight dtype) -> (threads, lanes per output).
# Swept on V100 at one and four rows.
_CONFIGS = {
    # GLM-5.3-Flash TP8: indexer wk and k-pool compress gate, indexer weights_proj,
    # MLA w_kc and w_vc absorption, KDA fused f/g up-projection.
    (1, 128, 4096, torch.float16): (256, 256),
    (1, 32, 4096, torch.float32): (256, 256),
    (8, 512, 256, torch.float16): (256, 8),
    (8, 256, 512, torch.float16): (256, 16),
    (2, 1024, 128, torch.float16): (256, 8),
}


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """x [B, M, K] or [M, K] fp16 with M <= 4; weight [B, N, K] or [N, K]."""
    return (
        envs.SGLANG_SM70_DENSE_GEMV.get()
        and x.is_cuda
        and x.dtype == torch.float16
        and x.ndim == weight.ndim
        and x.ndim in (2, 3)
        and 1 <= x.shape[-2] <= 4
        and x.shape[-1] == weight.shape[-1]
        and (x.ndim == 2 or x.shape[0] == weight.shape[0])
        and _key(weight) in _CONFIGS
        and _rows_aligned(x, 8)
        and _rows_aligned(weight, 16 // weight.element_size())
        and _is_sm70(x.device.index)
    )


def _rows_aligned(t: torch.Tensor, elems: int) -> bool:
    return (
        t.stride(-1) == 1
        and t.data_ptr() % 16 == 0
        and all(s % elems == 0 for s in t.stride()[:-1])
    )


@cache_once
def _is_sm70(device_index) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


def _key(weight):
    shape = tuple(weight.shape) if weight.ndim == 3 else (1, *weight.shape)
    return (*shape, weight.dtype)


@cache_once
def _module(rows, threads, lanes):
    return load_jit(
        "sm70_rows_gemv",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_rows_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_rows_gemv::run<{rows},{threads},{lanes}>")
        ],
    )


def bmm_with(x, weight, threads, lanes, out=None):
    """x [B, M, K], weight [B, N, K] -> out [B, M, N] in the weight dtype."""
    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[1], weight.shape[1]), dtype=weight.dtype, device=x.device
        )
    _module(x.shape[1], threads, lanes).run(x, weight, out)
    return out


def bmm(x, weight, out=None):
    return bmm_with(x, weight, *_CONFIGS[_key(weight)], out=out)


def linear(x, weight):
    """x [M, K] fp16, weight [N, K] -> [M, N] in the weight dtype."""
    return bmm(x.unsqueeze(0), weight.unsqueeze(0))[0]
