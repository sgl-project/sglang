"""AITER gfx950 MXFP4 GEMMs for Kimi-K3 latent projections.

available() bounds them to gfx950 with AITER's MXFP4 ops; pack() and run() raise
ValueError on inputs outside their contract.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from sglang.kernels.arch import current_arch

_PRESHUFFLE_LAYOUT = (16, 16)
_Ops = tuple[
    Callable[..., tuple[torch.Tensor, torch.Tensor]],
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
    torch.dtype,
]


@functools.cache
def _ops() -> _Ops | None:
    try:
        from aiter import QuantType, dtypes, gemm_a4w4
        from aiter.ops.quant import get_hip_quant
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return get_hip_quant(QuantType.per_1x32), shuffle_weight, gemm_a4w4, dtypes.fp4x2


@functools.cache
def available() -> bool:
    """Whether pack() and run() can launch here: gfx950 and AITER's MXFP4 ops."""
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch == "gfx950" and _ops() is not None


def pack(*, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preshuffled MXFP4 weight and E8M0 scales for a BF16 [out, in] weight."""
    # gemm_a4w4 and the preshuffled scales read whole 256x256 tiles.
    if (
        weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or weight.shape[0] == 0
        or weight.shape[0] % 256 != 0
        or weight.shape[1] == 0
        or weight.shape[1] % 256 != 0
        or weight.device.type != "cuda"
    ):
        raise ValueError(
            "MXFP4 needs a nonempty 2D BF16 GPU weight with output/input "
            "dimensions aligned to 256, got "
            f"{weight.dtype} {tuple(weight.shape)} on {weight.device}"
        )
    if not available():
        raise RuntimeError("latent MXFP4 GEMM requires AITER on gfx950")
    quant, shuffle, _, fp4x2 = _ops()
    q, scale = quant(weight.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return shuffle(q, layout=_PRESHUFFLE_LAYOUT), scale


def run(*, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x @ weight.T after per-1x32 MXFP4 activation quantization."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[0] == 0:
        raise ValueError("latent MXFP4 GEMM needs nonempty 2D BF16 activations")
    if (
        weight.ndim != 2
        or x.shape[1] != weight.shape[1] * 2
        or weight.device != x.device
        or scale.device != x.device
        or scale.numel() < weight.shape[0] * x.shape[1] // 32
        or not weight.is_contiguous()
        or not scale.is_contiguous()
    ):
        raise ValueError("latent MXFP4 weight and scales must match the activations")
    # pack() already checked available(), so skip it per call.
    ops = _ops()
    if ops is None or weight.dtype != ops[3]:
        raise ValueError("latent MXFP4 GEMM needs packed FP4 weights from pack()")
    quant, _, gemm, fp4x2 = ops
    xq, x_scale = quant(x.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return gemm(xq, weight, x_scale, scale, dtype=x.dtype, bpreshuffle=True)
