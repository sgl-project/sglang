"""Fail-closed adapter for AITER's gfx950 MXFP4 Kimi-K3 latent GEMMs."""

from __future__ import annotations

import functools

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

_PRESHUFFLE_LAYOUT = (16, 16)


@functools.cache
def _ops():
    if not (is_hip() and is_gfx95_supported()):
        return None
    try:
        from aiter import QuantType, dtypes, gemm_a4w4
        from aiter.ops.quant import get_hip_quant
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return get_hip_quant(QuantType.per_1x32), shuffle_weight, gemm_a4w4, dtypes.fp4x2


def supported() -> bool:
    return _ops() is not None


def pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preshuffled MXFP4 weight and E8M0 scales for a BF16 [out, in] weight."""
    quant, shuffle, _, fp4x2 = _ops()
    if weight.dtype != torch.bfloat16 or weight.shape[-1] % 32 != 0:
        raise ValueError(
            "MXFP4 needs a BF16 weight with a 32-aligned input dim, got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    q, scale = quant(weight.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return shuffle(q, layout=_PRESHUFFLE_LAYOUT), scale


def run(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x @ weight.T after per-1x32 MXFP4 activation quantization."""
    quant, _, gemm, fp4x2 = _ops()
    rows = x.shape[0]
    xq, x_scale = quant(x, quant_dtype=fp4x2, shuffle=True)
    return gemm(xq, weight, x_scale, scale, dtype=x.dtype, bpreshuffle=True)[:rows]
