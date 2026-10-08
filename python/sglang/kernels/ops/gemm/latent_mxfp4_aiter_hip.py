"""Fail-closed adapter for AITER's gfx950 MXFP4 Kimi-K3 latent GEMMs."""

from __future__ import annotations

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

_SUPPORTED = is_hip() and is_gfx95_supported()
_PRESHUFFLE_LAYOUT = (16, 16)


def supported() -> bool:
    return _SUPPORTED


def _ops():
    try:
        from aiter import QuantType, dtypes, gemm_a4w4
        from aiter.ops.quant import get_hip_quant
        from aiter.ops.shuffle import shuffle_weight
    except (ImportError, ModuleNotFoundError):
        return None
    return get_hip_quant(QuantType.per_1x32), shuffle_weight, gemm_a4w4, dtypes.fp4x2


def pack(
    weight: torch.Tensor, what: str = "weight"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 [out, in] weight for preshuffled AITER MXFP4 GEMM."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP4 quantizer is unavailable")
    quant, shuffle, _, fp4x2 = ops
    if weight.dtype != torch.bfloat16 or weight.shape[-1] % 32 != 0:
        raise RuntimeError(
            f"MXFP4 needs a bf16 {what} with a 32-aligned input dim, got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    q, scale = quant(weight.contiguous(), quant_dtype=fp4x2, shuffle=True)
    return shuffle(q, layout=_PRESHUFFLE_LAYOUT), scale


def packed_bytes(weight_shape: tuple[int, int]) -> int:
    """Return packed MXFP4 values plus E8M0 scale storage."""
    n, k = weight_shape
    return n * k // 2 + n * k // 32


def run(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Compute x @ weight.T after per-1x32 MXFP4 activation quantization."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP4 GEMM is unavailable")
    quant, _, gemm, fp4x2 = ops
    rows = x.shape[0]
    xq, x_scale = quant(x, quant_dtype=fp4x2, shuffle=True)
    return gemm(xq, weight, x_scale, scale, dtype=x.dtype, bpreshuffle=True)[:rows]
