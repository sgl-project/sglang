"""Fail-closed adapter for AITER's gfx950 MXFP4 GEMM on Kimi-K3's latent projections.

The checkpoint ships routed_expert_down_proj / routed_expert_up_proj in BF16 and
sglang has no MXFP4 linear method on ROCm (``Mxfp4Config`` hands ``LinearBase`` an
``UnquantizedLinearMethod``), so the conversion is done by hand here and the model
keeps only the policy for when to use it.

Weights are quantized once after loading; activations are quantized per call. Both
sides use the layout ``gemm_a4w4`` reads with ``bpreshuffle=True``, taken from
aiter's ``op_tests/test_gemm_a4w4.py``: the values and the E8M0 block scale both
come out of the shuffled quant, and the values then take the (16, 16) preshuffle.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

# gemm_a4w4 is a gfx950 ASM kernel (f4gemm_bf16_per1x32Fp4_BpreShuffle_*), absent
# on gfx942, so the whole path is inert anywhere else.
_SUPPORTED = is_hip() and is_gfx95_supported()
_PRESHUFFLE_LAYOUT = (16, 16)


def supported() -> bool:
    """Whether this device can run the MXFP4 path at all; the caller owns the flag."""
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
    """Quantize a BF16 ``[out, in]`` weight to MXFP4, preshuffled for gemm_a4w4."""
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
    """Footprint of :func:`pack`'s output: packed values plus the E8M0 scale."""
    n, k = weight_shape
    return n * k // 2 + n * k // 32


def run(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in MXFP4, quantizing ``x`` on the way in."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP4 GEMM is unavailable")
    quant, _, gemm, fp4x2 = ops
    rows = x.shape[0]
    xq, x_scale = quant(x, quant_dtype=fp4x2, shuffle=True)
    # gemm_a4w4 rounds its output up to the scale block; trim it back.
    return gemm(xq, weight, x_scale, scale, dtype=x.dtype, bpreshuffle=True)[:rows]
