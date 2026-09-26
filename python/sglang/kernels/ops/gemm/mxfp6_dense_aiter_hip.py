"""Fail-closed adapter for AITER's gfx950 MXFP6 GEMM on dense BF16 projections.

Same job as the MXFP4 adapter next door, one format wider. MXFP6 is E2M3 with
the same per-1x32 E8M0 block scale, so it buys two mantissa bits over MXFP4 --
and on CDNA4 it buys them for free in peak terms, because the matrix core runs
MXFP6 at the MXFP4 rate (10.1 PFLOPS each, against 5 for MXFP8). Measured on
Qwen3.5-397B's four widest BF16 projections that lands at ~1.9x over tuned BF16
with ~4% relative error, where MXFP4 is ~2.6x at ~16%.

That error figure is the whole reason this module exists: MXFP4 at 16% cost
8.5 points of gsm8k on this model, while MXFP6 sits at fp8's error (~3.7%) for
considerably more speed than fp8 can reach here (~1.4x).

Both operands must be packed by ``quant_mxfp6_gemm``, which fuses a 32-point
Walsh-Hadamard along K into the quant path. That rotation is part of the packed
operand contract rather than an option -- it is what keeps a per-32 E8M0 block
scale accurate when magnitudes inside a block differ, and since H is orthonormal
and applied to both sides, ``(A@H)(B@H)^T == A@B^T`` leaves the GEMM unchanged.
"""

from __future__ import annotations

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

# The a6w6 path is gfx950 ASM (hsa/gfx950/f6gemm), absent elsewhere, so this
# module is inert on any other device.
_SUPPORTED = is_hip() and is_gfx95_supported()

# quant_mxfp6_gemm pads rows to 256 and K to 128 internally, so shape handling
# is the packer's problem rather than ours.
_K_ALIGN = 128
_ROW_ALIGN = 256


def supported() -> bool:
    """Whether this device can run the MXFP6 path at all; the caller owns the flag."""
    return _SUPPORTED


def _ops():
    try:
        import aiter
        from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm
    except (ImportError, ModuleNotFoundError):
        return None
    return quant_mxfp6_gemm, aiter.gemm_a6w6


def packable(weight: torch.Tensor) -> bool:
    """Whether :func:`pack` would accept this weight, for a caller that must not raise.

    The packer pads both dims itself, so this only rejects what it cannot
    interpret at all. Padding a tiny projection would cost more than the GEMM
    saves, so require the shape to at least fill one kernel tile.
    """
    if not _SUPPORTED or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return False
    out_dim, in_dim = weight.shape
    return in_dim % _K_ALIGN == 0 and out_dim % _ROW_ALIGN == 0


def pack(
    weight: torch.Tensor, what: str = "weight"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 ``[out, in]`` weight to packed MXFP6 for gemm_a6w6."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP6 quantizer is unavailable")
    quant, _ = ops
    if not packable(weight):
        raise RuntimeError(
            f"MXFP6 needs a 2-D bf16 {what} with a {_K_ALIGN}-aligned input dim "
            f"and {_ROW_ALIGN}-aligned output dim, got {weight.dtype} "
            f"{tuple(weight.shape)}"
        )
    return quant(weight.contiguous())


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    """``x @ weight.T`` in MXFP6, quantizing (and rotating) ``x`` on the way in."""
    ops = _ops()
    if ops is None:
        raise RuntimeError("AITER MXFP6 GEMM is unavailable")
    quant, gemm = ops
    xq, x_scale = quant(x)
    return gemm(xq, weight, x_scale, scale, x.shape[0], out_features, x.shape[1])
