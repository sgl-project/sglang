"""DeepSeek-V4 ``wo_a`` (MLA output-absorb) fp8 GEMM for AMD gfx950.

``wo_a`` is the first half of the o_proj: it absorbs the attention output into
the low-rank o space before ``wo_b`` projects back to the hidden size. On CUDA
it runs in fp8 via DeepGEMM's ``fp8_einsum`` (``SGLANG_OPT_FP8_WO_A_GEMM``), but
DeepGEMM is CUDA-only, so ROCm falls back to a bf16 batched GEMM and pays for
loading bf16 weights on a decode step that is bound by weight traffic.

This module is the ROCm equivalent, built on aiter's microscaling (e8m0)
block-scale batched GEMM ``batched_gemm_a8w8_mxscale``:

* a Triton quantizer for the post-inverse-RoPE attention output, emitting the
  per-token-group fp8 codes and uint8 e8m0 scales that kernel expects, and
* the load-time conversion of the checkpoint's fp32 block scales to e8m0.

Shapes per attention-TP rank, with ``G`` local o-groups and 128-wide blocks::

    activation  o       [T, G, D]         bf16 -> fp8 e4m3
    act scale   o_s     [T, G, D/128]     uint8 e8m0 (per token-group)
    weight      wo_a    [G, R, D]         fp8 e4m3 (from the checkpoint)
    wgt scale   wo_a_s  [G, R/128, D/128] uint8 e8m0
    output      z       [T, G, R]         bf16

The e8m0 scale is a bare power-of-two exponent, so a block's scale must be
rounded *up* to a power of two before its values are divided by it. The
quantizer does that for activations; ``wo_a_weight_scale_to_e8m0`` does it for
weights, requantizing the weight itself if the checkpoint's scales are not
already powers of two (otherwise the exponent round-up would silently rescale
the weights).
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)

# Block size of the microscaling scales, on both operands and both axes.
WO_A_MXFP8_GROUP_SIZE = 128
_FP8_MAX = 448.0
_ABSMAX_EPS = 1e-10

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()

# The mxscale flatmm BMM is gfx950-only, so resolve availability once at import
# rather than per decode step.
_batched_gemm_a8w8_mxscale = None
if _is_hip and _is_gfx95_supported:
    try:
        from aiter.ops.batched_gemm_op_a8w8 import (
            batched_gemm_a8w8_mxscale as _batched_gemm_a8w8_mxscale,
        )
    except Exception as err:  # pragma: no cover - env-dependent
        logger.warning(
            "aiter batched_gemm_a8w8_mxscale import failed; the DSV4 wo_a fp8 "
            "path is unavailable on this build: %s",
            err,
        )


def is_wo_a_fp8_mxscale_supported() -> bool:
    """True when the ROCm fp8 ``wo_a`` path can run on this build/arch."""
    return _batched_gemm_a8w8_mxscale is not None


@triton.jit
def _wo_a_quant_mxfp8_kernel(
    x_ptr,
    xq_ptr,
    xs_ptr,
    D,
    NUM_GROUPS,
    NUM_GROUPS_PADDED: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    """One program per token-group row of the [T*G, D] view.

    The row is tiled as ``[NUM_GROUPS, GROUP_SIZE]`` so a single program reduces
    every 128-wide block of the row at once. Splitting the row across programs
    instead leaves each one with a single 128-element load, which on a decode
    step is all launch latency and no work.
    """
    row = tl.program_id(0)
    grp = tl.arange(0, NUM_GROUPS_PADDED)
    offs = row * D + grp[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    mask = grp[:, None] < NUM_GROUPS
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Smallest power-of-two scale that keeps a block inside the fp8 range: take
    # ceil(log2(absmax / FP8_MAX)) by bumping the float32 exponent whenever the
    # mantissa is non-zero (deep_gemm's ceil_to_ue8m0 convention).
    raw = tl.maximum(tl.max(tl.abs(x), axis=1) / FP8_MAX, EPS)
    bits = raw.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = exp + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    scale = (exp << 23).to(tl.float32, bitcast=True)

    q = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX)
    tl.store(xq_ptr + offs, q.to(xq_ptr.dtype.element_ty), mask=mask)
    # e8m0 stores the biased exponent itself (127 == scale 1.0).
    tl.store(xs_ptr + row * NUM_GROUPS + grp, exp.to(tl.uint8), mask=grp < NUM_GROUPS)


def quant_wo_a_act_mxfp8(o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the [T, G, D] attention output to fp8 + e8m0 block scales.

    Returns ``(o_fp8 [T, G, D], o_scale [T, G, D/128] uint8)`` in the layout
    ``batched_gemm_a8w8_mxscale`` consumes directly.
    """
    T, G, D = o.shape
    assert D % WO_A_MXFP8_GROUP_SIZE == 0, (
        f"wo_a in-features ({D}) must be divisible by {WO_A_MXFP8_GROUP_SIZE}"
    )
    if not o.is_contiguous():
        o = o.contiguous()

    num_groups = D // WO_A_MXFP8_GROUP_SIZE
    o_fp8 = torch.empty((T, G, D), device=o.device, dtype=torch.float8_e4m3fn)
    o_scale = torch.empty((T, G, num_groups), device=o.device, dtype=torch.uint8)

    _wo_a_quant_mxfp8_kernel[(T * G,)](
        o,
        o_fp8,
        o_scale,
        D,
        num_groups,
        NUM_GROUPS_PADDED=triton.next_power_of_2(num_groups),
        GROUP_SIZE=WO_A_MXFP8_GROUP_SIZE,
        FP8_MAX=_FP8_MAX,
        EPS=_ABSMAX_EPS,
        num_warps=8,
    )
    return o_fp8, o_scale


def apply_wo_a_fp8_mxscale(
    o: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """fp8 ``wo_a``: quantize [T, G, D] then batched-GEMM against [G, R, D].

    ``weight_scale`` is the uint8 e8m0 [G, R/128, D/128] tensor produced at load
    time by ``wo_a_weight_scale_to_e8m0``. Returns bf16 [T, G, R].
    """
    o_fp8, o_scale = quant_wo_a_act_mxfp8(o)
    return _batched_gemm_a8w8_mxscale(
        o_fp8, weight, o_scale, weight_scale, dtype=torch.bfloat16
    )


def _is_power_of_two(scale: torch.Tensor) -> bool:
    """True when every fp32 scale is an exact power of two (zero mantissa)."""
    bits = scale.detach().float().contiguous().view(torch.int32)
    return bool(((bits & 0x7FFFFF) == 0).all().item())


def wo_a_weight_scale_to_e8m0(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    num_groups: int,
    o_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint ``wo_a`` fp8 weight + fp32 block scales to e8m0.

    Returns ``(weight [G, R, D] fp8, scale [G, R/128, D/128] uint8)``.

    DeepSeek-V4 checkpoints that carry fp4 experts already store their linear
    block scales as powers of two, so the conversion is exact and the weight
    passes through untouched. Otherwise the weight is dequantized and requantized
    against power-of-two scales, because rounding the exponent up on its own
    would rescale every value in the block.
    """
    G, R = num_groups, o_lora_rank
    D = weight.shape[-1]
    block = WO_A_MXFP8_GROUP_SIZE

    scale = weight_scale_inv.detach().float()
    if not _is_power_of_two(scale):
        from sglang.srt.layers.quantization.fp8_utils import (
            block_quant_dequant,
            quant_weight_ue8m0,
        )

        logger.info_once(
            "DSV4 wo_a block scales are not power-of-two; requantizing the "
            "weight to ue8m0 for the aiter mxscale GEMM."
        )
        dequant = block_quant_dequant(
            weight.view(G * R, D),
            scale.view(-1, D // block),
            [block, block],
            torch.bfloat16,
        )
        weight, scale = quant_weight_ue8m0(
            weight_dequant=dequant, weight_block_size=[block, block]
        )

    exponent = (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return (
        weight.contiguous().view(G, R, D),
        exponent.contiguous().view(G, R // block, D // block),
    )


__all__ = [
    "WO_A_MXFP8_GROUP_SIZE",
    "apply_wo_a_fp8_mxscale",
    "is_wo_a_fp8_mxscale_supported",
    "quant_wo_a_act_mxfp8",
    "wo_a_weight_scale_to_e8m0",
]
