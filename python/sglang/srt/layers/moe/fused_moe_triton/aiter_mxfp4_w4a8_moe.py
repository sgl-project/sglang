# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/pull/46516
#
# MXFP4-weight / FP8-activation (W4A8) fused MoE for AMD gfx1250 (RDNA / gfx12).
#
# gfx1250's aiter CK/ASM ``fused_moe`` produces garbage for the GPT-OSS MXFP4
# W4A8 layout, so this path routes through aiter's *triton* ``moe_gemm_a8w4``
# kernel instead (the same kernel gfx950 uses). Two gfx1250-specific quirks are
# handled, mirroring the vLLM enablement:
#   1. The in-kernel TDM gather fails to compile on gfx1250, so we disable the
#      TDM routing path and gather activation rows into expert-sorted order in
#      torch (passing ``gather_indx=None`` to the GEMM).
#   2. The gfx1250 ``moe_gemm_a8w4`` reads a CDNA4-swizzled MX scale as garbage,
#      so the weight scale is kept unswizzled and ``swizzle_mx_scale=None`` is
#      passed to the kernel.

from __future__ import annotations

from typing import Optional

import torch

_TDM_DISABLED = False


def _import_aiter_w4a8():
    """Import the aiter triton routing + a8w4 GEMM entry points.

    Returns ``(routing, moe_gemm_a8w4, downcast_to_static_fp8)`` or ``None`` if
    the installed aiter build does not expose the triton W4A8 path.
    """
    try:
        try:
            import aiter.ops.triton.moe.moe_routing.routing as _routing_mod
        except ImportError:
            import aiter.ops.triton.moe_routing.routing as _routing_mod

        from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
        from aiter.ops.triton.moe.quant_moe import (
            downcast_to_static_fp8,
            downcast_to_static_fp8_gather,
        )
    except ImportError:
        return None

    global _TDM_DISABLED
    if not _TDM_DISABLED:
        # gfx1250: the in-kernel TDM gather emitted by the routing sort / GEMM
        # fails to compile (``TDM gather dst must be 2D``). Force the non-TDM
        # path; we gather activations manually below.
        _routing_mod.is_tdm_avail = lambda: False
        _TDM_DISABLED = True

    return _routing_mod.routing, moe_gemm_a8w4, downcast_to_static_fp8, downcast_to_static_fp8_gather


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Convert a SEPARATED ``[gate_0..gate_{I-1}, up_0..up_{I-1}]`` first dim
    (after the expert dim) into the INTERLEAVED ``[gate_0, up_0, gate_1, up_1,
    ...]`` order that ``moe_gemm_a8w4``'s fused SwiGLU expects (gate on the
    even lanes, up on the odd lanes)."""
    e, two_i = t.shape[0], t.shape[1]
    i = two_i // 2
    rest = t.shape[2:]
    t = t.view(e, 2, i, *rest)
    perm = (0, 2, 1) + tuple(range(3, t.dim()))
    return t.permute(*perm).reshape(e, two_i, *rest).contiguous()


def prepare_w4a8_gfx1250_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
):
    """Reshape SGLang's loaded Quark W4A8 MoE buffers into the ``[E, K, N]``
    (contraction-major) packed layout consumed by ``moe_gemm_a8w4``.

    Input (SGLang / HF Quark layout, per expert), output-channel major:
        w13_weight        [E, 2I, H//2]   uint8 (2 FP4 packed along H)
        w13_weight_scale  [E, 2I, H//32]  uint8 (e8m0), gate/up SEPARATED
        w13_weight_bias   [E, 2I]         fp32
        w2_weight         [E, H,  I//2]   uint8
        w2_weight_scale   [E, H,  I//32]  uint8 (e8m0)
        w2_weight_bias    [E, H]          fp32

    Output (moe_gemm_a8w4 layout), contraction (K) major, gate/up INTERLEAVED
    for w13:
        w13  [E, H//2, 2I]   w13_scale [E, H//32, 2I]   w13_bias [E, 2I]
        w2   [E, I//2, H]    w2_scale  [E, I//32, H]    w2_bias  [E, H]
    """
    # Interleave gate/up on w13 (output dim) so the fused SwiGLU picks gate on
    # even lanes and up on odd lanes. The interleave output is contiguous, so
    # the subsequent transpose(1, 2) yields a *column-major* [E, K, N] view
    # (stride(-2) == 1), which ``moe_gemm_a8w4`` requires for MXFP weights.
    w13_weight = _interleave_gate_up(w13_weight)
    w13_weight_scale = _interleave_gate_up(w13_weight_scale)
    w13_weight_bias = _interleave_gate_up(w13_weight_bias)

    # Transpose to contraction-major [E, K(packed), N] *without* making it
    # contiguous, so the K dimension stays unit-strided (column-major).
    w13_weight = w13_weight.transpose(1, 2)
    w13_weight_scale = w13_weight_scale.transpose(1, 2)
    w2_weight = w2_weight.contiguous().transpose(1, 2)
    w2_weight_scale = w2_weight_scale.contiguous().transpose(1, 2)

    return (
        w13_weight,
        w13_weight_scale,
        w13_weight_bias.contiguous(),
        w2_weight,
        w2_weight_scale,
        w2_weight_bias.contiguous(),
    )


def aiter_w4a8_gfx1250_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    a13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
    a2_scale: torch.Tensor,
    gemm1_alpha: float,
    gemm1_limit: float,
    renormalize: bool = True,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """MXFP4 W4A8 GPT-OSS MoE forward for gfx1250 via aiter triton
    ``moe_gemm_a8w4``.

    ``w*`` / ``w*_scale`` / ``w*_bias`` must already be in the
    ``moe_gemm_a8w4`` layout produced by :func:`prepare_w4a8_gfx1250_weights`.
    ``a13_scale`` / ``a2_scale`` are the static per-tensor FP8 activation scales
    for gate_up_proj and down_proj respectively.
    """
    imported = _import_aiter_w4a8()
    if imported is None:
        raise RuntimeError(
            "aiter triton W4A8 MoE (moe_gemm_a8w4) is required for the gfx1250 "
            "GPT-OSS MXFP4 path but was not found in the installed aiter build."
        )
    routing, moe_gemm_a8w4, downcast_to_static_fp8, downcast_to_static_fp8_gather = imported

    assert hidden_states.dtype == torch.bfloat16

    # aiter routing on the raw router logits. renormalize=True (GPT-OSS)
    # corresponds to applying softmax to the top-k selection inside the kernel
    # (sm_first=False).
    routing_data, gather_idx, scatter_idx = routing(
        router_logits, topk, sm_first=not renormalize
    )
    gammas = routing_data.gate_scal

    # gfx1250: the in-kernel gather is broken, so we pass gather_indx=None to
    # moe_gemm_a8w4 and perform the gather ourselves.
    if apply_router_weight_on_input:
        # Router weights must be applied in bf16 before quantization.
        gather_src = gather_idx.to(torch.long) // topk
        x = hidden_states[gather_src]
        x = x * gammas[:, None].to(x.dtype)
        x_fp8 = downcast_to_static_fp8(x, a13_scale)
    else:
        # Fully fused: divide routing indices, gather hidden_states rows, and
        # quantize to fp8 in a single Triton kernel — no intermediate bf16 buffer.
        x_fp8 = downcast_to_static_fp8_gather(
            hidden_states, a13_scale, gather_src_idx=gather_idx, topk=topk
        )

    # GEMM1: FP8 activations x MXFP4 weights, fused SwiGLU, requantize the
    # intermediate to FP8 using the down_proj activation scale (a2_scale) so
    # GEMM2 can consume it directly.
    intermediate_cache1 = moe_gemm_a8w4(
        x_fp8,
        w13_weight,
        None,
        w13_weight_scale,
        a13_scale,
        a2_scale,
        w13_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=None,
        gammas=None,
        swizzle_mx_scale=None,
        out_dtype=x_fp8.dtype,
        apply_swiglu=True,
        alpha=gemm1_alpha,
        limit=gemm1_limit,
    )

    # GEMM2: down projection, scatter back to token order and apply the router
    # weights (gammas) unless they were already applied on the input.
    intermediate_cache3 = moe_gemm_a8w4(
        intermediate_cache1,
        w2_weight,
        None,
        w2_weight_scale,
        a2_scale,
        None,
        w2_weight_bias,
        routing_data,
        gather_indx=None,
        scatter_indx=scatter_idx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=None,
        out_dtype=torch.bfloat16,
    )

    return intermediate_cache3.contiguous()
