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
#      torch (passing ``gather_indx=None`` to the GEMM). With
#      SGLANG_GFX1250_W4A8_BACKEND=triton, SGLANG_GFX1250_W4A8_KERNEL_GATHER=1
#      lets the plain-Triton kernel gather the rows instead.
#   2. The gfx1250 ``moe_gemm_a8w4`` reads a CDNA4-swizzled MX scale as garbage,
#      so the weight scale is kept unswizzled and ``swizzle_mx_scale=None`` is
#      passed to the kernel.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_TDM_DISABLED = False

# None lets aiter pick its default backend (gluon on gfx1250).
_BACKEND = envs.SGLANG_GFX1250_W4A8_BACKEND.get().strip().lower() or None
_KERNEL_GATHER = envs.SGLANG_GFX1250_W4A8_KERNEL_GATHER.get()
if _BACKEND not in (None, "triton", "gluon"):
    raise ValueError(
        f"SGLANG_GFX1250_W4A8_BACKEND={_BACKEND!r}: expected triton, gluon or unset"
    )
if _KERNEL_GATHER and _BACKEND != "triton":
    raise ValueError(
        "SGLANG_GFX1250_W4A8_KERNEL_GATHER=1 needs SGLANG_GFX1250_W4A8_BACKEND=triton: "
        "the gluon kernel gathers with TDM, which does not compile on gfx1250"
    )

FLYDSL_ENABLED = envs.SGLANG_GFX1250_W4A8_FLYDSL.get()
FLYDSL_MIN_TOKENS = envs.SGLANG_GFX1250_W4A8_FLYDSL_MIN_TOKENS.get()
if FLYDSL_ENABLED and _KERNEL_GATHER:
    raise ValueError(
        "SGLANG_GFX1250_W4A8_FLYDSL=1 runs small batches on the gluon kernel over "
        "preshuffled weights: SGLANG_GFX1250_W4A8_KERNEL_GATHER must be off"
    )
# FlyDSL's a8w4 MoE wants hidden and intermediate dims in multiples of 256
# (GPT-OSS 2880 -> 3072; TP2 intermediate 1440 -> 1536).
_FLYDSL_DIM_ALIGN = 256
# The gluon kernel's GFX1250_SCALE (n32k4) scales hold K in chunks of 4 e8m0 groups
# (128 elements), and its scale descriptor ends at cdiv(K, 32) groups: with
# K = 2880 the last chunk's columns fall outside it and read as 0 (scale 2^-127).
# So the gluon path runs on K rounded up to 128 (2880 -> 2944, zero activations
# against zero weights), and GEMM 1 on N = 2 * that, so GEMM 2's K follows.
_GLUON_K_ALIGN = 128
_E8M0_ONE = 127
_FUSED_GATHER_QUANT = envs.SGLANG_GFX1250_W4A8_FUSED_GATHER_QUANT.get()


@triton.jit
def _gather_pad_static_fp8_kernel(
    out_ptr,
    x_ptr,
    gather_idx_ptr,
    scale_ptr,
    topk,
    cols,
    width,
    stride_x,
    stride_out,
    BLOCK_N: tl.constexpr,
):
    # Row r of the output is input row gather_idx[r] // topk times 1 / scale, as
    # downcast_to_static_fp8 computes it, and zeros from cols to the padded width.
    r = tl.program_id(0).to(tl.int64)
    # gather_idx may be uint16 (aiter routing picks the narrowest index type)
    src = (tl.load(gather_idx_ptr + r).to(tl.int32) // topk).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + src * stride_x + offs, mask=offs < cols, other=0.0)
    scale = tl.load(scale_ptr)
    q = x * (1 / scale)
    tl.store(
        out_ptr + r * stride_out + offs,
        q.to(out_ptr.dtype.element_ty),
        mask=offs < width,
    )


def _gather_pad_static_fp8(
    x: torch.Tensor,
    gather_idx: torch.Tensor,
    topk: int,
    scale: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """fp8 rows x[gather_idx // topk] zero-padded to width, in one kernel: the same values as
    downcast_to_static_fp8(pad(x, width)[gather_idx // topk], scale)."""
    assert x.stride(-1) == 1 and scale.numel() == 1
    out = torch.empty(
        (gather_idx.shape[0], width), dtype=torch.float8_e4m3fn, device=x.device
    )
    if out.shape[0]:
        _gather_pad_static_fp8_kernel[(out.shape[0],)](
            out,
            x,
            gather_idx,
            scale,
            topk,
            x.shape[-1],
            width,
            x.stride(0),
            out.stride(0),
            BLOCK_N=triton.next_power_of_2(width),
        )
    return out


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
        from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8
    except ImportError:
        return None

    global _TDM_DISABLED
    if not _TDM_DISABLED:
        # gfx1250: the in-kernel TDM gather emitted by the routing sort / GEMM
        # fails to compile (``TDM gather dst must be 2D``). Force the non-TDM
        # path; we gather activations manually below.
        _routing_mod.is_tdm_avail = lambda: False
        _TDM_DISABLED = True

    return (
        _routing_mod,
        moe_gemm_a8w4,
        downcast_to_static_fp8,
    )


def build_ep_expert_map(
    num_experts: int, num_local_experts: int, ep_rank: int, device
) -> torch.Tensor:
    """Global expert id -> local expert id (-1 for experts owned by another
    rank), in the form ``ep_sort_routing`` expects: one entry longer than the
    global expert count."""
    expert_map = torch.full((num_experts + 1,), -1, dtype=torch.int32, device=device)
    start = ep_rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )
    return expert_map


def _ep_routing(
    routing_mod,
    router_logits: torch.Tensor,
    topk: int,
    renormalize: bool,
    expert_map: torch.Tensor,
    num_local_experts: int,
):
    """Routing for one expert-parallel rank from the GLOBAL router logits.

    Every rank sees all tokens (standard dispatcher); gates routed to experts
    owned by other ranks land in a sentinel bin that no GEMM block is scheduled
    for, and are excluded from the GEMM2 reduction via ``gate_valid``. The
    partial outputs are summed across ranks by the layer's all-reduce.
    """
    num_tokens, n_expts_tot = router_logits.shape
    n_gates = num_tokens * topk
    device = router_logits.device

    if renormalize:
        vals, topk_ids = torch.topk(router_logits, topk, dim=-1)
        topk_weights = torch.softmax(vals.float(), dim=-1)
    else:
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    # Same block_m heuristic as ``routing()``, keyed off the GLOBAL expert count.
    tokens_per_expt = max(1, n_gates // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data_bufs = routing_mod._compute_expt_data_internal(
        num_local_experts, n_gates, block_m, device
    )
    token_offs_raw, token_offs_pad, block_pid_map = expt_data_bufs[:3]
    num_local_tokens = torch.full((1,), num_tokens, dtype=torch.int32, device=device)

    hist_full, gather_idx, scatter_idx, gate_scal, gate_valid, _ = (
        routing_mod.ep_sort_routing(
            topk_weights,
            topk_ids,
            expert_map,
            num_local_experts,
            num_local_tokens,
            num_tokens,
            topk,
            n_gates,
            expt_data_bufs,
        )
    )
    hist = hist_full[:num_local_experts]
    expt_data = routing_mod.ExptData(
        hist, token_offs_raw, token_offs_pad, block_pid_map
    )
    routing_data = routing_mod.RoutingData(
        block_m, gate_scal, hist, num_local_experts, topk, expt_data
    )
    return routing_data, gather_idx, scatter_idx, gate_valid


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
    expert_map: Optional[torch.Tensor] = None,
    num_local_experts: Optional[int] = None,
    backend: Optional[str] = _BACKEND,
    preshuffled: bool = False,
    swizzle_mx_scale: Optional[str] = None,
    pad_hidden_to: Optional[int] = None,
) -> torch.Tensor:
    """MXFP4 W4A8 GPT-OSS MoE forward for gfx1250 via aiter triton
    ``moe_gemm_a8w4``.

    ``w*`` / ``w*_scale`` / ``w*_bias`` must already be in the
    ``moe_gemm_a8w4`` layout produced by :func:`prepare_w4a8_gfx1250_weights`,
    or (``preshuffled=True``, gluon, ``swizzle_mx_scale="GFX1250_SCALE"``) the
    views of FlyDSL's weights made by :func:`prepare_w4a8_gfx1250_flydsl_weights`.
    ``a13_scale`` / ``a2_scale`` are the static per-tensor FP8 activation scales
    for gate_up_proj and down_proj respectively.
    """
    imported = _import_aiter_w4a8()
    if imported is None:
        raise RuntimeError(
            "aiter triton W4A8 MoE (moe_gemm_a8w4) is required for the gfx1250 "
            "GPT-OSS MXFP4 path but was not found in the installed aiter build."
        )
    routing_mod, moe_gemm_a8w4, downcast_to_static_fp8 = imported

    assert hidden_states.dtype == torch.bfloat16

    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty(hidden_states.shape)
    fused_gather_quant = (
        _FUSED_GATHER_QUANT
        and not _KERNEL_GATHER
        and expert_map is None
        and not apply_router_weight_on_input
        and hidden_states.stride(-1) == 1
    )
    if fused_gather_quant:
        width = max(pad_hidden_to or 0, hidden_states.shape[-1])
    elif pad_hidden_to and pad_hidden_to > hidden_states.shape[-1]:
        hidden_states = torch.nn.functional.pad(
            hidden_states, (0, pad_hidden_to - hidden_states.shape[-1])
        )

    if expert_map is not None:
        routing_data, gather_idx, scatter_idx, gate_valid = _ep_routing(
            routing_mod,
            router_logits,
            topk,
            renormalize,
            expert_map,
            num_local_experts,
        )
    else:
        # aiter routing on the raw router logits. renormalize=True (GPT-OSS)
        # corresponds to applying softmax to the top-k selection inside the
        # kernel (sm_first=False).
        routing_data, gather_idx, scatter_idx = routing_mod.routing(
            router_logits, topk, sm_first=not renormalize
        )
        gate_valid = None
    gammas = routing_data.gate_scal

    if fused_gather_quant:
        x_fp8 = _gather_pad_static_fp8(
            hidden_states, gather_idx, topk, a13_scale, width
        )
        gemm1_gather = None
    elif (
        _KERNEL_GATHER
        and not preshuffled
        and expert_map is None
        and not apply_router_weight_on_input
    ):
        # Downcast the num_tokens rows, and let GEMM 1 gather them: cheaper than
        # gathering then downcasting num_tokens * topk bf16 rows.
        x_fp8 = downcast_to_static_fp8(hidden_states, a13_scale)
        gemm1_gather = gather_idx
    else:
        # gfx1250: the gluon in-kernel gather is broken, so we pass
        # gather_indx=None to moe_gemm_a8w4 and perform the gather ourselves.
        gather_src = gather_idx.to(torch.long) // topk
        if expert_map is not None:
            # Rows past the local experts (sentinel bin) are left unwritten by
            # ep_sort_routing and never consumed by the GEMMs.
            gather_src = gather_src.clamp_(0, hidden_states.shape[0] - 1)
        x = hidden_states[gather_src]
        if apply_router_weight_on_input:
            # Router weights must be applied in bf16 before quantization.
            x = x * gammas[:, None].to(x.dtype)
        x_fp8 = downcast_to_static_fp8(x, a13_scale)
        gemm1_gather = None

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
        gather_indx=gemm1_gather,
        scatter_indx=None,
        gammas=None,
        swizzle_mx_scale=swizzle_mx_scale,
        out_dtype=x_fp8.dtype,
        apply_swiglu=True,
        alpha=gemm1_alpha,
        limit=gemm1_limit,
        preshuffled=preshuffled,
        backend=backend,
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
        swizzle_mx_scale=swizzle_mx_scale,
        out_dtype=torch.bfloat16,
        gate_valid=gate_valid,
        preshuffled=preshuffled,
        backend=backend,
    )

    return intermediate_cache3.contiguous()


@dataclass
class FlydslW4A8Weights:
    """GPT-OSS W4A8 MoE weights in FlyDSL's layout, plus zero-copy gluon views.

    FlyDSL (``aiter.fused_moe``, ``gate_mode=INTERLEAVE``): gate/up rows
    interleaved, dims zero-padded to 256, ``moe_shuffle_weight`` /
    ``moe_shuffle_scale`` (n32k4) layouts, padded scales = e8m0 1.0.
    Gluon (``moe_gemm_a8w4``, ``preshuffled=True``, ``GFX1250_SCALE``): the same
    storage through ``moe_weight_decode_view``, cut to dims rounded up to 128
    (``_GLUON_K_ALIGN``): GEMM 1 to N = 2 * I_g, GEMM 2 to the real hidden N;
    the kernel takes K from the activations (hidden padded to ``gluon_hidden``,
    the intermediate to I_g by GEMM 1's zero rows).
    """

    hidden_size: int
    hidden_size_padded: int
    gluon_hidden: int
    w13: torch.Tensor  # [E, 2*I_pad, H_pad // 2] uint8, shuffled
    w13_scale: torch.Tensor  # [E, 2*I_pad // 32, H_pad // 32 * 32] uint8 (n32k4)
    w13_bias: torch.Tensor  # [E, 2*I_pad] fp32, gate/up interleaved
    w2: torch.Tensor  # [E, H_pad, I_pad // 2] uint8, shuffled
    w2_scale: torch.Tensor  # [E, H_pad // 32, I_pad // 32 * 32] uint8 (n32k4)
    w2_bias: torch.Tensor  # [E, H_pad] fp32
    gluon_w13: torch.Tensor
    gluon_w13_scale: torch.Tensor
    gluon_w13_bias: torch.Tensor
    gluon_w2: torch.Tensor
    gluon_w2_scale: torch.Tensor
    gluon_w2_bias: torch.Tensor


def _pad_dim(t: torch.Tensor, dim: int, size: int, value) -> torch.Tensor:
    if t.shape[dim] == size:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = size - t.shape[dim]
    return torch.cat([t, t.new_full(pad_shape, value)], dim=dim)


_weight_pool = None


def _to_weight_pool(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """Copies of ``tensors`` in a memory pool that holds nothing but them."""
    global _weight_pool
    if _weight_pool is None:
        _weight_pool = torch.cuda.MemPool()
    copies = []
    with torch.cuda.use_mem_pool(_weight_pool):
        for t in tensors:
            copy = t.clone()
            if getattr(t, "is_shuffled", False):
                copy.is_shuffled = True  # aiter.fused_moe reads it
            copies.append(copy)
    return copies


def prepare_w4a8_gfx1250_flydsl_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_bias: torch.Tensor,
) -> FlydslW4A8Weights:
    """SGLang's loaded Quark W4A8 MoE buffers (layout as in
    :func:`prepare_w4a8_gfx1250_weights`, gate/up SEPARATED) -> one copy in
    FlyDSL's layout, with the gluon views over it."""
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
    from aiter.ops.triton.utils.shuffle import moe_weight_decode_view

    num_experts, hidden, half_inter = w2_weight.shape
    inter = half_inter * 2
    hidden_pad = -(-hidden // _FLYDSL_DIM_ALIGN) * _FLYDSL_DIM_ALIGN
    inter_pad = -(-inter // _FLYDSL_DIM_ALIGN) * _FLYDSL_DIM_ALIGN
    inter_g = -(-inter // _GLUON_K_ALIGN) * _GLUON_K_ALIGN

    # GEMM1 [E, (gate|up), I, K]: pad I in each half and K, then interleave the
    # gate/up rows (moe_shuffle_* do it with is_guinterleave=True, gate_up=True).
    w13 = w13_weight.view(num_experts, 2, inter, hidden // 2)
    w13 = _pad_dim(_pad_dim(w13, 2, inter_pad, 0), 3, hidden_pad // 2, 0)
    w13 = moe_shuffle_weight(
        w13.reshape(num_experts, 2 * inter_pad, hidden_pad // 2),
        num_experts,
        is_guinterleave=True,
        gate_up=True,
    )
    w13_scale = w13_weight_scale.view(num_experts, 2, inter, hidden // 32)
    w13_scale = _pad_dim(
        _pad_dim(w13_scale, 2, inter_pad, _E8M0_ONE), 3, hidden_pad // 32, _E8M0_ONE
    )
    w13_scale = moe_shuffle_scale(
        w13_scale.reshape(num_experts, 2 * inter_pad, hidden_pad // 32),
        num_experts,
        is_guinterleave=True,
        gate_up=True,
    ).contiguous()
    w13_bias = _pad_dim(w13_weight_bias.view(num_experts, 2, inter), 2, inter_pad, 0)
    w13_bias = (
        w13_bias.transpose(1, 2)
        .reshape(num_experts, 2 * inter_pad)
        .float()
        .contiguous()
    )

    # GEMM2 [E, H, I]: pad N (hidden) and K (intermediate).
    w2 = _pad_dim(_pad_dim(w2_weight, 1, hidden_pad, 0), 2, inter_pad // 2, 0)
    w2 = moe_shuffle_weight(w2.contiguous(), num_experts)
    w2_scale = _pad_dim(
        _pad_dim(w2_weight_scale, 1, hidden_pad, _E8M0_ONE),
        2,
        inter_pad // 32,
        _E8M0_ONE,
    )
    w2_scale = moe_shuffle_scale(w2_scale.contiguous(), num_experts).contiguous()
    w2_bias = _pad_dim(w2_weight_bias, 1, hidden_pad, 0).float().contiguous()

    # The padded buffers are larger than the loaded ones, so the caching allocator
    # cannot reuse the loaded buffers' blocks for them and carves them (and the
    # small scales and biases) out of whatever the loaded buffers and the
    # temporaries above leave free. Once those are freed, every segment that
    # holds a piece of the result stays reserved: 13.6 GiB at TP1 over 36 layers,
    # counted as weight memory and taken from the KV pool. Copied into a pool of
    # their own, the results leave those segments wholly free for empty_cache.
    w13, w13_scale, w13_bias, w2, w2_scale, w2_bias = _to_weight_pool(
        w13, w13_scale, w13_bias, w2, w2_scale, w2_bias
    )

    # Gluon views: (E, K_pad * 16, N // 16) weights, (E, K_scale_pad * 32, N // 32)
    # scales (moe_gemm_a8w4 transposes them back), cut to the first N rows: GEMM 1
    # 2 * inter_g (gate/up interleaved, so real rows first, then zero rows), GEMM 2
    # the real hidden.
    return FlydslW4A8Weights(
        hidden_size=hidden,
        hidden_size_padded=hidden_pad,
        gluon_hidden=-(-hidden // _GLUON_K_ALIGN) * _GLUON_K_ALIGN,
        w13=w13,
        w13_scale=w13_scale,
        w13_bias=w13_bias,
        w2=w2,
        w2_scale=w2_scale,
        w2_bias=w2_bias,
        gluon_w13=moe_weight_decode_view(w13)[:, :, : 2 * inter_g // 16],
        gluon_w13_scale=w13_scale.view(num_experts, 2 * inter_pad // 32, -1).transpose(
            1, 2
        )[:, :, : 2 * inter_g // 32],
        gluon_w13_bias=w13_bias[:, : 2 * inter_g],
        gluon_w2=moe_weight_decode_view(w2)[:, :, : hidden // 16],
        gluon_w2_scale=w2_scale.view(num_experts, hidden_pad // 32, -1).transpose(1, 2)[
            :, :, : hidden // 32
        ],
        gluon_w2_bias=w2_bias[:, :hidden],
    )


def aiter_w4a8_gfx1250_flydsl_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk: int,
    weights: FlydslW4A8Weights,
    gemm1_limit: float,
    renormalize: bool = True,
) -> torch.Tensor:
    """GPT-OSS W4A8 MoE through aiter.fused_moe's FlyDSL a8w4 (gfx1250 grouped
    GEMM; MXFP8 activations quantized per 1x32 by FlyDSL itself). Routing as
    the triton path's: top-k of the logits, softmax over them (renormalize)."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import GateMode, fused_moe

    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty(hidden_states.shape)
    if renormalize:
        vals, topk_ids = torch.topk(router_logits, topk, dim=-1)
        topk_weights = torch.softmax(vals.float(), dim=-1)
    else:
        probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    x = hidden_states
    pad = weights.hidden_size_padded - weights.hidden_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    out = fused_moe(
        x,
        weights.w13,
        weights.w2,
        topk_weights.to(torch.bfloat16),
        topk_ids.to(torch.int32),
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=weights.w13_scale,
        w2_scale=weights.w2_scale,
        bias1=weights.w13_bias,
        bias2=weights.w2_bias,
        swiglu_limit=gemm1_limit,
        gate_mode=GateMode.INTERLEAVE.value,
        quant_dtype_a=dtypes.fp8,
    )
    if pad:
        out = out[:, : weights.hidden_size]
    return out.contiguous()
