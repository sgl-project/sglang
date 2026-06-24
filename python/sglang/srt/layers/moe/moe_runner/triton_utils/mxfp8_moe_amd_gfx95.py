"""Native MXFP8 (1x32 block, E8M0 scale) MoE for AMD CDNA4 (gfx950).

Replaces the prior SGLang MXFP8 MoE family (dense / hybrid / packed /
grouped_gemm1 / grouped_gemm12 / compact / fused_act) with a single grouped
``tl.dot_scaled`` kernel. Instead of an explicit ``argsort`` + ``index_select``
gather, a materialized intermediate, a separate activation quant, and a
``tl.atomic_add`` combine:

  * tokens are sorted by expert with ``moe_align_block_size``;
  * GEMM1 reads the activation by token-id indirection (``a_row = token // top_k``)
    so the hidden states are MXFP8-quantized exactly ONCE (not top_k times);
  * the SwiGLU-OAI activation is the fused fp32 Triton kernel (split layout);
  * GEMM2 applies the top-k weight inside the kernel and writes each route to a
    distinct output row (no atomics); the final reduction is a strided sum.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.layers.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize


@triton.jit
def _mxfp8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    E,
    N,
    K,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    valid_expert = (off_e >= 0) & (off_e < E)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=valid_expert & n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=valid_expert & n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _grouped_gemm_mxfp8(
    a_q: torch.Tensor,  # [M, K] fp8 e4m3
    a_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,  # [E, N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [E, N, K//32] uint8 (E8M0)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    out_dtype: torch.dtype,
    a_div: int,
    mul_weight_by: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M_routed = num_valid_tokens
    E, N, K = w.shape
    assert K % 128 == 0, f"MXFP8 native MoE requires K%128==0, got K={K}"
    # Keep zero-fill: moe_align_block_size reserves an extra expert bucket for
    # filtered routes, which should contribute zeros if present.
    out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
    if a_div == top_k and M_routed <= 32 and K >= 3072:
        BLOCK_N = 64
        num_warps = 4
    else:
        BLOCK_N = 128
        num_warps = 8
    BLOCK_K = 128
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m), triton.cdiv(N, BLOCK_N))
    _mxfp8_grouped_gemm_kernel[grid](
        a_q,
        a_scale,
        w,
        w_scale,
        out,
        mul_weight_by if mul_weight_by is not None else a_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        E,
        N,
        K,
        num_valid_tokens,
        top_k,
        a_q.stride(0),
        a_q.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        out.stride(0),
        out.stride(1),
        A_DIV=a_div,
        MUL_WEIGHT=mul_weight_by is not None,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _combine_topk_routes_kernel(
    route_ptr,
    out_ptr,
    H,
    top_k: tl.constexpr,
    stride_rm,
    stride_rh,
    stride_ot,
    stride_oh,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_block = tl.program_id(1)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for route_idx in range(0, top_k):
        route_row = token_id * top_k + route_idx
        vals = tl.load(
            route_ptr + route_row * stride_rm + offs_h * stride_rh,
            mask=h_mask,
            other=0.0,
        )
        acc += vals.to(tl.float32)

    tl.store(out_ptr + token_id * stride_ot + offs_h * stride_oh, acc, mask=h_mask)


def _combine_topk_routes(
    route_outputs: torch.Tensor,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty(
        (num_tokens, hidden_size), dtype=out_dtype, device=route_outputs.device
    )
    block_h = 1024
    grid = (num_tokens, triton.cdiv(hidden_size, block_h))
    _combine_topk_routes_kernel[grid](
        route_outputs,
        out,
        hidden_size,
        top_k,
        route_outputs.stride(0),
        route_outputs.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=8,
    )
    return out


def fused_moe_mxfp8_native(
    hidden_states: torch.Tensor,  # [T, H] bf16
    w13: torch.Tensor,  # [E, 2I, H] fp8
    w13_scale: torch.Tensor,  # [E, 2I, H//32] uint8
    w2: torch.Tensor,  # [E, H, I] fp8
    w2_scale: torch.Tensor,  # [E, H, I//32] uint8
    topk_weights: torch.Tensor,  # [T, top_k]
    topk_ids: torch.Tensor,  # [T, top_k] (local expert ids; -1 for non-local EP)
    *,
    alpha: float,
    beta: float,
    limit: Optional[float],
    no_combine: bool = False,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Lazy import: the jit_kernel package pulls in Triton at first use; importing
    # at call time avoids any import-time cycle with the moe runner package.
    from sglang.jit_kernel.minimax_m3 import swiglu_oai_mxfp8_quant, swiglu_oai_split

    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k
    local_num_experts = w13.shape[0]

    if expert_map is not None:
        valid_global = (topk_ids >= 0) & (topk_ids < expert_map.numel())
        topk_ids = expert_map[topk_ids.clamp(0, expert_map.numel() - 1).long()].to(
            torch.int32
        )
        topk_ids.masked_fill_(
            ~valid_global | (topk_ids < 0) | (topk_ids >= local_num_experts), -1
        )
    else:
        topk_ids = topk_ids.to(torch.int32, copy=True)
        topk_ids.masked_fill_((topk_ids < 0) | (topk_ids >= local_num_experts), -1)

    block_m = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_m, local_num_experts
    )

    # GEMM1: x (mxfp8) @ w13^T -> [M, 2I]. The activation is quantized ONCE over
    # the T hidden rows; the kernel gathers per route via a_row = token // top_k.
    a_q, a_s = mxfp8_e4m3_quantize(hidden_states)
    g1 = _grouped_gemm_mxfp8(
        a_q,
        a_s,
        w13,
        w13_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        top_k,
        block_m,
        hidden_states.dtype,
        a_div=top_k,
    )  # [M, 2I]

    if envs.SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8.get():
        # SwiGLU-OAI (split layout) + MiniMax MXFP8 quant in one kernel, fp32 all
        # the way to the E8M0 scale (no bf16 activation round-trip; matches the
        # vLLM/ame fused kernel). Opt-in until full-model accuracy is re-qualified.
        act_q, act_s = swiglu_oai_mxfp8_quant(g1, alpha=alpha, beta=beta, limit=limit)
    else:
        # Default accuracy path: keep the historical two-kernel boundary.
        act = swiglu_oai_split(
            g1, alpha=alpha, beta=beta, limit=limit, out_dtype=hidden_states.dtype
        )
        act_q, act_s = mxfp8_e4m3_quantize(act)

    if no_combine:
        # Per-route outputs, unweighted, no reduction: [T, top_k, H].
        g2 = _grouped_gemm_mxfp8(
            act_q,
            act_s,
            w2,
            w2_scale,
            sorted_ids,
            expert_ids,
            num_post,
            M,
            top_k,
            block_m,
            hidden_states.dtype,
            a_div=1,
        )
        return g2.view(T, top_k, H)

    # GEMM2: act (mxfp8) @ w2^T -> [M, H], weighted by topk_weights, then reduce.
    g2 = _grouped_gemm_mxfp8(
        act_q,
        act_s,
        w2,
        w2_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        top_k,
        block_m,
        torch.float32,
        a_div=1,
        mul_weight_by=topk_weights.reshape(-1).to(torch.float32),
    )  # [M, H] == [T*top_k, H]

    if envs.SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE.get():
        return _combine_topk_routes(g2, T, top_k, H, hidden_states.dtype)
    return g2.view(T, top_k, H).sum(dim=1).to(hidden_states.dtype)


def fused_experts_mxfp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    no_combine: bool = False,
    inplace: bool = False,
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    swiglu_limit: Optional[float] = None,
    gate_up_interleaved: bool = True,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Native MXFP8 MoE entry (CDNA4 ``dot_scaled``).

    Keeps the SGLang ``fused_experts_mxfp8`` call contract but routes to the
    single grouped-GEMM kernel. Only the MiniMax-M3 SwiGLU-OAI (split,
    uninterleaved, gated silu with ``gemm1_alpha``/``gemm1_limit``) configuration
    is supported -- the unsupported cases below never occur on the M3 path.
    """
    if not (activation == "silu" and is_gated):
        raise NotImplementedError(
            f"native MXFP8 MoE only supports gated swiglu-oai, got "
            f"{activation=} {is_gated=}."
        )
    if b1 is not None or b2 is not None:
        raise NotImplementedError("native MXFP8 MoE does not support expert bias.")
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "native MXFP8 MoE does not support apply_router_weight_on_input."
        )
    if gate_up_interleaved:
        raise NotImplementedError(
            "native MXFP8 MoE expects uninterleaved (split) gate/up layout."
        )

    # SwiGLU-OAI default activation alpha (gpt-oss); M3 may override via gemm1_alpha.
    alpha = 1.702 if gemm1_alpha is None else float(gemm1_alpha)
    beta = 1.0
    limit = None if gemm1_limit is None else float(gemm1_limit)
    # NOTE: routed_scaling_factor is intentionally NOT re-applied here. For M3
    # (sigmoid routing with apply_routed_scaling_factor_on_output=True) it is
    # already folded into topk_weights by the topk kernel; re-applying would
    # double-count it. This matches the prior SGLang MXFP8 behaviour.

    out = fused_moe_mxfp8_native(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        alpha=alpha,
        beta=beta,
        limit=limit,
        no_combine=no_combine,
        expert_map=expert_map,
    )

    if no_combine:
        return out
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
