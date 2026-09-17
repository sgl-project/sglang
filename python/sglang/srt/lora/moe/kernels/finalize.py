"""Shared-outer down-B and weighted combine in token space.

The one-pass path keeps the rank sum and delta in FP32; the staged path rounds
them to buffer dtypes. Neither is bitwise equivalent to per-pair BF16 deltas.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.route_view import RouteView


@triton.jit
def _shared_token_delta_reduce_kernel(
    bridge_ptr,
    token_rank_ptr,
    weights_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    num_tokens,
    stride_xm,
    stride_xk,
    stride_tm,
    stride_tk,
    stride_wm,
    stride_wk,
    rank: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    local_expert_count: tl.constexpr,
    block_t: tl.constexpr,
    block_r: tl.constexpr,
):
    tokens = tl.program_id(0) * block_t + tl.arange(0, block_t)
    token_mask = tokens < num_tokens
    tokens64 = tokens.to(tl.int64)
    rank_offsets = tl.arange(0, block_r).to(tl.int64)
    rank_mask = rank_offsets < rank
    adapter = tl.load(token_lora_mapping_ptr + tokens, mask=token_mask, other=-1)
    adapter_valid = (adapter >= 0) & (adapter < max_loras)
    acc = tl.zeros((block_t, block_r), tl.float32)
    for k in range(top_k):
        pairs = tokens * top_k + k
        expert = tl.load(topk_ids_ptr + pairs, mask=token_mask, other=-1)
        valid = (
            token_mask & adapter_valid & (expert >= 0) & (expert < local_expert_count)
        )
        weight = tl.load(
            weights_ptr + tokens64 * stride_wm + k * stride_wk,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            bridge_ptr
            + pairs.to(tl.int64)[:, None] * stride_xm
            + rank_offsets[None, :] * stride_xk,
            mask=valid[:, None] & rank_mask[None, :],
            other=0.0,
        )
        acc += weight[:, None] * x.to(tl.float32)
    tl.store(
        token_rank_ptr
        + tokens64[:, None] * stride_tm
        + rank_offsets[None, :] * stride_tk,
        acc.to(token_rank_ptr.dtype.element_ty),
        mask=token_mask[:, None] & rank_mask[None, :],
    )


@triton.jit
def _shared_token_delta_tail_kernel(
    down_ptr,
    pair_to_row_ptr,
    token_delta_ptr,
    output_ptr,
    weights_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    stride_dm,
    stride_dh,
    stride_om,
    stride_oh,
    stride_wm,
    stride_wk,
    stride_tdm,
    stride_tdh,
    routed_scaling,
    num_local_experts: tl.constexpr,
    hidden: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    block_h: tl.constexpr,
):
    # Apply routed scaling once, after adding base and LoRA.
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    token64 = token.to(tl.int64)
    h_offsets = pid_h.to(tl.int64) * block_h + tl.arange(0, block_h).to(tl.int64)
    h_mask = h_offsets < hidden
    base_acc = tl.zeros((block_h,), tl.float32)
    for k in range(top_k):
        pair = token * top_k + k
        expert = tl.load(topk_ids_ptr + pair)
        valid = (expert >= 0) & (expert < num_local_experts)
        dst = tl.load(pair_to_row_ptr + pair, mask=valid, other=0).to(tl.int64)
        base = tl.load(
            down_ptr + dst * stride_dm + h_offsets * stride_dh,
            mask=valid & h_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weights_ptr + token64 * stride_wm + k * stride_wk,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        base_acc += weight * base

    adapter = tl.load(token_lora_mapping_ptr + token)
    adapter_valid = (adapter >= 0) & (adapter < max_loras)
    delta = tl.load(
        token_delta_ptr + token64 * stride_tdm + h_offsets * stride_tdh,
        mask=adapter_valid & h_mask,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        output_ptr + token64 * stride_om + h_offsets * stride_oh,
        (routed_scaling * (base_acc + tl.where(adapter_valid, delta, 0.0))).to(
            output_ptr.dtype.element_ty
        ),
        mask=h_mask,
    )


@triton.jit
def _shared_one_pass_kernel(
    down_ptr,  # [rows, H] provider row order, unweighted
    pair_to_row_ptr,  # [M * top_k] raw pair -> provider row
    bridge_ptr,  # [M * top_k, R] raw pair order, unweighted
    weights_ptr,
    b_ptr,  # [S, H, R]
    topk_ids_ptr,
    token_lora_mapping_ptr,
    output_ptr,
    stride_dm,
    stride_dh,
    stride_xm,
    stride_xr,
    stride_wm,
    stride_wk,
    stride_bs,
    stride_bh,
    stride_br,
    stride_om,
    stride_oh,
    routed_scaling,
    num_local_experts: tl.constexpr,
    hidden: tl.constexpr,
    rank: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    block_h: tl.constexpr,
    block_r: tl.constexpr,
):
    """Weight and combine unweighted base and rank rows, then apply shared B."""
    token = tl.program_id(0)
    pid_h = tl.program_id(1)
    token64 = token.to(tl.int64)
    h_offsets = pid_h.to(tl.int64) * block_h + tl.arange(0, block_h).to(tl.int64)
    h_mask = h_offsets < hidden
    r_offsets = tl.arange(0, block_r).to(tl.int64)
    r_mask = r_offsets < rank
    base_acc = tl.zeros((block_h,), tl.float32)
    rank_acc = tl.zeros((block_r,), tl.float32)
    for k in tl.static_range(top_k):
        pair = token64 * top_k + k
        expert = tl.load(topk_ids_ptr + pair)
        valid = (expert >= 0) & (expert < num_local_experts)
        dst = tl.load(pair_to_row_ptr + pair, mask=valid, other=0).to(tl.int64)
        weight = tl.load(
            weights_ptr + token64 * stride_wm + k * stride_wk, mask=valid, other=0.0
        ).to(tl.float32)
        base = tl.load(
            down_ptr + dst * stride_dm + h_offsets * stride_dh,
            mask=valid & h_mask,
            other=0.0,
        ).to(tl.float32)
        rank_row = tl.load(
            bridge_ptr + pair * stride_xm + r_offsets * stride_xr,
            mask=valid & r_mask,
            other=0.0,
        ).to(tl.float32)
        base_acc += weight * base
        rank_acc += weight * rank_row

    adapter = tl.load(token_lora_mapping_ptr + token)
    adapter_valid = (adapter >= 0) & (adapter < max_loras)
    safe_adapter = tl.maximum(adapter, 0).to(tl.int64)
    b = tl.load(
        b_ptr
        + safe_adapter * stride_bs
        + h_offsets[:, None] * stride_bh
        + r_offsets[None, :] * stride_br,
        mask=adapter_valid & h_mask[:, None] & r_mask[None, :],
        other=0.0,
    )
    delta = tl.sum(b.to(tl.float32) * rank_acc[None, :], axis=1)
    tl.store(
        output_ptr + token64 * stride_om + h_offsets * stride_oh,
        (routed_scaling * (base_acc + tl.where(adapter_valid, delta, 0.0))).to(
            output_ptr.dtype.element_ty
        ),
        mask=h_mask,
    )


def invoke_shared_token_delta_reduce(
    *,
    num_local_experts: int,
    bridge: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    token_rank: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    num_tokens, top_k = routing.num_tokens, routing.width
    if num_tokens == 0:
        return
    rank = bridge.shape[1]
    block_t = int(config["BLOCK_SIZE_T"])
    _shared_token_delta_reduce_kernel[(triton.cdiv(num_tokens, block_t),)](
        bridge,
        token_rank,
        topk_weights,
        routing.group_ids,
        routing.token_slots,
        num_tokens,
        bridge.stride(0),
        bridge.stride(1),
        token_rank.stride(0),
        token_rank.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        rank=rank,
        top_k=top_k,
        max_loras=routing.max_loras,
        local_expert_count=num_local_experts,
        block_t=block_t,
        block_r=max(16, triton.next_power_of_2(rank)),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_shared_token_delta_tail(
    *,
    num_local_experts: int,
    down_rows: torch.Tensor,
    pair_to_row: torch.Tensor,
    token_delta: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    output: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    num_tokens = routing.num_tokens
    if num_tokens == 0:
        return
    hidden = down_rows.shape[-1]
    block_h = int(config["BLOCK_SIZE_H"])
    _shared_token_delta_tail_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_rows.view(-1, hidden),
        pair_to_row,
        token_delta,
        output,
        topk_weights,
        routing.group_ids,
        routing.token_slots,
        down_rows.stride(-2),
        down_rows.stride(-1),
        output.stride(0),
        output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        token_delta.stride(0),
        token_delta.stride(1),
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
        num_local_experts=num_local_experts,
        hidden=hidden,
        top_k=routing.width,
        max_loras=routing.max_loras,
        block_h=block_h,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_shared_one_pass(
    *,
    num_local_experts: int,
    down_rows: torch.Tensor,
    pair_to_row: torch.Tensor,
    bridge: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    output: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    num_tokens, top_k = routing.num_tokens, routing.width
    if num_tokens == 0:
        return
    hidden = down_rows.shape[-1]
    rank = bridge.shape[-1]
    block_h = int(config["BLOCK_SIZE_H"])
    _shared_one_pass_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_rows.view(-1, hidden),
        pair_to_row,
        bridge,
        topk_weights,
        b_down,
        routing.group_ids,
        routing.token_slots,
        output,
        down_rows.stride(-2),
        down_rows.stride(-1),
        bridge.stride(0),
        bridge.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        output.stride(0),
        output.stride(1),
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
        num_local_experts=num_local_experts,
        hidden=hidden,
        rank=rank,
        top_k=top_k,
        max_loras=routing.max_loras,
        block_h=block_h,
        block_r=max(16, triton.next_power_of_2(rank)),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


# Vectorize index/weight loads, retaining the shared finalizer's top-k sum order.
@triton.jit
def _small_finalize_kernel(
    down_ptr,  # [rows, H] base down rows (pair_to_row indexes them)
    delta_ptr,  # [T * TOPK, H] unweighted LoRA delta per pair (or down_ptr when absent)
    out_ptr,  # [T, H]
    pair_to_row_ptr,  # [T * TOPK]; < 0 = no row for this pair
    topk_weights_ptr,  # [T * TOPK] fp32
    hidden,
    scaling,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < hidden
    ks = tl.arange(0, TOPK_P2)
    kmask = ks < TOPK
    rows = tl.load(pair_to_row_ptr + t * TOPK + ks, mask=kmask, other=-1).to(tl.int64)
    weights = tl.load(topk_weights_ptr + t * TOPK + ks, mask=kmask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        pick = ks == k
        row = tl.sum(tl.where(pick, rows, 0), axis=0)
        weight = tl.sum(tl.where(pick, weights, 0.0), axis=0)
        live = row >= 0
        x = tl.load(down_ptr + row * hidden + cols, mask=cmask & live, other=0.0).to(
            tl.float32
        )
        if HAS_DELTA:
            x += tl.load(
                delta_ptr + (t * TOPK + k) * hidden + cols, mask=cmask & live, other=0.0
            ).to(tl.float32)
        acc += x * weight
    acc *= scaling
    tl.store(out_ptr + t * hidden + cols, acc.to(out_ptr.dtype.element_ty), mask=cmask)


def invoke_small_finalize(
    down_rows: torch.Tensor,
    output: torch.Tensor,
    pair_to_row: torch.Tensor,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float,
    *,
    lora_delta: torch.Tensor | None,
    block_h: int = 512,
) -> None:
    num_tokens, hidden = output.shape
    topk = topk_weights.shape[1]
    _small_finalize_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_rows,
        down_rows if lora_delta is None else lora_delta,
        output,
        pair_to_row,
        topk_weights,
        hidden,
        float(routed_scaling_factor),
        TOPK=topk,
        TOPK_P2=triton.next_power_of_2(topk),
        HAS_DELTA=lora_delta is not None,
        BLOCK_H=block_h,
        num_warps=4,
    )
