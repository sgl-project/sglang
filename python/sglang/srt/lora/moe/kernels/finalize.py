"""The shared-outer finalize pair: a rank-space reduction of the weighted
bridge rows per token, then a tail that adds the one down-B product to the
weighted base rows and scales once. Both row domains work, since the tail
reads base rows only through ``pair_to_row``.

The one-pass path keeps the rank sum and delta in FP32; the staged path rounds
them to buffer dtypes. Neither is bitwise equivalent to per-pair BF16 deltas.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.route_view import RouteView

SHARED_RANK_DEFAULT_CONFIG: dict[str, dict[str, int]] = {
    "reduce": {
        "BLOCK_SIZE_T": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
    "tail": {
        "BLOCK_SIZE_H": 128,
        "BLOCK_SIZE_K": 32,
        "num_warps": 4,
        "num_stages": 3,
    },
}


@triton.jit
def _shared_rank_reduce_kernel(
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
def _shared_from_scratch_finalize_kernel(
    down_ptr,
    pair_to_row_ptr,
    token_rank_ptr,
    b_ptr,
    output_ptr,
    weights_ptr,
    topk_ids_ptr,
    token_lora_mapping_ptr,
    stride_dm,
    stride_dh,
    stride_tm,
    stride_tk,
    stride_bg,
    stride_bh,
    stride_bk,
    stride_om,
    stride_oh,
    stride_wm,
    stride_wk,
    routed_scaling,
    num_local_experts: tl.constexpr,
    hidden: tl.constexpr,
    rank: tl.constexpr,
    top_k: tl.constexpr,
    max_loras: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
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
    safe_adapter = tl.maximum(adapter, 0).to(tl.int64)
    delta = tl.zeros((block_h,), tl.float32)
    for k_begin in range(0, rank, block_k):
        rank_offsets = k_begin + tl.arange(0, block_k).to(tl.int64)
        rank_mask = rank_offsets < rank
        x = tl.load(
            token_rank_ptr + token64 * stride_tm + rank_offsets * stride_tk,
            mask=adapter_valid & rank_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + safe_adapter * stride_bg
            + h_offsets[:, None] * stride_bh
            + rank_offsets[None, :] * stride_bk,
            mask=adapter_valid & h_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        delta += tl.sum(b.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

    tl.store(
        output_ptr + token64 * stride_om + h_offsets * stride_oh,
        (routed_scaling * (base_acc + tl.where(adapter_valid, delta, 0.0))).to(
            output_ptr.dtype.element_ty
        ),
        mask=h_mask,
    )


def invoke_shared_rank_reduce(
    *,
    bridge: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    token_rank: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    # This implementation applies the scaling in the tail.
    del routed_scaling_factor
    num_tokens, top_k = routing.topk_ids.shape
    if num_tokens == 0:
        return
    # The rank is adapter-file input, so it stays checked here.
    rank = bridge.shape[1]
    block_r = max(16, triton.next_power_of_2(rank))
    block_t = int(config["BLOCK_SIZE_T"])
    _shared_rank_reduce_kernel[(triton.cdiv(num_tokens, block_t),)](
        bridge,
        token_rank,
        topk_weights,
        routing.topk_ids,
        routing.token_lora_mapping,
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
        local_expert_count=routing.num_local_experts,
        block_t=block_t,
        block_r=block_r,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def invoke_shared_from_scratch_finalize(
    *,
    down_rows: torch.Tensor,
    pair_to_row: torch.Tensor,
    token_rank: torch.Tensor,
    b_down: torch.Tensor,
    routing: RouteView,
    topk_weights: torch.Tensor,
    routed_scaling_factor: float | None,
    output: torch.Tensor,
    config: Mapping[str, int],
) -> None:
    num_tokens = routing.topk_ids.shape[0]
    if num_tokens == 0:
        return
    hidden = down_rows.shape[-1]
    rank = token_rank.shape[1]
    block_h = int(config["BLOCK_SIZE_H"])
    _shared_from_scratch_finalize_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_rows.view(-1, hidden),
        pair_to_row,
        token_rank,
        b_down,
        output,
        topk_weights,
        routing.topk_ids,
        routing.token_lora_mapping,
        down_rows.stride(-2),
        down_rows.stride(-1),
        token_rank.stride(0),
        token_rank.stride(1),
        b_down.stride(0),
        b_down.stride(1),
        b_down.stride(2),
        output.stride(0),
        output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
        num_local_experts=routing.num_local_experts,
        hidden=hidden,
        rank=rank,
        top_k=routing.topk_ids.shape[1],
        max_loras=routing.max_loras,
        block_h=block_h,
        block_k=int(config["BLOCK_SIZE_K"]),
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
