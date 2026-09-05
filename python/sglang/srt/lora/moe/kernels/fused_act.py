"""Fuse gate/up LoRA-B with activation in either base row layout.

The delta stays FP32 through activation, unlike a materialized BF16 delta.
The optional pair-major copy serves down-A kernels that cannot gather base rows.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.kernels.activation_delta import (
    apply_activation,
)
from sglang.srt.lora.moe.kernels.routing import grouped_tile_coords
from sglang.srt.lora.moe.route_view import RouteView

FUSED_B_ACT_DEFAULT_CONFIG: dict[str, int] = {
    "BLOCK_SIZE_W": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}


@triton.jit
def _base_columns(
    offsets,
    width: tl.constexpr,
    num_slices: tl.constexpr,
    gate_first: tl.constexpr,
    interleaved: tl.constexpr,
):
    if num_slices == 1:
        return offsets, offsets
    if interleaved:
        gate_offsets = 2 * offsets
        up_offsets = 2 * offsets + 1
    else:
        gate_offsets = offsets
        up_offsets = width + offsets
    if not gate_first:
        gate_offsets, up_offsets = up_offsets, gate_offsets
    return gate_offsets, up_offsets


@triton.jit
def _delta_slice(
    bridge_ptr,
    weight_group_ptr,
    bridge_rows,
    pair_mask,
    w_offsets,
    w_mask,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    slice_id: tl.constexpr,
    rank: tl.constexpr,
    width: tl.constexpr,
    block_m: tl.constexpr,
    block_w: tl.constexpr,
    block_k: tl.constexpr,
):
    acc = tl.zeros((block_m, block_w), tl.float32)
    for k_begin in range(0, rank, block_k):
        k_offsets = k_begin + tl.arange(0, block_k).to(tl.int64)
        k_mask = k_offsets < rank
        lhs = tl.load(
            bridge_ptr
            + bridge_rows[:, None] * stride_xm
            + (slice_id * rank + k_offsets)[None, :] * stride_xk,
            mask=pair_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_group_ptr
            + (slice_id * width + w_offsets)[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=k_mask[:, None] & w_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(lhs, rhs, out_dtype=tl.float32)
    return acc


@triton.jit
def _b_act_kernel(
    bridge_ptr,
    b_ptr,
    base_ptr,
    act_rows_ptr,
    act_pairs_ptr,
    pair_to_row_ptr,
    topk_ids_ptr,
    sorted_pairs_ptr,
    block_veids_ptr,
    pairs_post_padded_ptr,
    num_pairs,
    stride_xm,
    stride_xk,
    stride_bg,
    stride_bn,
    stride_bk,
    stride_pm,
    stride_pn,
    stride_am,
    stride_an,
    stride_qm,
    stride_qn,
    num_local_experts: tl.constexpr,
    top_k: tl.constexpr,
    width: tl.constexpr,
    rank: tl.constexpr,
    num_slices: tl.constexpr,
    activation_type: tl.constexpr,
    gate_first: tl.constexpr,
    interleaved: tl.constexpr,
    bridge_token_major: tl.constexpr,
    num_m_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_w: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    store_pair_act: tl.constexpr,
):
    pid = tl.program_id(0)
    # Keep the tile count constexpr for grouped scheduling.
    num_w_tiles: tl.constexpr = (width + block_w - 1) // block_w
    pid_m, pid_w = grouped_tile_coords(pid, num_w_tiles, num_m_blocks, group_m)
    if pid_m * block_m >= tl.load(pairs_post_padded_ptr):
        return

    slots = pid_m * block_m + tl.arange(0, block_m).to(tl.int64)
    pair_ids = tl.load(sorted_pairs_ptr + slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    expert = tl.load(topk_ids_ptr + pair_ids, mask=pair_mask, other=-1)
    base_valid = pair_mask & (expert >= 0) & (expert < num_local_experts)
    dst_rows = tl.load(pair_to_row_ptr + pair_ids, mask=base_valid, other=0).to(
        tl.int64
    )
    veid = tl.load(block_veids_ptr + pid_m).to(tl.int64)

    w_offsets = pid_w * block_w + tl.arange(0, block_w).to(tl.int64)
    w_mask = w_offsets < width
    gate_cols, up_cols = _base_columns(
        w_offsets,
        width,
        num_slices,
        gate_first,
        interleaved,
    )

    delta_gate = tl.zeros((block_m, block_w), tl.float32)
    delta_up = tl.zeros((block_m, block_w), tl.float32)
    if veid != -1:
        bridge_rows = pair_ids // top_k if bridge_token_major else pair_ids
        group_ptr = b_ptr + veid * stride_bg
        delta_gate += _delta_slice(
            bridge_ptr,
            group_ptr,
            bridge_rows,
            pair_mask,
            w_offsets,
            w_mask,
            stride_xm,
            stride_xk,
            stride_bn,
            stride_bk,
            slice_id=0,
            rank=rank,
            width=width,
            block_m=block_m,
            block_w=block_w,
            block_k=block_k,
        )
        if num_slices == 2:
            delta_up += _delta_slice(
                bridge_ptr,
                group_ptr,
                bridge_rows,
                pair_mask,
                w_offsets,
                w_mask,
                stride_xm,
                stride_xk,
                stride_bn,
                stride_bk,
                slice_id=1,
                rank=rank,
                width=width,
                block_m=block_m,
                block_w=block_w,
                block_k=block_k,
            )

    base_gate = tl.load(
        base_ptr + dst_rows[:, None] * stride_pm + gate_cols[None, :] * stride_pn,
        mask=base_valid[:, None] & w_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    act = apply_activation(base_gate + delta_gate, activation_type)
    if num_slices == 2:
        base_up = tl.load(
            base_ptr + dst_rows[:, None] * stride_pm + up_cols[None, :] * stride_pn,
            mask=base_valid[:, None] & w_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        act = act * (base_up + delta_up)
    value = act.to(act_rows_ptr.dtype.element_ty)
    tl.store(
        act_rows_ptr + dst_rows[:, None] * stride_am + w_offsets[None, :] * stride_an,
        value,
        mask=base_valid[:, None] & w_mask[None, :],
    )
    if store_pair_act:
        # Invalid pairs must overwrite stale graph-buffer rows with zero.
        tl.store(
            act_pairs_ptr
            + pair_ids[:, None] * stride_qm
            + w_offsets[None, :] * stride_qn,
            tl.where(base_valid[:, None], act, 0.0).to(act_pairs_ptr.dtype.element_ty),
            mask=pair_mask[:, None] & w_mask[None, :],
        )


def _launch_b_act(
    *,
    activation: str,
    base_rows: torch.Tensor,  # [rows, slices * width] bf16, flat
    act_rows: torch.Tensor,  # [rows, width] bf16, flat
    act_pairs: torch.Tensor | None,  # [num_tokens, top_k, width] or None
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor,
    b_gate_up: torch.Tensor,
    bridge_top_k: int,
) -> None:
    ActivationFn.parse(activation)
    pairs = routing.topk_ids.numel()
    if pairs == 0:
        return
    width = act_rows.shape[1]
    slices = base_rows.shape[1] // width
    rank = b_gate_up.shape[2]
    block_w = int(config["BLOCK_SIZE_W"])
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    pair_target = act_pairs.view(-1, width) if act_pairs is not None else act_rows
    num_w_tiles = triton.cdiv(width, block_w)
    _b_act_kernel[(num_m_blocks * num_w_tiles,)](
        bridge_gateup,
        b_gate_up,
        base_rows,
        act_rows,
        pair_target,
        pair_to_row,
        routing.topk_ids,
        routing.sorted_pair_ids,
        routing.block_virtual_expert_ids,
        routing.num_pairs_post_padded,
        pairs,
        bridge_gateup.stride(0),
        bridge_gateup.stride(1),
        b_gate_up.stride(0),
        b_gate_up.stride(1),
        b_gate_up.stride(2),
        base_rows.stride(0),
        base_rows.stride(1),
        act_rows.stride(0),
        act_rows.stride(1),
        pair_target.stride(0),
        pair_target.stride(1),
        num_local_experts=num_local_experts,
        top_k=routing.topk_ids.shape[1],
        width=width,
        rank=rank,
        num_slices=slices,
        activation_type=activation,
        gate_first=gate_first,
        interleaved=interleaved,
        bridge_token_major=bridge_top_k != 1,
        num_m_blocks=num_m_blocks,
        block_m=routing.block_size,
        block_w=block_w,
        block_k=int(config["BLOCK_SIZE_K"]),
        group_m=int(config["GROUP_SIZE_M"]),
        store_pair_act=act_pairs is not None,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def fused_b_act_masked(
    *,
    activation: str,
    base_gateup: torch.Tensor,  # [E_local, m_max, slices * inter] bf16
    act_masked: torch.Tensor,  # [E_local, m_max, inter] bf16
    act_pairs: torch.Tensor | None,
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32 slab rows
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor,
    b_gate_up: torch.Tensor,
    bridge_top_k: int = 1,
) -> None:
    _launch_b_act(
        activation=activation,
        base_rows=base_gateup.view(-1, base_gateup.shape[-1]),
        act_rows=act_masked.view(-1, act_masked.shape[-1]),
        act_pairs=act_pairs,
        pair_to_row=pair_to_row,
        routing=routing,
        num_local_experts=num_local_experts,
        gate_first=gate_first,
        interleaved=interleaved,
        config=config,
        bridge_gateup=bridge_gateup,
        b_gate_up=b_gate_up,
        bridge_top_k=bridge_top_k,
    )


def fused_b_act_contiguous(
    *,
    activation: str,
    base_gateup: torch.Tensor,  # [m_pad_ceiling, slices * inter] bf16
    act_compact: torch.Tensor,  # [m_pad_ceiling, inter] bf16
    act_pairs: torch.Tensor | None,
    pair_to_row: torch.Tensor,  # [num_tokens * top_k] int32 COMPACT rows
    routing: RouteView,
    num_local_experts: int,
    gate_first: bool,
    interleaved: bool,
    config: Mapping[str, int],
    bridge_gateup: torch.Tensor,
    b_gate_up: torch.Tensor,
    bridge_top_k: int = 1,
) -> None:
    _launch_b_act(
        activation=activation,
        base_rows=base_gateup,
        act_rows=act_compact,
        act_pairs=act_pairs,
        pair_to_row=pair_to_row,
        routing=routing,
        num_local_experts=num_local_experts,
        gate_first=gate_first,
        interleaved=interleaved,
        config=config,
        bridge_gateup=bridge_gateup,
        b_gate_up=b_gate_up,
        bridge_top_k=bridge_top_k,
    )
