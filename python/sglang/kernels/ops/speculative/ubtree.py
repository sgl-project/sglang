# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See sglang/srt/speculative/ubtree_components/LICENSE for the original license.

"""Allocation-free metadata preparation for UBTree draft-KV commits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_unified_commit_metadata_kernel(
    req_pool_indices,
    target_req_to_token,
    draft_req_to_token,
    seq_lens,
    accept_index,
    accept_lens,
    out_cache_loc,
    target_cache_loc_out,
    source_cache_loc_out,
    draft_positions_out,
    draft_cache_loc_out,
    target_req_stride_row: tl.constexpr,
    draft_req_stride_row: tl.constexpr,
    accept_index_stride_row: tl.constexpr,
    target_out_stride_row: tl.constexpr,
    source_out_stride_row: tl.constexpr,
    draft_positions_stride_row: tl.constexpr,
    draft_cache_stride_row: tl.constexpr,
    width: tl.constexpr,
    block_size: tl.constexpr,
):
    """Emit target-move and draft-commit layouts from one logical path map."""

    request_index = tl.program_id(0)
    columns = tl.arange(0, block_size)
    in_width = columns < width

    seq_len = tl.load(seq_lens + request_index).to(tl.int64)
    req_pool_index = tl.load(req_pool_indices + request_index).to(tl.int64)
    positions = seq_len + columns

    target_cache_locs = tl.load(
        target_req_to_token + req_pool_index * target_req_stride_row + positions,
        mask=in_width,
        other=0,
    ).to(tl.int64)
    draft_cache_locs = tl.load(
        draft_req_to_token + req_pool_index * draft_req_stride_row + positions,
        mask=in_width,
        other=0,
    ).to(tl.int64)

    accept_len = tl.load(accept_lens + request_index).to(tl.int32)
    accepted = in_width & (columns < accept_len)
    accept_node_index = tl.load(
        accept_index + request_index * accept_index_stride_row + columns,
        mask=accepted,
        other=-1,
    ).to(tl.int64)
    source_cache_locs = tl.load(
        out_cache_loc + accept_node_index,
        mask=accepted & (accept_node_index >= 0),
        other=0,
    ).to(tl.int64)

    target_offsets = request_index * target_out_stride_row + columns
    source_offsets = request_index * source_out_stride_row + columns
    draft_position_offsets = request_index * draft_positions_stride_row + columns
    draft_cache_offsets = request_index * draft_cache_stride_row + columns
    tl.store(
        target_cache_loc_out + target_offsets,
        tl.where(accepted, target_cache_locs, 0),
        mask=in_width,
    )
    tl.store(
        source_cache_loc_out + source_offsets,
        source_cache_locs,
        mask=in_width,
    )
    tl.store(draft_positions_out + draft_position_offsets, positions, mask=in_width)
    tl.store(draft_cache_loc_out + draft_cache_offsets, draft_cache_locs, mask=in_width)


def prepare_unified_commit_metadata(
    *,
    req_pool_indices: torch.Tensor,
    target_req_to_token: torch.Tensor,
    draft_req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    target_cache_loc_out: torch.Tensor,
    source_cache_loc_out: torch.Tensor,
    draft_positions_out: torch.Tensor,
    draft_cache_loc_out: torch.Tensor,
) -> None:
    """Build all accepted-path commit layouts in one pass.

    The shared logical mapping is ``(request_index, accept_column)``. Target
    movement derives its source from the accepted tree node and its destination
    from the target request table. Draft injection uses the same sequence
    position but resolves it through the draft request table. Target movement is
    stored as a request-major dense layout; rejected tail slots are ``0 -> 0``.
    """

    if accept_index.ndim != 2:
        raise ValueError("accept_index must be two-dimensional")
    batch_size, width = map(int, accept_index.shape)
    if any(
        int(tensor.numel()) != batch_size
        for tensor in (req_pool_indices, seq_lens, accept_lens)
    ):
        raise ValueError("request metadata must contain one value per request")
    if target_req_to_token.ndim != 2 or draft_req_to_token.ndim != 2:
        raise ValueError("request-to-token tables must be two-dimensional")
    outputs = (
        target_cache_loc_out,
        source_cache_loc_out,
        draft_positions_out,
        draft_cache_loc_out,
    )
    if any(tensor.ndim != 2 for tensor in outputs):
        raise ValueError("unified commit outputs must be two-dimensional")
    expected_shape = (batch_size, width)
    if any(tuple(tensor.shape) != expected_shape for tensor in outputs):
        raise ValueError(f"unified commit outputs must have shape {expected_shape}")
    if any(tensor.dtype != torch.int64 for tensor in outputs):
        raise ValueError("unified commit outputs must use torch.int64")
    if any(not tensor.is_contiguous() for tensor in outputs):
        raise ValueError("unified commit outputs must be contiguous")
    tensors = (
        req_pool_indices,
        target_req_to_token,
        draft_req_to_token,
        seq_lens,
        accept_index,
        accept_lens,
        out_cache_loc,
        *outputs,
    )
    if any(tensor.device != seq_lens.device for tensor in tensors):
        raise ValueError("all unified commit tensors must use the same device")
    if batch_size == 0 or width == 0:
        return

    _prepare_unified_commit_metadata_kernel[(batch_size,)](
        req_pool_indices,
        target_req_to_token,
        draft_req_to_token,
        seq_lens,
        accept_index,
        accept_lens,
        out_cache_loc,
        target_cache_loc_out,
        source_cache_loc_out,
        draft_positions_out,
        draft_cache_loc_out,
        target_req_stride_row=target_req_to_token.stride(0),
        draft_req_stride_row=draft_req_to_token.stride(0),
        accept_index_stride_row=accept_index.stride(0),
        target_out_stride_row=target_cache_loc_out.stride(0),
        source_out_stride_row=source_cache_loc_out.stride(0),
        draft_positions_stride_row=draft_positions_out.stride(0),
        draft_cache_stride_row=draft_cache_loc_out.stride(0),
        width=width,
        block_size=triton.next_power_of_2(width),
    )


__all__ = ["prepare_unified_commit_metadata"]
