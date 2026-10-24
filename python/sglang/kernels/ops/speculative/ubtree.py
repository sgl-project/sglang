# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See sglang/srt/speculative/ubtree_components/LICENSE for the original license.

"""Allocation-free metadata preparation for UBTree draft-KV commits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_draft_commit_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    positions_out,
    cache_loc_out,
    req_to_token_stride_row: tl.constexpr,
    positions_stride_row: tl.constexpr,
    cache_loc_stride_row: tl.constexpr,
    width: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    valid = columns < width

    seq_len = tl.load(seq_lens + row).to(tl.int64)
    req_pool_index = tl.load(req_pool_indices + row).to(tl.int64)
    positions = seq_len + columns
    cache_locs = tl.load(
        req_to_token + req_pool_index * req_to_token_stride_row + positions,
        mask=valid,
        other=0,
    ).to(tl.int64)

    tl.store(
        positions_out + row * positions_stride_row + columns,
        positions,
        mask=valid,
    )
    tl.store(
        cache_loc_out + row * cache_loc_stride_row + columns,
        cache_locs,
        mask=valid,
    )


def prepare_draft_commit_metadata(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
) -> None:
    """Fill dense positions/cache locations without allocating temporary tensors.

    For each request ``b`` and commit slot ``i`` this writes::

        positions_out[b, i] = seq_lens[b] + i
        cache_loc_out[b, i] = req_to_token[req_pool_indices[b], positions_out[b, i]]

    The output tensors are caller-owned so their addresses can remain stable across
    decode steps and can later be consumed by a larger CUDA Graph region.
    """

    if seq_lens.ndim != 1 or req_pool_indices.ndim != 1:
        raise ValueError("seq_lens and req_pool_indices must be one-dimensional")
    batch_size = int(seq_lens.numel())
    if int(req_pool_indices.numel()) != batch_size:
        raise ValueError("req_pool_indices must contain one index per request")
    if positions_out.ndim != 2 or cache_loc_out.ndim != 2:
        raise ValueError("positions_out and cache_loc_out must be two-dimensional")
    if positions_out.shape != cache_loc_out.shape:
        raise ValueError("positions_out and cache_loc_out must have matching shapes")
    if int(positions_out.shape[0]) != batch_size:
        raise ValueError("output batch dimension must match seq_lens")
    if positions_out.dtype != torch.int64 or cache_loc_out.dtype != torch.int64:
        raise ValueError("positions_out and cache_loc_out must use torch.int64")
    if not positions_out.is_contiguous() or not cache_loc_out.is_contiguous():
        raise ValueError("positions_out and cache_loc_out must be contiguous")
    if req_to_token.ndim != 2:
        raise ValueError("req_to_token must be two-dimensional")
    if not (
        seq_lens.device
        == req_pool_indices.device
        == req_to_token.device
        == positions_out.device
        == cache_loc_out.device
    ):
        raise ValueError("all draft commit metadata tensors must use the same device")

    width = int(positions_out.shape[1])
    if batch_size == 0 or width == 0:
        return

    if seq_lens.device.type == "cuda":
        block_size = triton.next_power_of_2(width)
        _prepare_draft_commit_kernel[(batch_size,)](
            seq_lens,
            req_pool_indices,
            req_to_token,
            positions_out,
            cache_loc_out,
            req_to_token_stride_row=req_to_token.stride(0),
            positions_stride_row=positions_out.stride(0),
            cache_loc_stride_row=cache_loc_out.stride(0),
            width=width,
            block_size=block_size,
        )
        return

    offsets = torch.arange(width, dtype=torch.int64, device=seq_lens.device)
    torch.add(seq_lens.to(torch.int64).unsqueeze(1), offsets, out=positions_out)
    torch.gather(
        req_to_token[req_pool_indices.to(torch.int64)],
        1,
        positions_out,
        out=cache_loc_out,
    )


@triton.jit
def _prepare_target_kv_move_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    accept_index,
    accept_lens,
    out_cache_loc,
    target_cache_loc_out,
    source_cache_loc_out,
    req_to_token_stride_row: tl.constexpr,
    accept_index_stride_row: tl.constexpr,
    accept_width: tl.constexpr,
    batch_size: tl.constexpr,
):
    output_index = tl.program_id(0)
    cursor = tl.zeros((), dtype=tl.int32)
    selected_request = tl.zeros((), dtype=tl.int32)
    selected_column = tl.zeros((), dtype=tl.int32)
    output_valid = tl.zeros((), dtype=tl.int1)

    for request_index in range(batch_size):
        request_len = tl.load(accept_lens + request_index).to(tl.int32)
        belongs = (output_index >= cursor) & (output_index < cursor + request_len)
        selected_request = tl.where(belongs, request_index, selected_request)
        selected_column = tl.where(belongs, output_index - cursor, selected_column)
        output_valid |= belongs
        cursor += request_len

    seq_len = tl.load(seq_lens + selected_request).to(tl.int64)
    req_pool_index = tl.load(req_pool_indices + selected_request).to(tl.int64)
    target_cache_loc = tl.load(
        req_to_token
        + req_pool_index * req_to_token_stride_row
        + seq_len
        + selected_column,
        mask=output_valid,
        other=0,
    ).to(tl.int64)
    accept_node_index = tl.load(
        accept_index + selected_request * accept_index_stride_row + selected_column,
        mask=output_valid,
        other=0,
    ).to(tl.int64)
    source_cache_loc = tl.load(
        out_cache_loc + accept_node_index,
        mask=output_valid & (accept_node_index >= 0),
        other=0,
    ).to(tl.int64)

    tl.store(target_cache_loc_out + output_index, target_cache_loc)
    tl.store(source_cache_loc_out + output_index, source_cache_loc)


def prepare_target_kv_move_metadata(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    target_cache_loc_out: torch.Tensor,
    source_cache_loc_out: torch.Tensor,
) -> None:
    """Build dense source/target arrays for accepted target-KV movement.

    The accepted path is compacted in request-major order. Any unused output
    tail is explicitly zeroed to preserve the generic SGLang mover's padding
    semantics while keeping all output storage caller-owned.
    """

    if accept_index.ndim != 2:
        raise ValueError("accept_index must be two-dimensional")
    batch_size, accept_width = map(int, accept_index.shape)
    if any(
        int(tensor.numel()) != batch_size
        for tensor in (req_pool_indices, seq_lens, accept_lens)
    ):
        raise ValueError("request metadata must contain one value per request")
    size = batch_size * accept_width
    if target_cache_loc_out.ndim != 1 or source_cache_loc_out.ndim != 1:
        raise ValueError("target/source cache output buffers must be one-dimensional")
    if (
        int(target_cache_loc_out.numel()) != size
        or int(source_cache_loc_out.numel()) != size
    ):
        raise ValueError("target/source cache output buffers have the wrong size")
    if (
        target_cache_loc_out.dtype != torch.int64
        or source_cache_loc_out.dtype != torch.int64
    ):
        raise ValueError("target/source cache output buffers must use torch.int64")
    tensors = (
        req_pool_indices,
        req_to_token,
        seq_lens,
        accept_index,
        accept_lens,
        out_cache_loc,
        target_cache_loc_out,
        source_cache_loc_out,
    )
    if any(tensor.device != seq_lens.device for tensor in tensors):
        raise ValueError("all target KV move metadata tensors must use the same device")
    if size == 0:
        return

    if seq_lens.device.type == "cuda":
        _prepare_target_kv_move_kernel[(size,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            accept_index,
            accept_lens,
            out_cache_loc,
            target_cache_loc_out,
            source_cache_loc_out,
            req_to_token_stride_row=req_to_token.stride(0),
            accept_index_stride_row=accept_index.stride(0),
            accept_width=accept_width,
            batch_size=batch_size,
        )
        return

    target_cache_loc_out.zero_()
    source_cache_loc_out.zero_()
    output_offset = 0
    for request_index in range(batch_size):
        request_len = int(accept_lens[request_index])
        request_slice = slice(output_offset, output_offset + request_len)
        positions = seq_lens[request_index].to(torch.int64) + torch.arange(
            request_len, dtype=torch.int64, device=seq_lens.device
        )
        target_cache_loc_out[request_slice].copy_(
            req_to_token[int(req_pool_indices[request_index]), positions]
        )
        accept_nodes = accept_index[request_index, :request_len].to(torch.int64)
        valid_nodes = accept_nodes >= 0
        request_source = source_cache_loc_out[request_slice]
        request_source[valid_nodes] = out_cache_loc[accept_nodes[valid_nodes]]
        output_offset += request_len


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

    if seq_lens.device.type == "cuda":
        block_size = triton.next_power_of_2(width)
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
            block_size=block_size,
        )
        return

    offsets = torch.arange(width, dtype=torch.int64, device=seq_lens.device)
    torch.add(seq_lens.to(torch.int64).unsqueeze(1), offsets, out=draft_positions_out)
    request_rows = req_pool_indices.to(torch.int64)
    torch.gather(
        draft_req_to_token[request_rows],
        1,
        draft_positions_out,
        out=draft_cache_loc_out,
    )
    torch.gather(
        target_req_to_token[request_rows],
        1,
        draft_positions_out,
        out=target_cache_loc_out,
    )
    accepted = offsets.unsqueeze(0) < accept_lens.to(torch.int64).unsqueeze(1)
    target_cache_loc_out.masked_fill_(~accepted, 0)
    source_cache_loc_out.zero_()
    safe_accept_index = accept_index.to(torch.int64).clamp_min(0)
    gathered_source = out_cache_loc[safe_accept_index]
    source_cache_loc_out.copy_(torch.where(accepted, gathered_source, 0))


__all__ = [
    "prepare_draft_commit_metadata",
    "prepare_unified_commit_metadata",
    "prepare_target_kv_move_metadata",
]
