"""Interleave ownership for bounded replay, shared by eager and prefill graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.utils.common import async_h2d


@dataclass
class InterleaveTailBuffers:
    """Per-capacity buffers: replay changes contents, never captured addresses."""

    token_indices: torch.Tensor
    positions: torch.Tensor
    scatter_indices: torch.Tensor
    gather_index: torch.Tensor
    local_index: torch.Tensor
    moe_count: torch.Tensor

    @classmethod
    def create(
        cls, local_capacity: int, cp_size: int, positions: torch.Tensor
    ) -> InterleaveTailBuffers:
        zeros = lambda n: torch.zeros(n, dtype=torch.int64, device=positions.device)
        return cls(
            token_indices=zeros(local_capacity),
            positions=positions.new_zeros(local_capacity),
            scatter_indices=zeros(local_capacity),
            gather_index=zeros(local_capacity * cp_size),
            local_index=zeros(local_capacity),
            moe_count=torch.zeros((), dtype=torch.int32, device=positions.device),
        )


def interleave_tail_local_capacity(
    full_local_rows: int, max_tail_rows: int, cp_size: int, window: int
) -> int:
    # A request's tail can start on any rank. ceil(window / CP), rather than
    # ceil(total_tail / CP), bounds its largest share even for many short requests.
    max_requests = max_tail_rows // window
    return min(full_local_rows, max_requests * ((window + cp_size - 1) // cp_size))


def build_interleave_tail_layout(
    *,
    extend_lens: List[int],
    tail_lens: List[int],
    positions: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    buffers: Optional[InterleaveTailBuffers] = None,
    full_local_rows: Optional[int] = None,
):
    """Keep the original prompt's owners, not the compact tail's row modulo CP.

    Compute geometry from host-known lengths: no device nonzero/bincount D2H
    synchronization. Graphs use uniform *physical* counts and a separate live
    MoE count; eager uses exact logical counts. Padding metadata rows are unique.
    """
    by_rank = [[] for _ in range(cp_size)]
    local_lens = [0] * len(extend_lens)
    start = tail_row = 0
    for request, (length, tail) in enumerate(zip(extend_lens, tail_lens)):
        for token in range(start + length - tail, start + length):
            owner = token % cp_size
            by_rank[owner].append((token // cp_size, tail_row))
            if owner == cp_rank:
                local_lens[request] += 1
            tail_row += 1
        start += length
    counts = [len(rows) for rows in by_rank]
    local_capacity = (
        buffers.token_indices.shape[0] if buffers is not None else max(counts)
    )
    assert max(counts) <= local_capacity, (counts, local_capacity)
    total_capacity = local_capacity * cp_size
    gathered = [0] * total_capacity
    pad_cursor = tail_row
    local_index = []
    for rank, rows in enumerate(by_rank):
        metadata_rows = [row for _, row in rows]
        for slot, (_, row) in enumerate(rows):
            gathered[row] = rank * local_capacity + slot
        for slot in range(len(rows), local_capacity):
            gathered[pad_cursor] = rank * local_capacity + slot
            metadata_rows.append(pad_cursor)
            pad_cursor += 1
        if rank == cp_rank:
            local_index = metadata_rows
    device = positions.device
    index = lambda values: async_h2d(values, dtype=torch.int64, device=device)
    local_tokens = index([token for token, _ in by_rank[cp_rank]])
    global_tokens = local_tokens * cp_size + cp_rank
    local_positions = positions[global_tokens]
    pad_rows = local_capacity - counts[cp_rank]
    local_positions = torch.cat([local_positions, positions.new_zeros(pad_rows)])
    gather_index = index(gathered if buffers is not None else gathered[:tail_row])
    metadata_index = index(local_index)
    if buffers is not None:
        assert full_local_rows is not None
        n = counts[cp_rank]
        buffers.token_indices.zero_()
        buffers.token_indices[:n].copy_(local_tokens)
        buffers.positions.copy_(local_positions)
        buffers.scatter_indices.fill_(full_local_rows)
        buffers.scatter_indices[:n].copy_(local_tokens)
        buffers.gather_index.copy_(gather_index)
        buffers.local_index.copy_(metadata_index)
        buffers.moe_count.fill_(n)
        local_tokens, local_positions = buffers.token_indices, buffers.positions
        gather_index, metadata_index = buffers.gather_index, buffers.local_index
        moe_count = buffers.moe_count
    else:
        moe_count = async_h2d(
            [counts[cp_rank]], dtype=torch.int32, device=device
        ).squeeze(0)
    metadata = InterleaveContextParallelMetadata(
        per_rank_actual_token=[local_capacity] * cp_size,
        max_rank_len=[local_capacity] * cp_size,
        total_seq_lens=total_capacity if buffers is not None else tail_row,
        bs=len(extend_lens),
        per_rank_logical_token=(
            [local_capacity] * cp_size if buffers is not None else counts
        ),
        gather_index=gather_index,
        local_index=metadata_index,
        moe_local_token_count=moe_count,
    )
    return dict(
        cp_metadata=metadata,
        local_token_indices=local_tokens,
        local_positions=local_positions,
        local_lens_cpu=local_lens,
        pad_rows=0 if buffers is not None else pad_rows,
    )
