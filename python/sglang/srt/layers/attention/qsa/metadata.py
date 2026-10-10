"""Metadata owned by the simple QSA implementation.

QSA intentionally does not inherit the NSA metadata abstraction. This module
contains only fields and transforms consumed by the indexer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.mem_cache.qsa_kv_pool import (
    QSACompressedBlockSharding,
    assert_qsa_indices_in_bounds,
)


def build_qsa_row_ranges(
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    query_sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed compressed-key ranges for prefill scoring."""

    sequence_lengths = sequence_lengths.to(dtype=torch.int32)
    compressed_lengths = torch.div(
        sequence_lengths, compress_ratio, rounding_mode="floor"
    )
    compressed_cu_seqlens = torch.nn.functional.pad(
        compressed_lengths.cumsum(0), (1, 0)
    ).to(torch.int32)
    query_sequence_ids = query_sequence_ids.to(
        device=sequence_lengths.device, dtype=torch.long
    )
    row_starts = compressed_cu_seqlens.index_select(0, query_sequence_ids)
    visible_blocks = torch.div(
        query_positions.to(device=sequence_lengths.device, dtype=torch.int32) + 1,
        compress_ratio,
        rounding_mode="floor",
    )
    max_blocks = compressed_lengths.index_select(0, query_sequence_ids)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    return row_starts, row_ends, compressed_cu_seqlens


def count_visible_local_blocks(
    logical_block_positions: torch.Tensor,
    local_lengths: torch.Tensor,
    sequence_ids: torch.Tensor,
    visible_blocks: torch.Tensor,
) -> torch.Tensor:
    """Count visible owner-local blocks without a rows-by-prefix workspace."""

    if logical_block_positions.ndim != 2:
        raise ValueError("logical_block_positions must be rank 2")
    if local_lengths.ndim != 1:
        raise ValueError("local_lengths must be rank 1")
    if logical_block_positions.shape[0] != local_lengths.numel():
        raise ValueError("logical block rows must match local lengths")
    if sequence_ids.shape != visible_blocks.shape:
        raise ValueError("sequence_ids and visible_blocks must have matching shapes")

    counts = torch.empty_like(visible_blocks, dtype=torch.int32)
    for sequence_id, local_length in enumerate(local_lengths.tolist()):
        row_mask = sequence_ids == sequence_id
        if not bool(row_mask.any().item()):
            continue
        sorted_positions = logical_block_positions[
            sequence_id, : int(local_length)
        ].contiguous()
        counts[row_mask] = torch.searchsorted(
            sorted_positions,
            visible_blocks[row_mask].to(sorted_positions.dtype),
            right=False,
        ).to(torch.int32)
    return counts


class QSAIndexerMetadata(msgspec.Struct, frozen=True):
    """All per-forward metadata consumed specifically by ``QSAIndexer``.

    Row layout contract:

    - ``sequence_lengths``/``token_slot_table`` carry one row per *sequence*
      for extend modes and one row per *query token* for the paged modes
      (decode, target_verify, draft_extend).
    - ``token_to_batch_idx`` maps every query/token row handled by the indexer
      onto a row of ``sequence_lengths``/``token_slot_table``; DP attention
      token padding adds physical rows beyond this mapping, never inside it.
      For CP extend, it covers the global packed *new* tokens used by the
      compression write plan. Query selection uses a zigzag-sharded copy;
      cached prefix tokens are K/V context, not rows in this mapping.
    - For the paged modes the mapping is the identity
      (``arange(num_query_rows)``), so page-table/MQA inputs built per
      ``sequence_lengths`` row line up with per-query sparse-attention rows.
    """

    sequence_lengths: torch.Tensor
    token_to_batch_idx: torch.Tensor
    token_slot_table: torch.Tensor
    out_cache_loc: torch.Tensor
    token_to_kv_pool: object
    compress_ratio: int
    block_topk: int
    req_pool_indices: Optional[torch.Tensor] = None
    # Parallel per-group arrays for the groups compressed this forward:
    # slot, sequence-local group-end position, and owning metadata row.
    # The first member's token row in this forward's packed tensors is extend only,
    # where group-aligned chunks keep every member in-chunk; None on paged forwards.
    write_locs: Optional[torch.Tensor] = None
    write_owner_mask: Optional[torch.Tensor] = None
    local_write_locs: Optional[torch.Tensor] = None
    compress_group_positions: Optional[torch.Tensor] = None
    compress_sequence_ids: Optional[torch.Tensor] = None
    compress_member_rows: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    graph_write_locs: Optional[torch.Tensor] = None
    graph_compressed_page_table: Optional[torch.Tensor] = None
    graph_compressed_lengths: Optional[torch.Tensor] = None
    graph_prefix_lengths: Optional[torch.Tensor] = None
    prefill_page_table: Optional[torch.Tensor] = None
    prefill_lengths: Optional[torch.Tensor] = None
    prefill_block_positions: Optional[torch.Tensor] = None
    decode_page_table: Optional[torch.Tensor] = None
    decode_lengths: Optional[torch.Tensor] = None
    decode_block_positions: Optional[torch.Tensor] = None
    decode_logical_positions: Optional[torch.Tensor] = None
    pending_ring_slots: Optional[torch.Tensor] = None
    compress_group_ring_locs: Optional[torch.Tensor] = None
    extend_rope_matrix: Optional[torch.Tensor] = None
    graph_ring_group_locs: Optional[torch.Tensor] = None
    defer_block_expansion: bool = False
    # Scheduler lengths for the breakable prefill indexer.
    prefill_sequence_lengths_cpu: Optional[Tuple[int, ...]] = None

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.sequence_lengths.to(torch.int32)

    def get_token_slot_table(self) -> torch.Tensor:
        return self.token_slot_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.get_seqlens_int32().index_select(
            0, self.get_token_to_batch_idx().long()
        )

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
        row_ends: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk != self.block_topk:
            raise ValueError(
                f"QSA compressed top-k must be {self.block_topk}, got {topk}"
            )
        if row_starts is None or row_ends is None:
            raise ValueError("QSA top-k transform requires row_starts and row_ends")
        return qsa_fast_topk(logits, row_starts, row_ends, topk=self.block_topk)

    def get_prefill_mqa_inputs(
        self,
        layer_id: int,
        positions: torch.Tensor,
        query_sequence_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather packed compressed K and ragged ranges for prefill MQA.

        ``query_sequence_ids`` overrides ``token_to_batch_idx`` for callers that
        score a subset of the batch's rows (prefill CP: this rank's zigzag rows);
        ``positions`` must then be those rows' logical positions."""

        pool = self.token_to_kv_pool
        ratio = self.compress_ratio
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        sequence_lengths = self.sequence_lengths.to(torch.int32)
        parts: list[torch.Tensor] = []
        if self.prefill_page_table is not None:
            if self.prefill_lengths is None or self.prefill_block_positions is None:
                raise RuntimeError("QSA local prefill page metadata is incomplete")
            page_size = pool.qsa_compressed_page_size
            cache = compressed_buffer.reshape(
                -1, page_size, pool.qsa_index_kv_heads, pool.qsa_index_head_dim
            )
            for sequence_id, local_length in enumerate(self.prefill_lengths.tolist()):
                if local_length == 0:
                    continue
                pages_needed = -(int(local_length) // -page_size)
                pages = self.prefill_page_table[sequence_id, :pages_needed].long()
                parts.append(cache.index_select(0, pages).flatten(0, 1)[:local_length])
        else:
            sequence_lengths_list = self.prefill_sequence_lengths_cpu
            if sequence_lengths_list is None:
                sequence_lengths_list = sequence_lengths.tolist()
            for sequence_id in range(len(sequence_lengths_list)):
                complete_blocks = int(sequence_lengths_list[sequence_id]) // ratio
                if complete_blocks == 0:
                    continue
                compressed_locs = (
                    self.token_slot_table[
                        sequence_id, : complete_blocks * ratio : ratio
                    ].long()
                    // ratio
                )
                parts.append(compressed_buffer.index_select(0, compressed_locs))
        compressed_keys = (
            torch.cat(parts, dim=0)
            if parts
            else compressed_buffer.new_empty(
                (0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim)
            )
        )
        if query_sequence_ids is None:
            query_sequence_ids = self.token_to_batch_idx
        num_valid_tokens = query_sequence_ids.numel()
        if positions.numel() < num_valid_tokens:
            raise ValueError(
                "QSA prefill positions are shorter than the request mapping: "
                f"positions={positions.numel()}, mapping={num_valid_tokens}"
            )
        positions = positions[:num_valid_tokens]
        if self.prefill_lengths is None:
            row_starts, row_ends, _ = build_qsa_row_ranges(
                sequence_lengths,
                positions.to(sequence_lengths.device),
                query_sequence_ids.to(sequence_lengths.device),
                self.compress_ratio,
            )
        else:
            cumulative = torch.nn.functional.pad(
                self.prefill_lengths.to(torch.int32).cumsum(0), (1, 0)
            ).to(torch.int32)
            sequence_ids = query_sequence_ids.long()
            row_starts = cumulative.index_select(0, sequence_ids)
            visible_blocks = torch.div(
                positions.to(sequence_lengths.device, dtype=torch.int64) + 1,
                ratio,
                rounding_mode="floor",
            )
            visible_local = count_visible_local_blocks(
                self.prefill_block_positions,
                self.prefill_lengths,
                sequence_ids,
                visible_blocks,
            )
            row_ends = row_starts + visible_local
        return compressed_keys, row_starts, row_ends, sequence_lengths

    def get_decode_mqa_inputs(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Paged compressed-K cache inputs for decode MQA, one row per query row."""

        pool = self.token_to_kv_pool
        num_rows = self.sequence_lengths.numel()
        if self.token_slot_table.shape[0] != num_rows:
            raise ValueError(
                "QSA decode page-table rows must match the per-query sequence "
                f"lengths: table_rows={self.token_slot_table.shape[0]}, "
                f"rows={num_rows}"
            )
        compressed_cache = pool.get_qsa_compressed_k_buffer(layer_id).reshape(
            -1,
            pool.qsa_compressed_page_size,
            pool.qsa_index_kv_heads,
            pool.qsa_index_head_dim,
        )
        if self.is_cuda_graph:
            if (
                self.graph_compressed_page_table is None
                or self.graph_compressed_lengths is None
            ):
                raise RuntimeError("QSA CUDA graph decode metadata is incomplete")
            return (
                compressed_cache,
                self.graph_compressed_page_table,
                self.graph_compressed_lengths,
                self.graph_compressed_page_table.shape[1]
                * pool.qsa_compressed_page_size,
            )
        if self.decode_page_table is not None and self.decode_lengths is not None:
            return (
                compressed_cache,
                self.decode_page_table,
                self.decode_lengths,
                self.decode_page_table.shape[1] * pool.qsa_compressed_page_size,
            )
        compressed_page_table, compressed_lengths = compressed_decode_view(
            compressed_page_size=pool.qsa_compressed_page_size,
            compress_ratio=self.compress_ratio,
            sequence_lengths=self.sequence_lengths,
            token_slot_table=self.token_slot_table,
        )
        return (
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            compressed_page_table.shape[1] * pool.qsa_compressed_page_size,
        )


def build_pending_ring_slots(
    *,
    token_to_batch_idx: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sequence_lengths: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    is_extend: bool,
) -> torch.Tensor:
    """Pending-ring slot ``req_pool_idx * ratio + position % ratio`` per token.
    On extend, tokens before the pending tail dump into rows [0, ratio),
    which no request owns (request slot 0 is never allocated); CUDA-graph safe."""
    rows = token_to_batch_idx.long()[: logical_positions.numel()]
    requests = req_pool_indices.long()[rows]
    positions = logical_positions.long()
    slots = requests * compress_ratio + positions % compress_ratio
    if is_extend:
        lengths = sequence_lengths.long()[rows]
        pending = positions >= (lengths // compress_ratio) * compress_ratio
        slots = torch.where(pending, slots, positions % compress_ratio)
    return slots


def build_group_ring_slots(
    *,
    req_pool_indices: torch.Tensor,
    group_end_positions: torch.Tensor,
    sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Ring slots of a planned group's members, oldest first."""
    requests = req_pool_indices.long()[sequence_ids]
    offsets = torch.arange(
        compress_ratio - 1,
        -1,
        -1,
        device=group_end_positions.device,
        dtype=torch.long,
    )
    positions = (group_end_positions[:, None] - offsets[None, :]).clamp_min(0)
    return requests[:, None] * compress_ratio + positions % compress_ratio


def build_rope_position_matrix(
    rope_positions: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """This forward's RoPE coordinates as the [tokens, 3] layout the fused
    compress kernel reads."""
    if rope_positions.ndim == 1:
        return (
            rope_positions[:num_tokens].long().unsqueeze(1).expand(-1, 3)
        ).contiguous()
    return rope_positions[:, :num_tokens].long().transpose(0, 1).contiguous()


def compressed_decode_view(
    *,
    compressed_page_size: int,
    compress_ratio: int,
    sequence_lengths: torch.Tensor,
    token_slot_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed page table and lengths for decode MQA.

    Page-table entries are full-KV page ids read off the page-aligned
    token-slot rows; the scoring kernel converts them to compressed
    slots as page_id * compressed_page_size + block_in_page. Entries
    past a row's compressed length are stale-but-unread (bounded by
    compressed_lengths); clamp keeps them non-negative.
    """
    full_page = compressed_page_size * compress_ratio
    compressed_lengths = torch.div(
        sequence_lengths.to(torch.int32),
        compress_ratio,
        rounding_mode="floor",
    )
    compressed_page_table = (
        (token_slot_table[:, ::full_page].long() // full_page)
        .clamp_min(0)
        .to(torch.int32)
    )
    return compressed_page_table, compressed_lengths


def localize_compressed_page_table(
    *,
    global_page_table: torch.Tensor,
    compressed_lengths: torch.Tensor,
    sharding: QSACompressedBlockSharding,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Localize owned physical pages and preserve each logit's logical position.

    Page-table rows may contain non-monotonic physical page ids. Ownership and
    cache addressing follow those physical ids, while ``logical_positions``
    records the original sequence-local compressed-block position represented
    by each compacted logit.
    """

    if global_page_table.ndim != 2:
        raise ValueError(
            "global_page_table must be rank 2, got "
            f"shape={tuple(global_page_table.shape)}"
        )
    if compressed_lengths.ndim != 1:
        raise ValueError(
            "compressed_lengths must be rank 1, got "
            f"shape={tuple(compressed_lengths.shape)}"
        )
    if global_page_table.shape[0] != compressed_lengths.numel():
        raise ValueError(
            "page-table rows must match compressed lengths: "
            f"rows={global_page_table.shape[0]}, "
            f"lengths={compressed_lengths.numel()}"
        )
    if global_page_table.device != compressed_lengths.device:
        raise ValueError("global_page_table and compressed_lengths must share a device")
    if torch.any(compressed_lengths < 0):
        raise ValueError("compressed_lengths must be non-negative")

    page_size = sharding.compressed_page_size
    row_capacity = global_page_table.shape[1] * page_size
    if torch.any(compressed_lengths > row_capacity):
        raise ValueError(
            f"compressed length exceeds page-table capacity: capacity={row_capacity}"
        )

    logical_page_starts = (
        torch.arange(
            global_page_table.shape[1],
            dtype=torch.int64,
            device=global_page_table.device,
        )
        * page_size
    )
    visible_pages = logical_page_starts.unsqueeze(
        0
    ) < compressed_lengths.long().unsqueeze(1)
    visible_physical_pages = global_page_table[visible_pages]
    if not global_page_table.is_cuda and torch.any(
        (visible_physical_pages < 0) | (visible_physical_pages >= sharding.global_pages)
    ):
        raise ValueError("visible compressed physical page is out of range")
    assert_qsa_indices_in_bounds(
        global_page_table,
        sharding.global_pages,
        valid_mask=visible_pages,
        label="compressed KV page table",
    )

    local_page_ids = sharding.global_to_local_pages(global_page_table)
    owned_pages = visible_pages & (local_page_ids >= 0)
    assert_qsa_indices_in_bounds(
        local_page_ids,
        sharding.local_pages,
        valid_mask=owned_pages,
        label="compressed KV local page table",
    )
    local_page_counts = owned_pages.sum(dim=1, dtype=torch.int32)
    output_pages = (
        int(local_page_counts.max().item()) if local_page_counts.numel() else 0
    )
    local_page_table = torch.full(
        (global_page_table.shape[0], output_pages),
        -1,
        dtype=global_page_table.dtype,
        device=global_page_table.device,
    )
    local_lengths = torch.zeros_like(compressed_lengths, dtype=torch.int32)
    logical_positions = torch.full(
        (global_page_table.shape[0], output_pages * page_size),
        -1,
        dtype=torch.int64,
        device=global_page_table.device,
    )
    for row in range(global_page_table.shape[0]):
        owned_columns = torch.nonzero(owned_pages[row], as_tuple=False).flatten()
        row_position = 0
        for output_page, column in enumerate(owned_columns):
            local_page_table[row, output_page] = local_page_ids[row, column]
            logical_start = int(column.item()) * page_size
            visible_blocks = min(
                page_size,
                int(compressed_lengths[row].item()) - logical_start,
            )
            logical_positions[row, row_position : row_position + visible_blocks] = (
                torch.arange(
                    logical_start,
                    logical_start + visible_blocks,
                    dtype=torch.int64,
                    device=global_page_table.device,
                )
            )
            row_position += visible_blocks
        local_lengths[row] = row_position
    return local_page_table, local_lengths, logical_positions


__all__ = [
    "QSAIndexerMetadata",
    "build_qsa_row_ranges",
    "build_pending_ring_slots",
    "build_group_ring_slots",
    "build_rope_position_matrix",
    "compressed_decode_view",
    "count_visible_local_blocks",
    "localize_compressed_page_table",
]
