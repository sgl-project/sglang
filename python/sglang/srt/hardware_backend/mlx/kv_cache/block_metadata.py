"""Block-table paged-attention metadata helpers for the MLX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class MLXBlockPagedAttentionMetadata:
    """Metadata for one MLX block-table paged-attention decode call."""

    block_tables: mx.array
    seq_lens: mx.array
    slot_mapping: mx.array
    block_ids: mx.array
    block_offsets: mx.array
    block_size: int
    max_seq_len: int
    total_kv_tokens: int


def _read_block_table(req_to_block: Any, row: int, col: int) -> int:
    value = req_to_block[row, col]
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def build_decode_block_metadata(
    *,
    req_ids: list[str],
    req_pool_idx: dict[str, int],
    req_to_block_pool: Any,
    seq_lens_before_decode: list[int],
    block_size: int = 16,
    include_current_token: bool = True,
) -> MLXBlockPagedAttentionMetadata:
    """Build decode metadata for block-table paged attention.

    ``req_to_block_pool.req_to_block[row, logical_block]`` is treated as the
    source of truth for each request's physical block table. The returned table
    is padded with ``-1`` to a rectangular batch shape.
    """

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if len(req_ids) != len(seq_lens_before_decode):
        raise ValueError(
            "req_ids and seq_lens_before_decode must have the same length, "
            f"got {len(req_ids)} and {len(seq_lens_before_decode)}"
        )
    if req_to_block_pool is None or not hasattr(req_to_block_pool, "req_to_block"):
        raise ValueError("req_to_block_pool with req_to_block is required")

    req_to_block = req_to_block_pool.req_to_block
    per_req_tables: list[list[int]] = []
    seq_lens: list[int] = []
    slot_mapping: list[int] = []
    block_ids: list[int] = []
    block_offsets: list[int] = []

    for batch_idx, req_id in enumerate(req_ids):
        if req_id not in req_pool_idx:
            raise KeyError(f"missing req_pool_idx for request {req_id!r}")

        row = int(req_pool_idx[req_id])
        before_len = int(seq_lens_before_decode[batch_idx])
        if before_len < 0:
            raise ValueError(f"negative sequence length for request {req_id!r}")

        visible_len = before_len + 1 if include_current_token else before_len
        needed_blocks = (visible_len + block_size - 1) // block_size
        table: list[int] = []
        for logical_block in range(needed_blocks):
            physical_block = _read_block_table(req_to_block, row, logical_block)
            if physical_block < 0:
                raise ValueError(
                    "missing block for visible token range "
                    f"(request={req_id!r}, row={row}, logical_block={logical_block})"
                )
            table.append(physical_block)

        per_req_tables.append(table)
        seq_lens.append(visible_len)
        slot_mapping.append(before_len)
        if include_current_token:
            current_logical_block = before_len // block_size
            block_ids.append(table[current_logical_block])
            block_offsets.append(before_len % block_size)
        else:
            block_ids.append(-1)
            block_offsets.append(-1)

    max_num_blocks = max((len(table) for table in per_req_tables), default=0)
    padded_tables = [
        table + [-1] * (max_num_blocks - len(table)) for table in per_req_tables
    ]

    return MLXBlockPagedAttentionMetadata(
        block_tables=mx.array(padded_tables, dtype=mx.int32),
        seq_lens=mx.array(seq_lens, dtype=mx.int32),
        slot_mapping=mx.array(slot_mapping, dtype=mx.int32),
        block_ids=mx.array(block_ids, dtype=mx.int32),
        block_offsets=mx.array(block_offsets, dtype=mx.int32),
        block_size=block_size,
        max_seq_len=max(seq_lens, default=0),
        total_kv_tokens=sum(seq_lens),
    )
