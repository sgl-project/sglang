"""Paged-attention metadata helpers for the MLX backend.

MLX paged-attention uses page-size 1 over the existing flat
``MlxAttentionKVPool`` layout.  The metadata mirrors SGLang's ragged attention
vocabulary: ``req_to_token`` remains the source of truth, while
``kv_indptr``/``kv_indices`` describe the visible physical KV slots for each
decode request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class MLXPagedAttentionMetadata:
    """Metadata for one MLX paged-attention call."""

    kv_indptr: mx.array
    kv_indices: mx.array
    qo_indptr: mx.array
    slot_mapping: mx.array
    seq_lens: list[int]
    max_seq_len: int
    total_kv_tokens: int
    page_size: int = 1


def _read_req_to_token_slot(req_to_token: Any, row: int, col: int) -> int:
    value = req_to_token[row, col]
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def build_decode_paged_metadata(
    *,
    req_ids: list[str],
    req_pool_idx: dict[str, int],
    req_to_token_pool: Any,
    seq_lens_before_decode: list[int],
    include_current_token: bool = True,
) -> MLXPagedAttentionMetadata:
    """Build page-size-1 decode metadata from ``req_to_token``.

    Args:
        req_ids: Request IDs in decode batch order.
        req_pool_idx: Mapping from request ID to scheduler req-pool row.
        req_to_token_pool: Object exposing ``req_to_token[row, position]``.
        seq_lens_before_decode: Per-request cache offsets before this decode
            token is written.
        include_current_token: If true, the visible KV segment includes the
            current decode token at ``seq_lens_before_decode[i]``.  This is the
            contract used by the Metal decode path after RoPE+KV scatter.

    Returns:
        ``MLXPagedAttentionMetadata`` with MLX int32 arrays.

    Raises:
        ValueError: If visible metadata references slot 0, which is reserved as
            padding in ``MlxAttentionKVPool``.
    """

    if len(req_ids) != len(seq_lens_before_decode):
        raise ValueError(
            "req_ids and seq_lens_before_decode must have the same length, "
            f"got {len(req_ids)} and {len(seq_lens_before_decode)}"
        )
    if req_to_token_pool is None or not hasattr(req_to_token_pool, "req_to_token"):
        raise ValueError("req_to_token_pool with req_to_token is required")

    req_to_token = req_to_token_pool.req_to_token
    kv_indptr = [0]
    kv_indices: list[int] = []
    slot_mapping: list[int] = []
    visible_lens: list[int] = []

    for batch_idx, req_id in enumerate(req_ids):
        if req_id not in req_pool_idx:
            raise KeyError(f"missing req_pool_idx for request {req_id!r}")

        row = int(req_pool_idx[req_id])
        before_len = int(seq_lens_before_decode[batch_idx])
        if before_len < 0:
            raise ValueError(f"negative sequence length for request {req_id!r}")

        current_slot = _read_req_to_token_slot(req_to_token, row, before_len)
        slot_mapping.append(current_slot)

        visible_len = before_len + 1 if include_current_token else before_len
        visible_lens.append(visible_len)

        for pos in range(visible_len):
            slot = _read_req_to_token_slot(req_to_token, row, pos)
            if slot == 0:
                raise ValueError(
                    "MLX paged decode metadata references slot 0, which is "
                    f"reserved for padding (request={req_id!r}, row={row}, pos={pos})"
                )
            kv_indices.append(slot)
        kv_indptr.append(len(kv_indices))

    batch_size = len(req_ids)
    return MLXPagedAttentionMetadata(
        kv_indptr=mx.array(kv_indptr, dtype=mx.int32),
        kv_indices=mx.array(kv_indices, dtype=mx.int32),
        qo_indptr=mx.arange(batch_size + 1, dtype=mx.int32),
        slot_mapping=mx.array(slot_mapping, dtype=mx.int32),
        seq_lens=visible_lens,
        max_seq_len=max(visible_lens, default=0),
        total_kv_tokens=len(kv_indices),
    )
