"""Double Sparsity token-to-physical-slot adapter.

Translates the DS selector's logical-token output
``(selected_indices, valid_lengths)`` into the physical token index tensor
that the FlashMLA ``flashmla_kv`` sparse path consumes.

The DS selector returns logical sequence positions (0-indexed within the
request's sequence) sorted ascending with ``-1`` padding.  FlashMLA's
sparse path (Option B) consumes ``int32[bs, get_dsa_index_topk]`` physical
token indices — the same values that ``dsa_indexer.py`` uses for
``page_table_1`` (values from ``req_to_token``).

This adapter performs a single ``req_to_token`` gather:
    physical_slots[b, k] = req_to_token[req_pool_indices[b], logical_topk[b, k]]
with ``-1`` preserved for padding slots.

Error tracking: the adapter returns a scalar ``error_count`` alongside
``physical_slots``.  Callers pass this count to the error-containment
counter used by the AC-8 ``error_containment`` check.  Errors occur only
when ``req_pool_indices`` values are out of range for ``req_to_token``.
"""

from __future__ import annotations

from typing import Tuple

import torch


class DSAdapterError(Exception):
    """Base class for Double Sparsity adapter errors."""


def logical_to_physical(
    selected_indices: torch.Tensor,    # int32 [bs, max_top_k] logical positions, -1 padded
    req_pool_indices: torch.Tensor,    # int32 [bs]
    req_to_token: torch.Tensor,        # int32 [num_pools, max_seqlen]
    out: torch.Tensor,                 # int32 [bs, max_top_k]  pre-allocated output
) -> int:
    """Convert logical token positions to physical KV-cache slot indices.

    Writes into the pre-allocated ``out`` tensor (capture-safe, no allocation).
    Returns a scalar error count (rows where ``req_pool_indices`` was out of
    range for ``req_to_token``; typically 0 in production).
    """

    bs, max_top_k = selected_indices.shape
    if bs == 0:
        out.fill_(-1)
        return 0

    # is_valid: True for positions that are not the -1 padding sentinel.
    is_valid = selected_indices >= 0  # [bs, max_top_k]

    # Clamp padding positions to 0 so the gather doesn't OOB on the seqlen dim.
    safe_positions = selected_indices.clamp(min=0)  # [bs, max_top_k]

    # Clamp pool indices to valid range for error containment.
    num_pools = req_to_token.shape[0]
    bad_pool = (req_pool_indices < 0) | (req_pool_indices >= num_pools)  # [bs]
    error_count = int(bad_pool.to(torch.int32).sum().item())
    safe_pool = req_pool_indices.clamp(0, max(num_pools - 1, 0)).long()  # [bs]

    # Gather physical slots: req_to_token[safe_pool[b], safe_positions[b, k]]
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_top_k)       # [bs, max_top_k]
    physical = req_to_token[pool_expanded, safe_positions.long()]       # [bs, max_top_k] int32

    # Restore -1 for padding and for rows with bad pool indices.
    pad_mask = ~is_valid | bad_pool.unsqueeze(1)                        # [bs, max_top_k]
    out.copy_(torch.where(pad_mask, torch.full_like(physical, -1), physical))
    return error_count
