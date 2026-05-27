"""Token-label table for Double Sparsity.

The token label table is GPU metadata living next to the KV cache.
Each row is a compressed ``label_dim``-wide projection of one KV slot's
128-d nope K, used by the runtime selector's top-K kernel.

The table is sized by the physical KV slot address space (``max_tokens =
token_to_kv_pool.size + token_to_kv_pool.page_size``), not by the request-row
count (``req_to_token_pool.size`` is much smaller and would cause out-of-bounds
writes for ``out_cache_loc`` values beyond the request count).  When a slot is
freed or evicted by the KV allocator the label entry is left in place;
the per-request range mask (M2) prevents cross-request picks, and the
write hook overwrites the entry before it is read for the new request.

Memory budget for V3.2 at a realistic pool size:

  num_tokens ≈ 262_144  (256 K slots, TP=8, 80 GB HBM)
  worst case = num_layers_local * max_tokens * num_heads_local * label_dim * dtype_bytes
             = 60 * 262_144 * 16 * 16 * 2  (TP=8, label_dim=16, fp16)
             ≈ 8 GB / rank (vs ~480 MB for the page-level table)

Operators must watch the boot-time GB/rank log and ensure the pool size
fits the HBM budget before enabling DS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TokenLabelTable:
    """GPU-resident table of per-(layer, token-slot, head) projection labels.

    Layout: signatures[layer, slot, head, dim] in fp16;
    written[layer, slot] as bool — True once a slot has been populated by
    the write hook.
    """

    num_layers_local: int
    max_tokens: int
    num_heads_local: int
    label_dim: int
    dtype: torch.dtype
    device: torch.device
    signatures: torch.Tensor  # [L, T, H_local, label_dim]
    written: torch.Tensor  # bool [L, T]
    page_size: int

    def bytes_per_rank(self) -> int:
        """Total HBM footprint of the signatures tensor on a single rank."""
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        return (
            self.num_layers_local
            * self.max_tokens
            * self.num_heads_local
            * self.label_dim
            * elem_size
        )


def allocate_token_label_table(
    *,
    num_layers_local: int,
    max_tokens: int,
    num_heads_local: int,
    label_dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> TokenLabelTable:
    """Allocate the token label table on the target device.

    ``max_tokens`` must equal ``token_to_kv_pool.size + token_to_kv_pool.page_size``
    so the table covers all possible ``out_cache_loc`` physical KV slot indices.
    """

    if num_layers_local <= 0 or max_tokens <= 0 or num_heads_local <= 0 or label_dim <= 0:
        raise ValueError(
            "token label table dimensions must all be positive: "
            f"L={num_layers_local} T={max_tokens} H={num_heads_local} D={label_dim}."
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signatures = torch.zeros(
        (num_layers_local, max_tokens, num_heads_local, label_dim),
        dtype=dtype,
        device=device,
    )
    written = torch.zeros(
        (num_layers_local, max_tokens), dtype=torch.bool, device=device
    )

    table = TokenLabelTable(
        num_layers_local=num_layers_local,
        max_tokens=max_tokens,
        num_heads_local=num_heads_local,
        label_dim=label_dim,
        dtype=dtype,
        device=device,
        signatures=signatures,
        written=written,
        page_size=page_size,
    )

    gb = table.bytes_per_rank() / (1024 ** 3)
    logger.info(
        "token_label_table: %.2f GB/rank  L=%d T=%d H=%d D=%d page=%d dtype=%s",
        gb,
        num_layers_local,
        max_tokens,
        num_heads_local,
        label_dim,
        page_size,
        dtype,
    )
    return table


def validate_table_covers_kv_pool(
    table: "TokenLabelTable",
    kv_pool_size: int,
    page_size: int,
) -> None:
    """Raise ValueError if ``table.max_tokens != kv_pool_size + page_size``.

    Called at bind time when reusing a pre-existing table to guard against
    a mis-sized table that would allow ``out_cache_loc`` writes to go
    out-of-bounds or leave un-tracked physical slots.
    """
    expected = kv_pool_size + page_size
    if table.max_tokens != expected:
        raise ValueError(
            f"token_label_table.max_tokens={table.max_tokens} does not match "
            f"kv_pool.size + kv_pool.page_size={expected} "
            f"(size={kv_pool_size}, page_size={page_size}). "
            "The table must be sized from the physical KV slot address space."
        )


def estimate_hbm_bytes(
    *,
    num_layers_local: int,
    max_tokens: int,
    num_heads_local: int,
    label_dim: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Worst-case HBM footprint without allocating."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return num_layers_local * max_tokens * num_heads_local * label_dim * elem_size
