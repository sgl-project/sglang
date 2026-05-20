"""Allocator-owned page signature table.

The page signature table is GPU metadata living next to the KV page table.
Each row is a compressed ``label_dim``-wide projection of one KV page's
keys, used by the runtime selector's top-K kernel. The table is owned by
the KV-page allocator's lifetime; lifecycle hooks (assign / free / evict /
retract) keep ``valid_mask`` in lockstep with the KV-page allocator.

Memory budget (DEC-9 / CMT-12) for V3.2:

  worst case = num_layers_local * max_pages * num_heads_local * label_dim * dtype_bytes
             = 60 * 15_625 * 16 * 16 * 2  (TP=8, label_dim=16, fp16, 1M context, page=64)
             ~ 480 MB / rank.

The kernel that populates entries lives in :mod:`page_signature_write`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class PageSignatureTable:
    """GPU-resident table of per-(layer, page, head) projection signatures.

    Layout: signatures[layer, page, head, dim] in fp16; valid_mask[layer, page]
    as bool. Selection ignores rows where ``valid_mask`` is False.
    """

    num_layers_local: int
    max_pages: int
    num_heads_local: int
    label_dim: int
    dtype: torch.dtype
    device: torch.device
    signatures: torch.Tensor  # [L, P, H_local, label_dim]
    valid_mask: torch.Tensor  # bool [L, P]
    page_size: int

    # Bookkeeping for fast invalidation of pages that move through the
    # eviction / retraction path; not used in the hot path.
    _layer_ids: List[int] = field(default_factory=list)
    _hot_page_per_layer: Dict[int, Optional[int]] = field(default_factory=dict)

    def bytes_per_rank(self) -> int:
        """Total HBM footprint of the table on a single rank."""
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        return (
            self.num_layers_local
            * self.max_pages
            * self.num_heads_local
            * self.label_dim
            * elem_size
        )

    def on_page_assigned(self, layer_id: int, page_id: int) -> None:
        if not (0 <= layer_id < self.num_layers_local):
            raise IndexError(
                f"layer_id={layer_id} out of range [0, {self.num_layers_local})."
            )
        if not (0 <= page_id < self.max_pages):
            raise IndexError(f"page_id={page_id} out of range [0, {self.max_pages}).")
        self.signatures[layer_id, page_id].zero_()
        self.valid_mask[layer_id, page_id] = False  # written by populate kernel

    def on_pages_assigned(self, layer_id: int, page_ids: Sequence[int]) -> None:
        if not page_ids:
            return
        ids = torch.as_tensor(page_ids, dtype=torch.long, device=self.signatures.device)
        self.signatures[layer_id].index_fill_(0, ids, 0)
        self.valid_mask[layer_id].index_fill_(0, ids, False)

    def on_page_freed(self, layer_id: int, page_id: int) -> None:
        self.valid_mask[layer_id, page_id] = False
        if self._hot_page_per_layer.get(layer_id) == page_id:
            self._hot_page_per_layer[layer_id] = None

    def on_pages_freed(self, layer_id: int, page_ids: Sequence[int]) -> None:
        if not page_ids:
            return
        ids = torch.as_tensor(page_ids, dtype=torch.long, device=self.signatures.device)
        self.valid_mask[layer_id].index_fill_(0, ids, False)
        if self._hot_page_per_layer.get(layer_id) in page_ids:
            self._hot_page_per_layer[layer_id] = None

    def on_page_evicted(self, layer_id: int, page_id: int) -> None:
        # Idempotent with free: evicted pages no longer reflect live KV.
        self.on_page_freed(layer_id, page_id)

    def on_page_retracted(self, layer_id: int, page_id: int) -> None:
        # Retract restores a previously freed page; selection-side metadata
        # must be repopulated by the kernel before the page is read again.
        self.valid_mask[layer_id, page_id] = False

    def mark_populated(self, layer_id: int, page_ids: Sequence[int]) -> None:
        """Called by the populate kernel once a page's signature is fresh."""
        if not page_ids:
            return
        ids = torch.as_tensor(page_ids, dtype=torch.long, device=self.signatures.device)
        self.valid_mask[layer_id].index_fill_(0, ids, True)

    def set_hot_page(self, layer_id: int, page_id: Optional[int]) -> None:
        self._hot_page_per_layer[layer_id] = page_id

    def get_hot_page(self, layer_id: int) -> Optional[int]:
        return self._hot_page_per_layer.get(layer_id)


def allocate_page_signature_table(
    *,
    num_layers_local: int,
    max_pages: int,
    num_heads_local: int,
    label_dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
) -> PageSignatureTable:
    """Allocate the page signature table on the target device.

    Per DEC-9 / CMT-12 the default operating point is ``dtype=fp16``,
    ``label_dim=16``, TP-head-sharded such that the per-rank footprint is
    ~480 MB at 1 M context, page=64, TP=8.
    """

    if num_layers_local <= 0 or max_pages <= 0 or num_heads_local <= 0 or label_dim <= 0:
        raise ValueError(
            "page signature table dimensions must all be positive: "
            f"L={num_layers_local} P={max_pages} H={num_heads_local} D={label_dim}."
        )
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signatures = torch.zeros(
        (num_layers_local, max_pages, num_heads_local, label_dim),
        dtype=dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        (num_layers_local, max_pages), dtype=torch.bool, device=device
    )

    table = PageSignatureTable(
        num_layers_local=num_layers_local,
        max_pages=max_pages,
        num_heads_local=num_heads_local,
        label_dim=label_dim,
        dtype=dtype,
        device=device,
        signatures=signatures,
        valid_mask=valid_mask,
        page_size=page_size,
        _layer_ids=list(range(num_layers_local)),
        _hot_page_per_layer={i: None for i in range(num_layers_local)},
    )

    logger.info(
        "Allocated PageSignatureTable: L=%d P=%d H=%d D=%d page=%d dtype=%s -> %.1f MB",
        num_layers_local,
        max_pages,
        num_heads_local,
        label_dim,
        page_size,
        dtype,
        table.bytes_per_rank() / (1024 * 1024),
    )
    return table


def estimate_hbm_bytes(
    *,
    num_layers_local: int,
    max_pages: int,
    num_heads_local: int,
    label_dim: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Worst-case HBM footprint without allocating."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return num_layers_local * max_pages * num_heads_local * label_dim * elem_size
