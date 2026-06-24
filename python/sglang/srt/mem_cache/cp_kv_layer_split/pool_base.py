"""Generic base class for CP KV LayerSplit-aware token-to-KV pools."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from sglang.srt.mem_cache.cp_kv_layer_split.broadcast import (
    BroadcastSlots,
    broadcast_inline,
    get_pynccl_broadcast_comm,
)
from sglang.srt.mem_cache.cp_kv_layer_split.ownership import kv_layer_owner
from sglang.srt.mem_cache.cp_kv_layer_split.staging import (
    StagingBufferManager,
    active_pages_for_indices,
    remap_indices_to_staging,
)

logger = logging.getLogger(__name__)


class CpKvLayerSplitPoolBase:
    """Model-agnostic ownership, staging, and broadcast helpers."""

    def __init__(
        self,
        *,
        cp_rank: int,
        cp_size: int,
        model_num_hidden_layers: int,
        broadcast_slot_kinds: Iterable[str],
        staging_context_len: Optional[int] = None,
        staging_chunked_prefill_size: Optional[int] = None,
        staging_max_prefill_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.model_num_hidden_layers = model_num_hidden_layers
        self._staging_context_len = staging_context_len
        self._staging_chunked_prefill_size = staging_chunked_prefill_size
        self._staging_max_prefill_tokens = staging_max_prefill_tokens
        self._staging = StagingBufferManager()
        self._broadcast_slots = BroadcastSlots(broadcast_slot_kinds, cp_rank=cp_rank)
        super().__init__(**kwargs)

    def _kv_owner_cp_rank(self, layer_id: int) -> int:
        return kv_layer_owner(layer_id, self.cp_size, self.model_num_hidden_layers)

    @staticmethod
    def _pool_num_pages(pool) -> int:
        # Paged KV allocators reserve page 0 for dummy/padded tokens and allocate
        # real cache pages from 1..size//page_size.
        return pool.size // pool.page_size + 1

    def _active_pages_for_indices(
        self,
        indices: torch.Tensor,
        page_size: int,
        max_pages: int,
    ) -> torch.Tensor:
        return active_pages_for_indices(
            indices, page_size, max_pages, get_pynccl_broadcast_comm()
        )

    def _compact_broadcast_for_read(
        self,
        layer_id: int,
        indices: torch.Tensor,
        source: Optional[torch.Tensor],
        staging: torch.Tensor,
        selected_pages: torch.Tensor,
        page_size: int,
        max_pages: int,
        async_kind: Optional[str] = None,
        remap_indices: bool = True,
    ) -> torch.Tensor:
        """Broadcast owner's compact active pages and optionally remap indices."""
        if selected_pages.numel() > staging.shape[0]:
            raise RuntimeError(
                "CP KV LayerSplit staging buffer is smaller than active page set: "
                f"active_pages={selected_pages.numel()}, capacity={staging.shape[0]}"
            )

        active_pages = selected_pages.numel()
        broadcast_pages = max(1, active_pages)
        owner_cp = self._kv_owner_cp_rank(layer_id)
        if self.cp_rank == owner_cp:
            assert source is not None
            if active_pages > 0:
                staging[:active_pages].copy_(source[selected_pages.to(torch.long)])
            else:
                staging[:1].copy_(source[:1])

        pynccl_comm = get_pynccl_broadcast_comm()
        if async_kind is not None:
            self._broadcast_slots.start(
                async_kind, layer_id, staging[:broadcast_pages], owner_cp, pynccl_comm
            )
        else:
            broadcast_inline(staging[:broadcast_pages], owner_cp, pynccl_comm)
        if not remap_indices:
            return indices
        return remap_indices_to_staging(indices, selected_pages, page_size, max_pages)

    def reset_batch_active_pages(self) -> None:
        """Drop per-forward active-page caches before the next forward."""

    def is_any_family_sharded(self) -> bool:
        """True when at least one KV family is sharded across CP ranks."""
        return True


def is_cp_kv_layer_split_pool(pool) -> bool:
    """True when ``pool`` participates in CP KV LayerSplit (any model)."""
    return isinstance(pool, CpKvLayerSplitPoolBase)
