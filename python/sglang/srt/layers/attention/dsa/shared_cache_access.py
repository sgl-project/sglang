"""DSA adapter for model-neutral owner-sharded Shared-KV primitives."""

from __future__ import annotations

from typing import Any, Optional

import torch


class DSASharedCacheAccess:
    """The sole attention-side entry point for DSA Shared-KV behavior.

    Main KV and Indexer both use the Shared-KV Demand Cache architecture. Main
    KV uses a Slot Demand Cache in FlashMLA; Indexer Decode uses a Pool Demand
    Cache before its paged MQA consumer. The pool owns storage, while this
    adapter keeps model/backend code independent of its VMM implementation.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def publish_writes(self) -> None:
        self._pool.synchronize_shared_writes()

    @property
    def uses_shared_indexer(self) -> bool:
        return bool(self._pool.share_indexer)

    def main_owner_translation_args(self) -> dict[str, int]:
        layout = self._pool.main_layout
        if layout is None:
            raise RuntimeError("DSA Shared Main-KV layout is not initialized")
        return {
            "owner_cp_size": layout.cp_size,
            "owner_page_size": layout.page_size,
            "owner_pages_per_rank": layout.pages_per_rank,
        }

    def translate_main_slots(self, slots: torch.Tensor) -> torch.Tensor:
        return self._pool.translate_main_slots(slots)

    def prepare_indexer_pages(self, pages: torch.Tensor) -> torch.Tensor:
        return self._pool.prepare_paged_index_page_table(pages)

    def materialize_indexer_pages(
        self,
        layer_id: int,
        source_pages: torch.Tensor,
        target_pages: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one layer's VMM pages into token-pool page slots."""
        return self._pool.materialize_index_pages(
            layer_id, source_pages, target_pages, seq_lens
        )

    def invalidate_indexer_cache(self) -> None:
        self._pool.invalidate_indexer_cache()


def get_dsa_shared_cache_access(pool: Any) -> Optional[DSASharedCacheAccess]:
    access = getattr(pool, "shared_cache_access", None)
    if access is None:
        return None
    if not isinstance(access, DSASharedCacheAccess):
        raise TypeError("invalid DSA Shared cache-access adapter")
    return access
