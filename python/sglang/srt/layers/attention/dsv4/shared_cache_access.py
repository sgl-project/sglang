"""DeepSeek V4 view of the model-neutral owner-sharded cache layout."""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout


class DSV4SharedCacheAccess:
    """The sole attention-side entry point for DSV4 Shared cache behavior."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def publish_writes(self) -> None:
        self._pool.synchronize_shared_writes()

    @property
    def has_prefill_demand_cache(self) -> bool:
        return getattr(self._pool, "prefill_demand_cache", None) is not None

    def flashmla_prefill_demand_kwargs(self, layer_id: int) -> dict[str, Any]:
        cache = self._pool.prefill_demand_cache
        if cache is None:
            raise RuntimeError("DSV4 Shared Prefill demand cache is not allocated")
        view = cache.next_view()
        (
            shared_rank,
            shared_size,
            swa_page_size,
            swa_pages_per_rank,
            extra_page_size,
            extra_pages_per_rank,
        ) = self._pool.get_flashmla_demand_layout(layer_id)
        return {
            "shared_kv_row_cache": view.rows,
            "shared_kv_cache_tags": view.tags,
            "shared_kv_cache_stats": view.stats,
            "shared_kv_cache_epoch": view.epoch,
            "shared_kv_cache_ways": view.ways,
            "shared_kv_cache_direct_slots": (
                self._pool.prefill_demand_cache_direct_slots
            ),
            "shared_kv_rank": shared_rank,
            "shared_kv_size": shared_size,
            "shared_swa_page_size": swa_page_size,
            "shared_swa_pages_per_rank": swa_pages_per_rank,
            "shared_extra_page_size": extra_page_size,
            "shared_extra_pages_per_rank": extra_pages_per_rank,
        }

    def translate_slots(
        self, family: str, slots: torch.Tensor, *, layer_id: int
    ) -> torch.Tensor:
        if family == "swa":
            return self._pool.translate_swa_slots_for_read(slots)
        if family == "extra":
            return self._pool.translate_extra_slots_for_read(layer_id, slots)
        raise ValueError(f"unknown DSV4 Shared slot family: {family}")

    def shared_dequant_params(self, family: str, *, layer_id: int) -> tuple[int, int]:
        if family == "swa":
            return self._pool.get_swa_shared_dequant_params(layer_id)
        if family == "extra":
            return self._pool.get_extra_shared_dequant_params(layer_id)
        raise ValueError(f"unknown DSV4 Shared dequant family: {family}")

    def kv_owner_write_target(
        self, layer_id: int, *, is_indexer: bool
    ) -> tuple[torch.Tensor, int, int]:
        return self._pool.get_compressor_write_info(layer_id, is_indexer=is_indexer)

    @staticmethod
    def compressor_state_layout(state_pool: Any) -> tuple[int, int, int]:
        get_layout = getattr(state_pool, "get_shared_state_layout", None)
        if get_layout is None:
            return 0, 1, 0
        return get_layout()


def get_dsv4_shared_cache_access(pool: Any) -> Optional[DSV4SharedCacheAccess]:
    access = getattr(pool, "shared_cache_access", None)
    if access is None:
        return None
    if not isinstance(access, DSV4SharedCacheAccess):
        raise TypeError("invalid DSV4 Shared cache-access adapter")
    return access


@dataclass(frozen=True)
class DSV4SharedPageLayout:
    owner_layout: OwnerShardedLayout

    @property
    def cp_size(self) -> int:
        return self.owner_layout.cp_size

    @property
    def page_size(self) -> int:
        return self.owner_layout.ownership_granule

    @property
    def pages_per_rank(self) -> int:
        return self.owner_layout.blocks_per_rank

    @property
    def padding_value(self) -> int:
        return -1

    def translate_pages(self, pages: torch.Tensor) -> torch.Tensor:
        slots = pages * self.page_size
        translated = self.owner_layout.physical_rows(slots)
        valid = pages >= 0
        return torch.where(valid, translated // self.page_size, pages)

    def translate_slots(self, slots: torch.Tensor) -> torch.Tensor:
        return self.owner_layout.physical_rows(slots)

    def translate_slots_for_rank(
        self, slots: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        return self.owner_layout.rank_relative_rows(slots, rank=rank)
