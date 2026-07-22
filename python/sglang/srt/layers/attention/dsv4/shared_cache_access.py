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

    def plan_flashmla_kv_read(
        self, pages: torch.Tensor, *, single_request: bool = False
    ) -> tuple[dict[str | int, torch.Tensor], torch.Tensor]:
        return self._pool.prepare_compressed_pages_for_read(
            pages, single_request=single_request
        )

    def stage_sparse_pages(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return self._pool.stage_compressed_pages_with_indexer_plan(
            layer_id, physical_pages
        )

    def prepare_indexer_pages(
        self, pages: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.prepare_indexer_pages_for_read(pages)

    def stage_indexer_pages(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return self._pool.stage_indexer_pages_with_plan(layer_id, physical_pages)

    def prepare_swa_pages(
        self, slots: torch.Tensor, *, single_request: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pool.prepare_swa_slots_for_read(
            slots, single_request=single_request
        )

    def stage_swa_pages(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return self._pool.stage_swa_slots_with_plan(layer_id, physical_pages)

    def prepare_extra_pages(
        self,
        layer_id: int,
        slots: torch.Tensor,
        *,
        single_request: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if single_request:
            return self._pool.prepare_extra_slots_for_read(
                layer_id, slots, single_request=True
            )
        return self._pool.prepare_extra_slots_for_read(layer_id, slots)

    def stage_extra_pages(
        self, layer_id: int, physical_pages: torch.Tensor
    ) -> torch.Tensor:
        return self._pool.stage_extra_slots_with_plan(layer_id, physical_pages)

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
        return state_pool.get_shared_state_layout()


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
