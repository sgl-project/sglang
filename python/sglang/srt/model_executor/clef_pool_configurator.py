"""Reserve KV-aligned prompt states for Clef's trained head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.pool_configurator import (
    DefaultPoolConfigurator,
    MemoryPoolConfig,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


class ClefPoolConfigurator(DefaultPoolConfigurator):
    def __init__(self, kvc: KVCacheConfigurator) -> None:
        super().__init__(kvc)
        self._hidden_bytes_per_token = (
            kvc.model_config.clef_config["hidden_size"] * torch.bfloat16.itemsize
        )
        self._cell_size += self._hidden_bytes_per_token

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        return super().calculate_pool_sizes(
            available_bytes - page_size * self._hidden_bytes_per_token, page_size
        )
