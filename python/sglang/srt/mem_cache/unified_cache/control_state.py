# SPDX-License-Identifier: Apache-2.0
"""Generation and configuration for the cache's single control owner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.managers.kv_control.contracts import ControlError, Identifier, validated

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode


@dataclass(frozen=True)
class CacheControlConfig:
    namespace: str
    max_query_nodes: int = 4096
    max_result_targets: int = 4096
    max_prepared_prefixes: int = 128
    max_concurrent_storage_reads: int = 4
    max_storage_read_inflight_tokens: int = 0
    max_speculative_storage_read_tokens: int = 0
    demand_host_reserve_tokens: int = 0
    storage_demand_priority: bool = False

    def __post_init__(self):
        validated(self.namespace, Identifier)
        for name in (
            "max_prepared_prefixes",
            "max_query_nodes",
            "max_result_targets",
        ):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= 1 << 24
            ):
                raise ValueError(f"{name} must be a positive bounded integer")
        if (
            type(self.max_concurrent_storage_reads) is not int
            or not 1 <= self.max_concurrent_storage_reads <= 64
        ):
            raise ValueError(
                "max_concurrent_storage_reads must be an integer in [1, 64]"
            )
        for name in (
            "max_storage_read_inflight_tokens",
            "max_speculative_storage_read_tokens",
            "demand_host_reserve_tokens",
        ):
            if (
                type(getattr(self, name)) is not int
                or not 0 <= getattr(self, name) <= 1 << 40
            ):
                raise ValueError(f"{name} must be a nonnegative bounded integer")
        if type(self.storage_demand_priority) is not bool:
            raise ValueError("storage_demand_priority must be a boolean")


@dataclass(frozen=True)
class CopySelection:
    node: UnifiedTreeNode
    component: ComponentType
    start: int
    end: int


class CacheControlState:
    def __init__(self, config: CacheControlConfig):
        self.config = config
        self.clock = time.monotonic
        self.generation = 0
        self.host_priorities: dict[int, int] = {}
        self._binding: tuple[str, str] | None = None
        self._binding_generation = -1

    def bind(self, engine_id: str, engine_epoch: str):
        binding = validated(engine_id, Identifier), validated(engine_epoch, Identifier)
        if self._binding is not None:
            if self._binding_generation == self.generation:
                if binding != self._binding:
                    raise ControlError(
                        "busy",
                        "Reset the cache before changing its control owner or epoch",
                    )
            elif engine_epoch == self._binding[1]:
                raise ControlError(
                    "stale_epoch", "Cache reset requires a fresh control epoch"
                )
        self._binding, self._binding_generation = binding, self.generation

    def reset(self):
        self.generation += 1
        self.host_priorities.clear()
