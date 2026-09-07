"""Test-only inspection adapter for the Rust Unified TreeCore."""

from __future__ import annotations

from typing import Optional

import torch
from unified_tree_core_inspection_interface import (
    UnifiedTreeCoreInspectionInterface,
)

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.rust_tree_core.adapter import (
    RustUnifiedTreeCore,
    _fill_evict_result,
    _match_result_from_binding,
    _radix_key_buffer,
)
from sglang.srt.mem_cache.rust_tree_core.extension import load_tree_core_extension
from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BaseEvictionResult,
    NodeId,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="Rust TreeCore test inspector"
)

_inspection_bindings = load_tree_core_extension(inspection=True)


class RustUnifiedTreeCoreInspector(
    RustUnifiedTreeCore, UnifiedTreeCoreInspectionInterface
):
    """Rust TreeCore variant used by the shared backend-conformance tests.

    The production adapter deliberately implements only
    ``UnifiedTreeCoreInterface``. These forwarding methods keep white-box state
    controls in test code while the binding returns snapshots rather than Rust
    iterators across the Python boundary.
    """

    _bindings = _inspection_bindings

    def contains_node(self, node_id: NodeId) -> bool:
        return self._binding.inspect_contains_node(node_id)

    def get_parent_node_id(self, node_id: NodeId) -> Optional[NodeId]:
        return self._binding.inspect_get_parent_node_id(node_id)

    def get_child_node_ids(self, node_id: NodeId) -> list[NodeId]:
        return self._binding.inspect_get_child_node_ids(node_id)

    def get_node_key_length(self, node_id: NodeId) -> int:
        return self._binding.inspect_get_node_key_length(node_id)

    def get_node_token_ids(self, node_id: NodeId) -> list[int]:
        return self._binding.inspect_get_node_token_ids(node_id)

    def is_node_key_bigram(self, node_id: NodeId) -> bool:
        return self._binding.inspect_is_node_key_bigram(node_id)

    def get_component_host_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        return self._binding.inspect_get_component_host_value(
            node_id, int(component_type)
        )

    def get_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType
    ) -> int:
        return self._binding.inspect_get_component_device_lock_ref(
            node_id, int(component_type)
        )

    def get_node_hit_count(self, node_id: NodeId) -> int:
        return self._binding.inspect_get_node_hit_count(node_id)

    def get_write_through_pending_id(self, node_id: NodeId) -> Optional[int]:
        return self._binding.inspect_get_write_through_pending_id(node_id)

    def is_node_in_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        return self._binding.inspect_is_node_in_device_lru(node_id, int(component_type))

    def is_node_in_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        return self._binding.inspect_is_node_in_host_lru(node_id, int(component_type))

    def get_component_device_lru_node_ids(
        self, component_type: ComponentType
    ) -> list[NodeId]:
        return self._binding.inspect_get_component_device_lru_node_ids(
            int(component_type)
        )

    def is_device_evictable_leaf(self, node_id: NodeId) -> bool:
        return self._binding.inspect_is_device_evictable_leaf(node_id)

    def is_host_evictable_leaf(self, node_id: NodeId) -> bool:
        return self._binding.inspect_is_host_evictable_leaf(node_id)

    def is_device_leaf(self, node_id: NodeId) -> bool:
        return self._binding.inspect_is_device_leaf(node_id)

    def get_all_node_ids(self) -> list[NodeId]:
        return self._binding.inspect_get_all_node_ids()

    def component_protected_size(self, component_type: ComponentType) -> int:
        return self._binding.inspect_component_protected_size(int(component_type))

    def set_node_hash_values(
        self, node_id: NodeId, hash_values: Optional[list[str]]
    ) -> None:
        self._binding.inspect_set_node_hash_values(node_id, hash_values)

    def set_component_device_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        self._binding.inspect_set_component_device_value_raw(
            node_id, int(component_type), value
        )

    def set_component_host_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        self._binding.inspect_set_component_host_value_raw(
            node_id, int(component_type), value
        )

    def set_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType, lock_ref: int
    ) -> None:
        assert lock_ref >= 0
        self._binding.inspect_set_component_device_lock_ref(
            node_id, int(component_type), lock_ref
        )

    def remove_node_from_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        self._binding.inspect_remove_node_from_device_lru(node_id, int(component_type))

    def insert_node_into_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        self._binding.inspect_insert_node_into_host_lru(node_id, int(component_type))

    def set_component_evictable_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        assert value >= 0
        self._binding.inspect_set_component_evictable_size(int(component_type), value)

    def set_component_protected_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        assert value >= 0
        self._binding.inspect_set_component_protected_size(int(component_type), value)

    def update_duplicate_tracking(self, node_id: NodeId) -> None:
        self._binding.inspect_update_duplicate_tracking(node_id)

    def advance_insert_walk_once(self) -> None:
        self._binding.inspect_advance_insert_walk_once()

    def evict_component(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> BaseEvictionResult:
        binding_result = self._binding.inspect_evict_component(
            node_id, int(component_type), int(target)
        )
        return _fill_evict_result(binding_result, BaseEvictionResult())

    def validate_cascade_evict(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> None:
        self._binding.inspect_validate_cascade_evict(
            node_id, int(component_type), int(target)
        )

    def cleanup_tombstone_ancestors(self, node_id: NodeId) -> BaseEvictionResult:
        binding_result = self._binding.inspect_cleanup_tombstone_ancestors(node_id)
        return _fill_evict_result(binding_result, BaseEvictionResult())

    def finalize_component_match_result(
        self,
        component_type: ComponentType,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        binding_result = self._binding.inspect_finalize_component_match_result(
            int(component_type),
            result,
            _radix_key_buffer(params.key),
            params.key.extra_key,
            params.key.cache_salt,
            value_chunks,
            best_value_len,
        )
        return _match_result_from_binding(binding_result)._replace(
            cache_protected_len=result.cache_protected_len,
            cache_actions=result.cache_actions,
        )

    def build_backup_node_ids(
        self, node_id: NodeId, write_back: bool = False
    ) -> list[NodeId]:
        return self._binding.inspect_build_backup_node_ids(node_id, write_back)
