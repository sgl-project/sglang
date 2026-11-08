"""Test-only Python TreeCore implementation with white-box capabilities."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    _InsertPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BaseEvictionResult,
    NodeId,
)
from sglang.test.kits.unified_radix_cache.unified_tree_core_inspection_interface import (
    UnifiedTreeCoreInspectionInterface,
)


class UnifiedTreeCoreInspector(UnifiedTreeCore, UnifiedTreeCoreInspectionInterface):
    """Python TreeCore variant used by the shared backend-conformance tests."""

    def contains_node(self, node_id: NodeId) -> bool:
        """Whether the node id is live in the tree."""
        return node_id in self._node_arena

    def get_parent_node_id(self, node_id: NodeId) -> Optional[NodeId]:
        parent = self.node_by_id(node_id).parent
        return None if parent is None else parent.id

    def get_child_node_ids(self, node_id: NodeId) -> list[NodeId]:
        """The node's child ids."""
        return [child.id for child in self.node_by_id(node_id).children.values()]

    def get_node_key_length(self, node_id: NodeId) -> int:
        """The node's logical radix-key length."""
        key = self.node_by_id(node_id).key
        assert key is not None
        return len(key)

    def get_node_token_ids(self, node_id: NodeId) -> list[int]:
        key = self.node_by_id(node_id).key
        assert key is not None
        return list(key.raw_token_ids())

    def is_node_key_bigram(self, node_id: NodeId) -> bool:
        key = self.node_by_id(node_id).key
        assert key is not None
        return key.is_bigram

    def get_component_host_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        return self.node_by_id(node_id).component_data[component_type].host_value

    def get_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType
    ) -> int:
        return self.node_by_id(node_id).component_data[component_type].lock_ref

    def get_component_host_lock_ref(
        self, node_id: NodeId, component_type: ComponentType
    ) -> int:
        return self.node_by_id(node_id).component_data[component_type].host_lock_ref

    def get_node_hit_count(self, node_id: NodeId) -> int:
        """The node's accumulated match count."""
        return self.node_by_id(node_id).hit_count

    def get_write_through_pending_id(self, node_id: NodeId) -> Optional[int]:
        """The node's pending write-through id, if any."""
        return self.node_by_id(node_id).write_through_pending_id

    def is_external_cache_stored(self, node_id: NodeId) -> bool:
        return self.node_by_id(node_id).external_cache_stored

    def is_node_in_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        lru = self.lru_lists.get(component_type)
        return lru is not None and lru.in_list(self.node_by_id(node_id))

    def is_node_in_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        lru = self.host_lru_lists.get(component_type)
        return lru is not None and lru.in_list(self.node_by_id(node_id))

    @staticmethod
    def _lru_node_ids(lru: UnifiedLRUList) -> list[NodeId]:
        """Return real LRU members from most to least recent."""
        node_ids = []
        node = lru.head.lru_next[lru._pt]
        while node is not lru.tail:
            if node.id in lru.cache:
                node_ids.append(node.id)
            node = node.lru_next[lru._pt]
        return node_ids

    def get_component_device_lru_node_ids(
        self, component_type: ComponentType
    ) -> list[NodeId]:
        lru = self.lru_lists.get(component_type)
        return [] if lru is None else self._lru_node_ids(lru)

    def is_device_evictable_leaf(self, node_id: NodeId) -> bool:
        """Whether the node belongs to the device-evictable leaf set."""
        node = self._node_arena.get(node_id)
        return node is not None and node in self.evictable_device_leaves

    def is_host_evictable_leaf(self, node_id: NodeId) -> bool:
        """Whether the node belongs to the host-evictable leaf set."""
        node = self._node_arena.get(node_id)
        return node is not None and node in self.evictable_host_leaves

    def is_device_leaf(self, node_id: NodeId) -> bool:
        return self._is_device_leaf(self.node_by_id(node_id))

    def get_all_node_ids(self) -> list[NodeId]:
        """All live tree node ids."""
        return [node.id for node in self._collect_all_nodes()]

    def component_protected_size(self, component_type: ComponentType) -> int:
        return self.component_protected_size_.get(component_type, 0)

    def set_node_hash_values(
        self, node_id: NodeId, hash_values: Optional[list[str]]
    ) -> None:
        self.node_by_id(node_id).hash_value = hash_values

    def set_component_device_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        self.node_by_id(node_id).component_data[component_type].value = value

    def set_component_host_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        self.node_by_id(node_id).component_data[component_type].host_value = value

    def set_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType, lock_ref: int
    ) -> None:
        assert lock_ref >= 0
        self.node_by_id(node_id).component_data[component_type].lock_ref = lock_ref

    def remove_node_from_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        self.lru_lists[component_type].remove_node(self.node_by_id(node_id))

    def insert_node_into_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        self.host_lru_lists[component_type].insert_mru(self.node_by_id(node_id))

    def set_component_evictable_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        assert value >= 0
        self.component_evictable_size_[component_type] = value

    def set_component_protected_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        assert value >= 0
        self.component_protected_size_[component_type] = value

    def update_duplicate_tracking(self, node_id: NodeId) -> None:
        self._update_duplicate_tracking(self.node_by_id(node_id))

    def advance_insert_walk_once(self) -> None:
        state = self._ongoing_insert_walk_state
        if state is None:
            raise RuntimeError("no in-flight insert")
        if state.phase is not _InsertPhase.WALK:
            raise RuntimeError("in-flight insert is not in walk phase")
        self._insert_walk_step(state)

    def evict_component(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> BaseEvictionResult:
        result = BaseEvictionResult()
        self._evict_component_and_detach_lru(
            self.node_by_id(node_id),
            self.components_by_type[component_type],
            result.device_frees,
            result.host_frees,
            result.cache_actions,
            target,
            result.tracker,
        )
        return result

    def validate_cascade_evict(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> None:
        node = self.node_by_id(node_id)
        trigger = self.components_by_type[component_type]
        is_leaf = self._is_cascade_evict_leaf(node, target)
        for comp in self.components:
            self._should_cascade_evict_component(node, trigger, comp, target, is_leaf)

    def cleanup_tombstone_ancestors(self, node_id: NodeId) -> BaseEvictionResult:
        result = BaseEvictionResult()
        self._iteratively_delete_tombstone_ancestors(
            self.node_by_id(node_id),
            result.tracker,
            result.device_frees,
            result.host_frees,
            result.cache_actions,
        )
        return result

    def finalize_component_match_result(
        self,
        component_type: ComponentType,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        node_result = result._replace(
            last_device_node=self.node_by_id(result.last_device_node),
            last_host_node=self.node_by_id(result.last_host_node),
            best_match_node=self.node_by_id(result.best_match_node),
        )
        finalized = self.components_by_type[
            component_type
        ].finalize_match_result_in_tree_core(
            result=node_result,
            params=params,
            value_chunks=value_chunks,
            best_value_len=best_value_len,
        )
        return finalized._replace(
            last_device_node=finalized.last_device_node.id,
            last_host_node=finalized.last_host_node.id,
            best_match_node=finalized.best_match_node.id,
        )

    def build_backup_node_ids(
        self, node_id: NodeId, write_back: bool = False
    ) -> list[NodeId]:
        return self._build_backup_kv_action(
            self.node_by_id(node_id), write_back
        ).node_ids
