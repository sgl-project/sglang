"""Test-only TreeCore surface used by backend-neutral white-box tests.

Production cache and controller code depends only on ``UnifiedTreeCoreInterface``.
A TreeCore backend implements this extended interface only when it opts into the
shared unified radix-cache conformance suite. Some methods intentionally mutate
internal state to construct edge cases; they must not be used by runtime code.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BaseEvictionResult,
    NodeId,
    UnifiedTreeCoreInterface,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="TreeCore inspection test helper"
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
    from sglang.srt.mem_cache.unified_cache.components import ComponentType, EvictLayer


class UnifiedTreeCoreInspectionInterface(UnifiedTreeCoreInterface):
    """Test-only inspection and control contract for shared backend tests."""

    # ==== Read-only inspection ====

    @abstractmethod
    def contains_node(self, node_id: NodeId) -> bool:
        """Whether the node id is live in the tree."""
        ...

    @abstractmethod
    def get_parent_node_id(self, node_id: NodeId) -> Optional[NodeId]:
        """The parent node id, or None for the root."""
        ...

    @abstractmethod
    def get_child_node_ids(self, node_id: NodeId) -> list[NodeId]:
        """The node's child ids."""
        ...

    @abstractmethod
    def get_node_key_length(self, node_id: NodeId) -> int:
        """The node's logical radix-key length."""
        ...

    @abstractmethod
    def get_node_token_ids(self, node_id: NodeId) -> list[int]:
        """The raw token ids spanned by the node's radix key."""
        ...

    @abstractmethod
    def is_node_key_bigram(self, node_id: NodeId) -> bool:
        """Whether the node's radix key uses bigram atoms."""
        ...

    @abstractmethod
    def get_component_host_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        """The component's host value on the node, or None if absent."""
        ...

    @abstractmethod
    def get_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType
    ) -> int:
        """The component's device lock count on the node."""
        ...

    @abstractmethod
    def get_node_hit_count(self, node_id: NodeId) -> int:
        """The node's accumulated match count."""
        ...

    @abstractmethod
    def get_write_through_pending_id(self, node_id: NodeId) -> Optional[int]:
        """The node's pending write-through id, if any."""
        ...

    @abstractmethod
    def is_node_in_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        """Whether the node belongs to the component's device LRU."""
        ...

    @abstractmethod
    def is_node_in_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        """Whether the node belongs to the component's host LRU."""
        ...

    @abstractmethod
    def get_component_device_lru_node_ids(
        self, component_type: ComponentType
    ) -> list[NodeId]:
        """The component's device LRU members from most to least recent."""
        ...

    @abstractmethod
    def is_device_evictable_leaf(self, node_id: NodeId) -> bool:
        """Whether the node belongs to the device-evictable leaf set."""
        ...

    @abstractmethod
    def is_host_evictable_leaf(self, node_id: NodeId) -> bool:
        """Whether the node belongs to the host-evictable leaf set."""
        ...

    @abstractmethod
    def is_device_leaf(self, node_id: NodeId) -> bool:
        """Whether the node has no device-resident descendants."""
        ...

    @abstractmethod
    def get_all_node_ids(self) -> list[NodeId]:
        """All live tree node ids."""
        ...

    @abstractmethod
    def component_protected_size(self, component_type: ComponentType) -> int:
        """Protected token count for one component (0 if the component is absent)."""
        ...

    # ==== White-box state controls ====

    @abstractmethod
    def set_node_hash_values(
        self, node_id: NodeId, hash_values: Optional[list[str]]
    ) -> None:
        """Replace the node's page-hash field."""
        ...

    @abstractmethod
    def set_component_device_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        """Replace the device-value field without updating tree bookkeeping."""
        ...

    @abstractmethod
    def set_component_host_value_raw(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        value: Optional[torch.Tensor],
    ) -> None:
        """Replace the host-value field without updating tree bookkeeping."""
        ...

    @abstractmethod
    def set_component_device_lock_ref(
        self, node_id: NodeId, component_type: ComponentType, lock_ref: int
    ) -> None:
        """Replace the component's device lock count."""
        ...

    @abstractmethod
    def remove_node_from_device_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        """Remove the node from the component's device LRU."""
        ...

    @abstractmethod
    def insert_node_into_host_lru(
        self, node_id: NodeId, component_type: ComponentType
    ) -> None:
        """Insert the node as the component's most-recent host entry."""
        ...

    @abstractmethod
    def set_component_evictable_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        """Replace the component's evictable device-token count."""
        ...

    @abstractmethod
    def set_component_protected_size(
        self, component_type: ComponentType, value: int
    ) -> None:
        """Replace the component's protected device-token count."""
        ...

    @abstractmethod
    def update_duplicate_tracking(self, node_id: NodeId) -> None:
        """Refresh duplicate-host tracking for the node."""
        ...

    # ==== Targeted white-box operations ====

    @abstractmethod
    def evict_component(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> BaseEvictionResult:
        """Evict one component layer from a node and detach its LRU entry."""
        ...

    @abstractmethod
    def validate_cascade_evict(
        self,
        node_id: NodeId,
        component_type: ComponentType,
        target: EvictLayer,
    ) -> None:
        """Validate the locks for a component-triggered cascade eviction."""
        ...

    @abstractmethod
    def cleanup_tombstone_ancestors(self, node_id: NodeId) -> BaseEvictionResult:
        """Delete childless tombstone ancestors until a live or locked node is reached."""
        ...

    @abstractmethod
    def finalize_component_match_result(
        self,
        component_type: ComponentType,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """Run one component's match finalizer with NodeId boundaries."""
        ...

    @abstractmethod
    def build_backup_node_ids(
        self, node_id: NodeId, write_back: bool = False
    ) -> list[NodeId]:
        """Build the ordered node list for a device-to-host backup."""
        ...
