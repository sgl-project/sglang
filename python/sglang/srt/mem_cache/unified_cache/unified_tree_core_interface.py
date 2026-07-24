"""Interface of the TreeCore: the radix tree mechanism that owns the tree
structure, per-node values, the LRU(s), and bookkeeping. The Controller drives a
TreeCore exclusively through this surface, so an alternative implementation (e.g.
a Rust TreeCore) can satisfy it without subclassing the Python tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence

from sglang.srt.mem_cache.events import KVCacheEventMixin

# Tree node id -- the node handle used outside the TreeCore. The concrete tree
# node is a TreeCore-internal type.
NodeId = int

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import (
        DecLockRefParams,
        DecLockRefResult,
        IncLockRefResult,
        InsertParams,
        InsertResult,
        MatchPrefixParams,
        MatchResult,
    )
    from sglang.srt.mem_cache.hicache_storage import PoolTransfer, PoolTransferResult
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        BackupKV,
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        StorageBackupSpec,
        UnifiedTreeNode,
    )
    from sglang.srt.mem_cache.unified_cache_components import (
        CacheTransferPhase,
        ComponentType,
    )


class UnifiedTreeCoreInterface(KVCacheEventMixin, ABC):
    """Methods the Controller invokes on the Tree Core. The Controller treats the
    Tree Core as opaque behind this surface, which grows as tree operations
    migrate onto the TreeCore. Inherits KVCacheEventMixin for the KV-event API
    (take_events, _record_* recorders)."""

    # ==== Tree-owned state the Controller reads (or, via its facade setters, writes) ====
    page_size: int
    is_eagle: bool
    device: torch.device
    enable_hicache: bool
    enable_storage: bool
    write_through_threshold: int
    is_write_back: bool

    # ==== Tree API ====

    @abstractmethod
    def reset(self) -> None:
        """Drop the entire tree and reinitialize empty state."""
        ...

    @abstractmethod
    def node_by_id(self, node_id: NodeId) -> UnifiedTreeNode:
        """Resolve a NodeId -- the tree-node identity the Controller passes
        across the boundary (e.g. MatchResult fields, lock-ref args) -- back to
        its tree node.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        ...

    @abstractmethod
    def is_backuped(self, node_id: NodeId) -> bool:
        """Whether the node's KV is already backed up to host."""
        ...

    @abstractmethod
    def inc_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        """Bump the reference count on a node's component locks."""
        ...

    @abstractmethod
    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        """Decrease the reference count on a node's component locks."""
        ...

    @abstractmethod
    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Decrease only the SWA (and lower-priority co-located) reference counts,
        collecting freed device slots into ``device_frees``."""
        ...

    # ==== Device eviction (driven step-wise by the Controller's evict()) ====

    @abstractmethod
    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        """Begin a device-eviction walk for one component."""
        ...

    @abstractmethod
    def evict_device_next_node(
        self,
        component_type: ComponentType,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """The next evictable node, or None when the walk is exhausted."""
        ...

    @abstractmethod
    def evict_device_leaf(
        self,
        node_id: NodeId,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        is_write_back: bool,
    ) -> Optional[BackupKV]:
        """Evict a leaf's device value; returns a BackupKV when a D->H backup must
        run (write_back) before the node can be demoted."""
        ...

    @abstractmethod
    def demote(
        self,
        node_id: NodeId,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Demote a backed-up node: drop its device value after a successful backup."""
        ...

    @abstractmethod
    def evict_device_end(self, component_type: ComponentType) -> None:
        """Finish the component's device-eviction walk."""
        ...

    @abstractmethod
    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        """Bump the reference count on a node's host-side component locks."""
        ...

    @abstractmethod
    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """Decrease the reference count on a node's host-side component locks."""
        ...

    @abstractmethod
    def evictable_size(self) -> int: ...

    @abstractmethod
    def protected_size(self) -> int: ...

    @abstractmethod
    def component_evictable_size(self, component_type: ComponentType) -> int:
        """Evictable token count for one component (0 if the component is absent)."""
        ...

    @abstractmethod
    def full_evictable_size(self) -> int: ...

    @abstractmethod
    def full_protected_size(self) -> int: ...

    @abstractmethod
    def swa_evictable_size(self) -> int: ...

    @abstractmethod
    def mamba_evictable_size(self) -> int: ...

    @abstractmethod
    def swa_protected_size(self) -> int: ...

    @abstractmethod
    def mamba_protected_size(self) -> int: ...

    @abstractmethod
    def total_size(self) -> tuple[int, int]:
        """(full_tokens, aux_tokens) summed across the whole tree."""
        ...

    @abstractmethod
    def all_values_flatten(self) -> torch.Tensor: ...

    @abstractmethod
    def all_mamba_values_flatten(self) -> torch.Tensor: ...

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Match a key against the tree; returns device indices + boundary NodeIds."""
        ...

    @property
    @abstractmethod
    def empty_match_result(self) -> MatchResult:
        """A shared empty MatchResult (empty device indices + boundary NodeIds)."""
        ...

    @abstractmethod
    def is_full_device_evicted(self, node_id: NodeId) -> bool:
        """Whether the node's FULL device value has been evicted."""
        ...

    @abstractmethod
    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:
        """Concatenate FULL device values from from_node up to (exclusive) until_node."""
        ...

    @abstractmethod
    def insert(self, params: InsertParams) -> InsertResult:
        """Insert device values to the tree per the provided key."""
        ...

    @abstractmethod
    def drive_host_eviction(
        self,
        component_type: ComponentType,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict a component's host-side resources; no-op if the component is absent."""
        ...

    # ==== HiCache ====

    @abstractmethod
    def set_hicache_enabled(self) -> None:
        """Mark the host tier (HiCache) as wired."""
        ...

    @abstractmethod
    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        """Insert a host-side (backuped) tree path descending from the given node."""
        ...

    @abstractmethod
    def build_backup_spec(
        self, node_id: NodeId
    ) -> tuple[torch.Tensor, dict[ComponentType, list[PoolTransfer]]]:
        """Read a node's device->host backup spec (device value + transfers) now."""
        ...

    @abstractmethod
    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:
        """Gather a node's device->storage backup spec; None if not backuped."""
        ...

    @abstractmethod
    def build_hicache_transfers(
        self,
        component_type: ComponentType,
        node_id: NodeId,
        phase: CacheTransferPhase,
        *,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Build a component's HiCache transfers for the given node and phase."""
        ...

    @abstractmethod
    def build_load_back_spec(
        self, node_id: NodeId, req: Optional[Req] = None
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        """Build the H->D load-back KV transfer plus per-component aux transfers."""
        ...

    @abstractmethod
    def prefetch_anchor_info(self, node_id: NodeId) -> Optional[str]:
        """The anchor node's key extra_key."""
        ...

    @abstractmethod
    def commit_hicache_transfers(
        self,
        node_id: NodeId,
        phase: CacheTransferPhase,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        """Commit each component's HiCache transfers onto the node."""
        ...

    @abstractmethod
    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:
        """Commit a successful backup to the node."""
        ...

    @abstractmethod
    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[CacheAction | ComponentAction]:
        """Commit a successful H->D load-back onto the node; returns any cache actions."""
        ...

    @abstractmethod
    def mark_write_through_pending(self, node_id: NodeId) -> None:
        """Mark a node as having an in-flight write-through backup."""
        ...

    @abstractmethod
    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        """Clear the write-through-pending mark (when it matches ack_id) and record the
        host store event for each acked node."""
        ...

    @abstractmethod
    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        """Store an auxiliary (non-Full) component's device value onto a node."""
        ...

    @abstractmethod
    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        """The component's device value on the node, or None if evicted."""
        ...

    @abstractmethod
    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        """Whether the component's data is device-evicted but host-backed."""
        ...

    # ==== Others ====

    @abstractmethod
    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:
        """Verify tree invariants and raise AssertionError on any violation.

        ongoing_write_through/ongoing_load_back are (id, node_id) pairs for in-flight ops.
        """
        ...

    @abstractmethod
    def pretty_print(self) -> None:
        """Print the tree structure for debugging."""
        ...
