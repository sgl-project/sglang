from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from enum import Enum, IntFlag
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from numpy import float64

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import PoolTransfer

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class ComponentType(int, Enum):
    """Integer enum so that per-node list/tuple storage can be indexed directly."""

    FULL = 0
    SWA = 1
    MAMBA = 2

    def __str__(self) -> str:  # keep human-readable logging
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_mamba(self) -> bool:
        return self == ComponentType.MAMBA


BASE_COMPONENT_TYPE = ComponentType.FULL
_NUM_COMPONENT_TYPES = len(ComponentType)

_LAST_ACCESS_TIME_COUNTER_FLOAT = float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None
    lock_ref: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    host_value: Optional[torch.Tensor] = None
    host_lock_ref: int = 0


class EvictLayer(IntFlag):
    """Which storage layer(s) to evict.  Combinable via bitwise OR."""

    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


class CacheTransferPhase(str, Enum):

    BACKUP_HOST = "backup_host"  # D→H
    LOAD_BACK = "load_back"  # H→D
    BACKUP_STORAGE = "backup_storage"  # H→Storage
    PREFETCH = "prefetch"  # Storage→H


def get_and_increase_time_counter() -> float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        self.cache = cache

    # Subclasses MUST set this as a class attribute (not @property)
    component_type: ComponentType

    def node_has_component_data(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> bool:
        cd = node.component_data[self.component_type]
        if target is EvictLayer.DEVICE:
            return cd.value is not None
        return cd.host_value is not None

    def value_len(self, node: UnifiedTreeNode) -> int:
        value = node.component_data[self.component_type].value
        return len(value) if value is not None else 0

    @abstractmethod
    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        """Return a per-match stateful predicate that decides whether a node
        is a valid match boundary for this component.
        Called once per match_prefix; the returned closure may carry state.
        - Full: always True (every node is valid).
        - SWA: tracks accumulated length since last gap; returns True only
          when the contiguous window reaches swa_sliding_window_size.
        - Mamba: returns True iff the node has mamba component data."""
        ...

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the match result after prefix matching completes.
        - Full & SWA: pass through unchanged.
        - Mamba: performs copy-on-write — allocates a new mamba slot, copies
          the matched node's mamba state into the request pool, and records
          branching_seqlen in result."""
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: torch.Tensor,
        params: InsertParams,
    ) -> int:
        """Called per-node when an insert's key overlaps an existing node.
        Returns the index within value_slice from which this component
        consumed (took ownership of) the underlying KV pool slots.
        Returns prefix_len if nothing was consumed (default).
        _insert_helper uses this to free only the non-consumed duplicate
        portion: value_slice[dup_start:consumed_from]."""
        return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        """Return True to veto leaf creation when the entire new leaf would
        be a tombstone for this component."""
        return False

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        """Finalize component data on the target (leaf) node after the insert
        walk completes. Called once per insert.
        - Full: no-op (full data is handled by _add_new_node).
        - SWA: for new leaves, checks whether the node straddles the SWA
          eviction boundary (swa_evicted_seqlen). If so, splits the node
          via _split_node — the parent becomes a tombstone (no SWA) and the
          child (the deeper portion) receives SWA data. If the entire node
          is within the window, sets SWA directly. If entirely outside,
          leaves SWA as None (tombstone).
        - Mamba: sets the mamba component value from params, inserts into
          mamba LRU list, and increments evictable size. If the node already
          has mamba data, resets its LRU position instead."""
        pass

    @abstractmethod
    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        """Redistribute component data between new_parent and child when a
        node is split. new_parent is the newly created prefix node.
        - Full: copies child's lock_ref to new_parent.
        - SWA: slices (or clones) the swa value for new_parent, copies
          lock_ref and component_uuid metadata, then syncs child's swa
          value with its (now-trimmed) full_value.
        - Mamba: sets new_parent's mamba value to None and lock_ref to 0
          (mamba data stays on the original leaf, not on prefix nodes)."""
        ...

    @abstractmethod
    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """Free this component's KV resources on a node being evicted.

        *target* controls which layer(s) to evict:
          - DEVICE: free device memory and tombstone (value = None).
                    Host data is untouched.
          - HOST:   free host memory (host_value = None).
                    Device data is untouched.
          - ALL:    free both device and host memory.
                    No tombstone — caller will delete the node.

        Returns (device_freed, host_freed) token counts."""
        ...

    def eviction_priority(self, is_leaf: bool) -> int:
        """Eviction priority on this node type. Higher = evicted later.
        When a component is evicted, all other components with equal or
        lower priority on the same node are also cascade-evicted.

        Leaf: all components equal (0) — evicting any cascades to all,
        because the node will be deleted.

        Internal: full=2 > swa=1 > mamba=0.
        Why swa > mamba: SWA data on internal nodes is *path data* —
        the sliding window needs continuous SWA coverage along the path
        from root to the match boundary. E.g. A->B->C->D->E where C
        and E both have mamba and the window covers C->E: if C's mamba
        is evicted, C's SWA must stay so E remains reachable.
        Mamba data, by contrast, is only meaningful at the match
        boundary node; on internal nodes it
        contributes nothing to the path. So SWA is more valuable to
        keep and should be evicted later.

        Cascade consequences:
        - Mamba evict internal: no cascade.
        - SWA evict internal: cascades to Mamba. SWA gone -> SWA
          validator fails -> mamba data is useless (match requires all
          validators to pass).
        - Full evict internal: cascades to SWA + Mamba."""
        return 0

    @abstractmethod
    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        """Drive eviction from this component's LRU list.
        Each component extracts its own request from params, walks its own
        LRU, evicts, and calls cache._cascade_evict for priority cascade.
        Updates the shared tracker with freed amounts for all components.
        - Full: walks leaf LRU, evicts full then cascades entire leaf.
        - Mamba: walks full LRU; tombstones internal nodes (with cascade
          to equal-priority components like swa), cascades leaves to all."""
        ...

    @abstractmethod
    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        """Increment lock_ref for this component, protecting nodes from
        eviction. Updates evictable → protected size on first lock.
        - Full: path-lock — walks from node up to root, incrementing
          lock_ref on every ancestor.
        - SWA: path-lock — walks upward collecting swa values until the
          sliding window is filled; records a component_uuid at the
          boundary for release_component_lock to know where to stop.
        - Mamba: single-node lock — only increments lock_ref on the
          node itself (mamba state is per-leaf, not per-path)."""
        ...

    @abstractmethod
    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        """Decrement lock_ref for this component, un-protecting nodes.
        Updates protected → evictable size when lock_ref drops to 0.
        - Full: path-unlock — walks from node up to root, decrementing
          lock_ref on every ancestor.
        - SWA: path-unlock — walks upward, stopping at the node whose
          component_uuid matches the one recorded during acquire.
        - Mamba: single-node unlock — only decrements lock_ref on the
          node itself."""
        ...

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        """Prepare component-specific data before insert, fill component
        fields in insert_params, return effective cache_len.
        Return None for no truncation opinion (use full length);
        return int >= 0 for effective cache length.
        - Full: no-op, returns None.
        - SWA: sets insert_params.swa_evicted_seqlen on finished; returns None.
        - Mamba: prepares mamba_value (finished from ping-pong buffer,
          unfinished fork from req); returns mamba_last_track_seqlen."""
        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        """Post-cache cleanup for component-specific resources.

        ``is_finished`` — whether the request has finished generation.
        True means the request is complete and its resources can be released;
        ``insert_result`` is None when insert was skipped (cache disabled
        or effective_cache_len <= 0); treat as "no insert happened".
        ``insert_params`` is None only on the disabled path; on early-return
        paths it is still provided so components can free their resources."""
        pass

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self, node: UnifiedTreeNode, phase: CacheTransferPhase, **kw
    ) -> Optional[list[PoolTransfer]]:
        """Build transfer descriptors for this component in the given phase.
        Returns None if the component has nothing to transfer."""
        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
    ) -> None:
        """Post-transfer bookkeeping: store host indices, update LRU, etc."""
        pass

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict from this component's host-side resources.
        Called by HostPoolGroup when the host pool is full.
        Default no-op for components without host storage."""
        pass
