"""The radix tree mechanism.

``UnifiedTreeCore`` owns the tree's member-var state -- the tree structure (root
node), the per-component LRU lists, the per-component size counters, the
evictable device/host leaf sets, and the empty match result -- plus ``reset()``.
It also defines the tree's building blocks (``UnifiedTreeNode``,
``UnifiedLRUList``).

The cache builds the component drivers and passes them in; the tree holds them
(``self.components_by_type``) to drive their tree-level hooks. The components hold the
cache for cache-level logic, but the TreeCore itself never touches it.
"""

# TODO(Jialin): Split this file into smaller cohesive modules (e.g. tree
# building blocks, insert/match walks, eviction drivers).

from __future__ import annotations

import logging
import sys
from array import array
from collections import defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Sequence

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeDeviceKV,
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    DemoteResult,
    DriveHostEvictionResult,
    DropSubtreeNoHostResult,
    EvictDeviceLeafResult,
    EvictDeviceNextNodeResult,
    InsertStepResult,
    NodeId,
    RadixCacheWalkResult,
    UnifiedTreeCoreInterface,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class StorageBackupSpec(NamedTuple):
    """A node's device->storage backup spec, gathered tree-side."""

    host_value: torch.Tensor
    token_ids: array
    hash_value: list[str]
    prefix_keys: Optional[list[str]]
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        # Plain dict (not defaultdict): a missing-key read must raise, never
        # silently mint an unregistered node outside the TreeCore arena.
        self.children: dict[Any, UnifiedTreeNode] = {}
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.component_types = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        self.write_through_pending_id: Optional[int] = None

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
        return (
            self.parent is not None
            and self.component_data[ComponentType.FULL].value is None
        )

    def __lt__(self, other: UnifiedTreeNode):
        return self.last_access_time < other.last_access_time

    def get_last_hash_value(self) -> Optional[str]:
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: UnifiedTreeNode) -> list[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value


class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
        is_referenced: Optional[Callable[[UnifiedTreeNode], bool]] = None,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}
        # Session partition: [head .. mid) holds session-referenced nodes and
        # (mid .. tail] unreferenced ones, so evictions directly walks (tail -> head).
        self._is_referenced = is_referenced
        self.mid: Optional[UnifiedTreeNode] = None
        self.cursor: Optional[UnifiedTreeNode] = None
        if is_referenced is not None:
            self.mid = UnifiedTreeNode(tree_components)
            self.cursor = UnifiedTreeNode(tree_components)
            for ct in tree_components:
                for node in (self.mid, self.cursor):
                    node.component_data[ct].lock_ref = 1
                    node.component_data[ct].host_lock_ref = 1
            self._add_node_after(self.head, self.mid)

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        if self._is_referenced is None or self._is_referenced(node):
            self._add_node_after(self.head, node)
        else:
            self._add_node_after(self.mid, node)

    def _remove_node(self, node: UnifiedTreeNode):
        pt = self._pt
        node.lru_prev[pt].lru_next[pt] = node.lru_next[pt]
        node.lru_next[pt].lru_prev[pt] = node.lru_prev[pt]
        # Clear self pointers to break reference cycles among evicted nodes.
        node.lru_prev[pt] = None
        node.lru_next[pt] = None

    def insert_mru(self, node: UnifiedTreeNode):
        assert node.id not in self.cache
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        del self.cache[node.id]
        self._remove_node(node)

    def reset_node_mru(self, node: UnifiedTreeNode):
        assert node.id in self.cache
        self._remove_node(node)
        self._add_node(node)

    def cursor_begin(self):
        if self.cursor.lru_prev[self._pt] is not None:
            self._remove_node(self.cursor)
        self._add_node_after(self.tail.lru_prev[self._pt], self.cursor)

    def cursor_next(self, *, host_lock: bool = False):
        pt = self._pt
        ct = self.component_type
        x = self.cursor.lru_prev[pt]
        while x is not self.head:
            cd = x.component_data[ct]
            if (cd.host_lock_ref if host_lock else cd.lock_ref) == 0:
                break
            x = x.lru_prev[pt]
        if x is self.head:
            return None
        self._remove_node(self.cursor)
        self._add_node_after(x.lru_prev[pt], self.cursor)
        return x

    def cursor_end(self):
        """Unlink the walk-cursor sentinel after a walk."""
        self._remove_node(self.cursor)

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        is_referenced = self._is_referenced
        prev_ref = self.head
        prev_unref = self.head if self.mid is None else self.mid
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                part = is_referenced is None or is_referenced(node)
                self._remove_node(node)
                if part:
                    self._add_node_after(prev_ref, node)
                    prev_ref = node
                else:
                    self._add_node_after(prev_unref, node)
                    prev_unref = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        is_referenced = self._is_referenced
        prev_ref = self.head
        prev_unref = self.head if self.mid is None else self.mid
        accumulated = 0
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                part = is_referenced is None or is_referenced(node)
                self._remove_node(node)
                if part:
                    self._add_node_after(prev_ref, node)
                    prev_ref = node
                else:
                    self._add_node_after(prev_unref, node)
                    prev_unref = node
            accumulated += len(node.key)
            node = node.parent

    def in_list(self, node: Optional[UnifiedTreeNode]):
        return node is not None and node.id in self.cache

    def get_prev_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].lock_ref > 0 or len(x.children) > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_prev_no_host_lock(self, node: UnifiedTreeNode, check_id: bool = True):
        """Host-LRU walker: skip nodes whose component host_lock_ref > 0."""
        if check_id:
            assert node.id in self.cache
        pt = self._pt
        ct = self.component_type
        x = node.lru_prev[pt]
        while x.component_data[ct].host_lock_ref > 0:
            x = x.lru_prev[pt]
        if x == self.head:
            return None
        return x

    def get_lru_no_lock(self):
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self):
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_lru_no_host_lock(self):
        return self.get_prev_no_host_lock(self.tail, check_id=False)


# WALK (one node per step) -> COMMIT (leaf + commit hooks) -> TAIL (refresh + backup).
class _InsertPhase(Enum):
    WALK = auto()
    COMMIT = auto()
    TAIL = auto()


class _InsertWalkState(msgspec.Struct):
    """In-flight resumable-insert state persisted across step barriers."""

    phase: _InsertPhase
    node: UnifiedTreeNode
    key: RadixKey
    value: torch.Tensor
    params: InsertParams
    priority: int
    total_prefix_length: int = 0
    is_new_leaf: bool = False
    target_node: Optional[UnifiedTreeNode] = None
    result: Optional[InsertResult] = None
    # Emitted actions awaiting the next barrier flush (or the final step).
    pending_actions: list[CacheAction | ComponentAction] = []


class UnifiedTreeCore(UnifiedTreeCoreInterface):
    """The radix tree mechanism: owns the tree structure, per-node values, the
    per-component LRUs, the size/leaf bookkeeping, and the component drivers,
    plus ``reset()``.

    TODO(Jialin): the tree operations still live on ``UnifiedRadixCache`` and
    reach this state through its proxy properties; they migrate onto this class
    as the TreeCore split completes.
    """

    def __init__(
        self,
        params: CacheInitParams,
        components: dict[ComponentType, TreeComponent],
    ):
        self.page_size = params.page_size
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components
        self.enable_hicache = False
        self.enable_storage = False
        self.write_through_threshold = 256
        self.is_write_back = False
        self.has_swa_host_pool = False
        self.enable_session_radix_cache = params.enable_session_radix_cache
        self.eviction_strategy = get_eviction_strategy(params.eviction_policy.lower())

        # ``device`` is derived from the construction-time allocator; the
        # allocator/pool themselves are owned by the cache, not the tree.
        if params.token_to_kv_pool_allocator:
            self.device = params.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # The cache builds and owns the component drivers; the tree references
        # them to drive their tree-level hooks, attaching itself as tree_core.
        assert components
        self.component_types = tuple(components.keys())
        self.components_by_type: dict[ComponentType, TreeComponent] = components
        for component in components.values():
            component.tree_core = self
        self.components: tuple[TreeComponent, ...] = tuple(
            self.components_by_type.values()
        )

        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []

        self.reset()

    # ==== Tree API ====

    def _session_lru_predicate(self, ct: ComponentType):
        if not self.enable_session_radix_cache or ct is ComponentType.FULL:
            return None
        return lambda node: node.component_data[ct].session_ref > 0

    def reset(self) -> None:
        """Rebuild the root, LRUs, sizes, evictable-leaf sets, and the empty
        match result."""
        # Maintains the NodeId -> active tree node mapping.
        self._node_arena: dict[NodeId, UnifiedTreeNode] = {}

        # The single in-flight resumable insert, if suspended at a barrier.
        self._ongoing_insert_walk_state: Optional[_InsertWalkState] = None

        self.root_node = self._new_node()
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        for ct in self.component_types:
            self.root_node.component_data[ct].lock_ref = 1

        self.component_evictable_size_ = {ct: 0 for ct in self.component_types}
        self.component_protected_size_ = {ct: 0 for ct in self.component_types}

        self.lru_lists = {
            ct: UnifiedLRUList(
                ct, self.component_types, is_referenced=self._session_lru_predicate(ct)
            )
            for ct in self.component_types
        }

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.host_lru_lists = {
            ct: UnifiedLRUList(
                ct,
                self.component_types,
                use_host_ptr=True,
                is_referenced=self._session_lru_predicate(ct),
            )
            for ct in self.component_types
        }

        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node.id,
            last_host_node=self.root_node.id,
            best_match_node=self.root_node.id,
            cache_actions=[],
        )

    def node_by_id(self, node_id: NodeId) -> UnifiedTreeNode:
        """Resolve a NodeId back to its tree node.

        TODO(Jialin): Make TreeCore-internal after the Unified Radix Cache split.
        """
        return self._node_arena[node_id]

    def is_backuped(self, node_id: NodeId) -> bool:
        """Whether the node's KV is already backed up to host."""
        return self._node_arena[node_id].backuped

    def is_root(self, node_id: NodeId) -> bool:
        """Whether the node is the tree root."""
        return self.node_by_id(node_id) is self.root_node

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        """The node's last page hash, or None when it was never hashed."""
        return self.node_by_id(node_id).get_last_hash_value()

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        """The hash chain of the node's ancestors, in root-to-parent order."""
        node = self.node_by_id(node_id)
        return node.get_prefix_hash_values(node.parent)

    def _new_node(self, priority: int = 0) -> UnifiedTreeNode:
        """Create and register a tree node in the arena."""
        node = UnifiedTreeNode(self.component_types, priority=priority)
        self._register_node(node)
        return node

    def _register_node(self, node: UnifiedTreeNode) -> None:
        """Register a tree node in the arena."""
        self._node_arena[node.id] = node

    def _unregister_node(self, node: UnifiedTreeNode) -> None:
        """Drop a tree node from the arena."""
        self._node_arena.pop(node.id, None)

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        node = self.node_by_id(node_id)
        result = IncLockRefResult()
        for component in self.components:
            if component.component_type in skip_lock_components:
                # Leave this component's value evictable and record every
                # non-root node (incl tombstones) so the matching dec skips a
                # lock we never took, which may be another req's on a shared node.
                if node is not self.root_node:
                    result.skip_lock_node_ids.setdefault(
                        component.component_type, set()
                    ).add(node.id)
                continue
            result = component.acquire_component_lock(node=node, result=result)
        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        node = self.node_by_id(node_id)
        for component in self.components:
            if skip_swa and component.component_type == ComponentType.SWA:
                continue
            component.release_component_lock(node=node, params=params)
        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        """Early-release the SWA portion of a request's tree lock, plus any
        strictly-lower-priority locks (e.g. Mamba) co-located on the node."""
        result = DecSwaLockOnlyResult()
        node = self.node_by_id(node_id)
        swa_component = self.components_by_type.get(ComponentType.SWA)
        if swa_component is None:
            return result
        swa_component.release_window_lock(
            node, swa_uuid_for_lock, result.device_frees, result.host_frees
        )

        # Drop strictly-lower-priority locks (e.g. Mamba) co-located on the node,
        # honoring skip ids so we don't drop a lock a partial inc never took
        # (matters for FULL+SWA+MAMBA models, e.g. Inkling).
        swa_priority = swa_component.eviction_priority(is_leaf=False)
        dec_params = DecLockRefParams(
            swa_uuid_for_lock=swa_uuid_for_lock,
            skip_lock_node_ids=skip_lock_node_ids or {},
        )
        for comp in self.components:
            if comp.eviction_priority(is_leaf=False) < swa_priority:
                comp.release_component_lock(node, dec_params)
        return result

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        node = self.node_by_id(node_id)
        result = IncLockRefResult()
        for component in self.components:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )
        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        node = self.node_by_id(node_id)
        for component in self.components:
            component.release_component_lock(node=node, params=params, lock_host=True)
        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )

    def _match_prefix_helper(self, key: RadixKey) -> tuple[
        list[torch.Tensor],
        UnifiedTreeNode,
        UnifiedTreeNode,
        int,
        int,
        Optional[CacheAction | ComponentAction],
    ]:
        # Non-HiCache mode has only device-resident matches, so the scheduler
        # device anchor follows the best match. In HiCache mode, host-backed
        # nodes can also match, so we separately track the best device-resident
        # match for scheduler prefix indices and locking.
        node = self.root_node
        child_key = key.child_key(self.page_size)
        value: list[torch.Tensor] = []
        best_match_node = node
        best_match_device_node = node
        best_match_device_value_len = 0
        full_kv_hit_length = 0
        action: Optional[CacheAction | ComponentAction] = None
        separate_device_match = self.enable_hicache
        if separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in self.components
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self.components
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self.components
            )

        def _all_valid(validators, node):
            return all([v(node) for v in validators])

        def _update_best_if_valid(node):
            nonlocal best_match_node
            nonlocal best_match_device_value_len, best_match_device_node
            matched = _all_valid(validators, node)
            if matched:
                best_match_node = node

            if not separate_device_match:
                if matched:
                    best_match_device_value_len = len(value)
                    best_match_device_node = node
                return
            if _all_valid(device_validators, node):
                best_match_device_value_len = len(value)
                best_match_device_node = node

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # HiCache: dead node (evicted + not backuped) — stop traversal
            if child.evicted and not child.backuped:
                break

            prefix_len = child.key.match(key, page_size=self.page_size)
            full_kv_hit_length += prefix_len
            if prefix_len < len(child.key):
                node, action = self._split_node(child.key, child, prefix_len)
                if not node.evicted:
                    value.append(node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(node)
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        return (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
            full_kv_hit_length,
            action,
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
        full_kv_hit_length: int,
        action: Optional[CacheAction | ComponentAction],
    ) -> MatchResult:
        node_update = best_match_node
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue  # Full uses last_access_time, not LRU
            comp.refresh_lru(LRURefreshPhase.MATCH_END, node_update, self.root_node)

        cur_time = get_and_increase_time_counter()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        # last_host_node will be used as the starting node for the subsequent
        # `prefetch_from_storage` flow. We directly use best_match_node here,
        # because best_match_node represents the node where all components
        # have reached consensus on both device & host availability.
        last_host_node = (
            best_match_node if self.enable_hicache else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
            full_kv_hit_length=full_kv_hit_length,
        )

        for component in self.components:
            result = component.finalize_match_result_in_tree_core(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        # Expose only NodeIds outside TreeCore.
        return result._replace(
            last_device_node=result.last_device_node.id,
            last_host_node=result.last_host_node.id,
            best_match_node=result.best_match_node.id,
            cache_actions=[action] if action is not None else [],
        )

    @property
    def empty_match_result(self) -> MatchResult:
        """A shared empty MatchResult (empty device indices + boundary NodeIds)."""
        return self._empty_match_result

    def is_full_device_evicted(self, node_id: NodeId) -> bool:
        """Whether the node's FULL device value has been evicted."""
        return self.node_by_id(node_id).evicted

    def collect_full_device_indices(
        self, from_node_id: NodeId, until_node_id: NodeId
    ) -> torch.Tensor:
        """Concatenate FULL device values from ``from_node`` up to (exclusive)
        ``until_node``, in root order; empty tensor if the path is empty."""
        until_node = self.node_by_id(until_node_id)
        prefix_chunks: list[torch.Tensor] = []
        node = self.node_by_id(from_node_id)
        while node is not until_node:
            value = node.component_data[BASE_COMPONENT_TYPE].value
            assert value is not None
            prefix_chunks.append(value)
            node = node.parent
        if not prefix_chunks:
            return self._empty_match_result.device_indices
        prefix_chunks.reverse()
        return torch.cat(prefix_chunks)

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            for comp in self.components:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def _inc_hit_count_and_check(
        self, node: UnifiedTreeNode, chunked: bool = False
    ) -> bool:
        """Increment hit count; check whether a write backup should be fired."""
        if node.evicted or chunked:
            return False
        if self.is_write_back:
            return False
        node.hit_count += 1
        return (
            self.enable_hicache
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        )

    def begin_insert(self, params: InsertParams) -> InsertStepResult:
        """Start the insert, running to its first barrier or completion."""
        # Insert walks are single-flight; a live walk means re-entrancy.
        assert self._ongoing_insert_walk_state is None, "concurrent insert walks"
        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(self.root_node)
        self.root_node.priority = max(self.root_node.priority, priority)
        if len(key) == 0:
            return InsertStepResult(
                actions=[],
                result=InsertResult(
                    prefix_len=0,
                    mamba_exist=True,
                    last_device_node=self.root_node.id,
                ),
            )

        self._ongoing_insert_walk_state = _InsertWalkState(
            phase=_InsertPhase.WALK,
            node=self.root_node,
            key=key,
            value=value,
            params=params,
            priority=priority,
        )
        return self._advance_insert()

    def resume_insert(self) -> InsertStepResult:
        """Continue the suspended insert after its step actions were executed."""
        assert self._ongoing_insert_walk_state is not None, "no in-flight insert"
        return self._advance_insert()

    def has_ongoing_insert(self) -> bool:
        """Whether an insert walk is suspended at a barrier."""
        return self._ongoing_insert_walk_state is not None

    def end_insert(self) -> list[CacheAction | ComponentAction]:
        """Finish the insert (idempotent); returns still-pending actions to drain."""
        state = self._ongoing_insert_walk_state
        self._ongoing_insert_walk_state = None
        return state.pending_actions if state is not None else []

    def _advance_insert(self) -> InsertStepResult:
        """Run the in-flight insert to its next barrier or to completion."""
        state = self._ongoing_insert_walk_state
        while True:
            flushed_len = len(state.pending_actions)
            if state.phase is _InsertPhase.WALK:
                self._insert_walk_step(state)
            elif state.phase is _InsertPhase.COMMIT:
                self._insert_commit_step(state)
            elif state.phase is _InsertPhase.TAIL:
                self._insert_tail_step(state)
                self._ongoing_insert_walk_state = None
                return InsertStepResult(
                    actions=state.pending_actions, result=state.result
                )
            else:
                raise AssertionError(f"unsupported insert phase: {state.phase}")
            new_actions = state.pending_actions[flushed_len:]
            # Suspend only when a step emitted a non-deferrable action.
            if new_actions and not all(map(self._is_deferrable_action, new_actions)):
                flushed = state.pending_actions
                state.pending_actions = []
                return InsertStepResult(actions=flushed)

    @staticmethod
    def _is_deferrable_action(action: CacheAction | ComponentAction) -> bool:
        """Fire-and-forget actions safe to batch until the next barrier."""
        return isinstance(action, (FreeDeviceKV, ReplaceWriteThroughOnNodeSplit))

    def _insert_walk_step(self, state: _InsertWalkState) -> None:
        """Process one walked node, appending its barrier actions to the state."""
        key = state.key
        child_key = key.child_key(self.page_size) if len(key) else None
        if child_key not in state.node.children:
            state.phase = _InsertPhase.COMMIT
            return
        step_actions = state.pending_actions
        node = state.node.children[child_key]
        self._touch_node(node)
        prefix_len = node.key.match(key, page_size=self.page_size)
        if prefix_len < len(node.key):
            node, action = self._split_node(node.key, node, prefix_len)
            if action is not None:
                step_actions.append(action)
        node.priority = max(node.priority, state.priority)

        if node.evicted:
            self._unevict_node_on_insert(node, state.value[:prefix_len])
            # FULL was restored from the request's fresh KV. Aux
            # components (e.g. SWA) may still hold tombstones and need
            # to rebuild their value from the same slice.
            for component in self.components:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.recover_after_unevict(
                    node=node,
                    prefix_len=prefix_len,
                    total_prefix_len=state.total_prefix_length,
                    params=state.params,
                    cache_actions=step_actions,
                )
        else:
            value_slice = state.value[:prefix_len]
            consumed_from = prefix_len
            # Let each component claim ownership of overlapping KV slots
            for component in self.components:
                comp_consumed_from = component.update_component_on_insert_overlap(
                    node=node,
                    prefix_len=prefix_len,
                    total_prefix_len=state.total_prefix_length,
                    value_slice=value_slice,
                    params=state.params,
                    cache_actions=step_actions,
                )
                consumed_from = min(consumed_from, comp_consumed_from)

            dup_start = max(0, state.params.prev_prefix_len - state.total_prefix_length)
            if dup_start < consumed_from:
                step_actions.append(
                    FreeDeviceKV([value_slice[dup_start:consumed_from]])
                )

        if self._inc_hit_count_and_check(node, state.params.chunked):
            step_actions.append(self._build_backup_kv_action(node))
        state.node = node
        state.total_prefix_length += prefix_len
        state.key = key[prefix_len:]
        state.value = state.value[prefix_len:]

    def _insert_commit_step(self, state: _InsertWalkState) -> None:
        """Create the tail leaf and run the component commit hooks."""
        # Create new leaf for remaining suffix. A leaf survives on its Full
        # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        # window). Materialize it anyway so the Full KV stays cacheable.
        if len(state.key):
            state.target_node = self._add_new_node(
                state.node, state.key, state.value, priority=state.priority
            )
            state.is_new_leaf = True
        else:
            state.target_node = state.node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        # All hooks run before their emitted actions execute; an action failure
        # fail-stops the process, so partial-commit state is never observed.
        state.result = InsertResult(
            prefix_len=state.total_prefix_length,
            last_device_node=state.target_node.id,
        )
        for component in self.components:
            component.commit_insert_component_data(
                node=state.target_node,
                is_new_leaf=state.is_new_leaf,
                params=state.params,
                result=state.result,
                cache_actions=state.pending_actions,
            )
        state.phase = _InsertPhase.TAIL

    def _insert_tail_step(self, state: _InsertWalkState) -> None:
        """Refresh the LRUs and append the terminal new-leaf backup."""
        if state.target_node is not self.root_node:
            for component in self.components:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, state.target_node, self.root_node
                )

        if state.is_new_leaf and self._inc_hit_count_and_check(
            state.target_node, state.params.chunked
        ):
            state.pending_actions.append(
                self._build_backup_kv_action(state.target_node)
            )

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> tuple[UnifiedTreeNode, Optional[CacheAction | ComponentAction]]:
        new_node = self._new_node(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count
        new_node.creation_time = child.creation_time

        self._for_each_component_lru(child, UnifiedLRUList.remove_node)

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        for component in self.components:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # A split of a backuped node tells the cache to fix its publish list.
        action: Optional[CacheAction | ComponentAction] = None
        if child.write_through_pending_id is not None:
            ack_id = child.write_through_pending_id
            new_node.write_through_pending_id = ack_id
            action = ReplaceWriteThroughOnNodeSplit(
                ack_id=ack_id,
                old_node_id=child.id,
                new_node_id=new_node.id,
                new_child_node_id=child.id,
            )

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node, action

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        new_node = self._new_node(priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.clone()
        parent.children[key.child_key(self.page_size)] = new_node
        self.component_evictable_size_[BASE_COMPONENT_TYPE] += len(value)
        if self.enable_storage:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        self._record_store_event(new_node)
        return new_node

    def _unevict_node_on_insert(
        self, node: UnifiedTreeNode, fresh_value: torch.Tensor
    ) -> None:
        """Restore an evicted node's Full device value from fresh KV indices
        during insert."""
        ct = BASE_COMPONENT_TYPE
        cd = node.component_data[ct]
        assert cd.value is None
        n = len(fresh_value)
        cd.value = fresh_value.clone()
        self.component_evictable_size_[ct] += n
        self._update_evictable_leaf_sets(node)
        if node.parent is not None:
            self._update_evictable_leaf_sets(node.parent)
        self._record_store_event(node, medium=StorageMedium.GPU)

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
        else:
            self.evictable_host_leaves.discard(node)

    def _for_each_component_lru(
        self,
        node: UnifiedTreeNode,
        lru_op,
        target: EvictLayer = EvictLayer.DEVICE,
        skip_existing: bool = False,
    ):
        """Apply lru_op to each aux component's LRU that has data on this node.
        If skip_existing=True, skip components already in the target LRU list."""
        lru_dict = self.host_lru_lists if target is EvictLayer.HOST else self.lru_lists
        for ct in self.component_types:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        """Begin a component's device-eviction walk for up to request_cnt tokens."""
        self.components_by_type[component_type].evict_device_start(request_cnt)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        """Return the next device leaf to evict for a component, or None when done."""
        result = EvictDeviceNextNodeResult()
        # The walk reads running totals for its doneness check; the result
        # carries only this step's delta.
        updated_tracker = defaultdict(int, tracker)
        result.node_id = self.components_by_type[component_type].evict_device_next_node(
            updated_tracker, result.device_frees, result.host_frees
        )
        for ct, n in updated_tracker.items():
            delta = n - tracker.get(ct, 0)
            if delta:
                result.tracker[ct] = delta
        return result

    def evict_device_end(self, component_type: ComponentType) -> None:
        """Finish a component's device-eviction walk."""
        self.components_by_type[component_type].evict_device_end()

    def evict_device_leaf(
        self, node_id: NodeId, is_write_back: bool
    ) -> EvictDeviceLeafResult:
        """Evict one device leaf (demote if backuped, delete if write-through);
        for an unbacked write-back node, the result carries the BackupKV for
        the cache to execute and then demote."""
        result = EvictDeviceLeafResult()
        node = self.node_by_id(node_id)
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            if is_write_back:
                result.backup_kv = self._build_backup_kv_action(node, write_back=True)
                return result
            # Write-through: node has no backup, delete entirely.
            self._delete_unbacked_device_leaf(
                node,
                result.tracker,
                device_frees=result.device_frees,
                host_frees=result.host_frees,
            )
            return result
        self._demote(
            node,
            result.tracker,
            device_frees=result.device_frees,
            host_frees=result.host_frees,
        )
        return result

    def drop_subtree_no_host(self, node_id: NodeId) -> DropSubtreeNoHostResult:
        """Write-back fallback when a D-leaf's D->H backup fails under host
        memory pressure: drop the subtree rooted at the unbacked leaf so
        device eviction keeps making progress instead of leaving its KV
        unevictable until host space frees up."""
        result = DropSubtreeNoHostResult(is_dropped=False)
        node = self.node_by_id(node_id)
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        # A failed backup never issues the D->H copy, so the subtree root has
        # no host state and no in-flight DMA reading its device slots.
        assert not node.backuped and node.write_through_pending_id is None
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return result
        descendants: list[UnifiedTreeNode] = []
        stack = list(node.children.values())
        while stack:
            cur = stack.pop()
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                return result
            descendants.append(cur)
            stack.extend(cur.children.values())
        for desc in reversed(descendants):
            # Host-only by construction: a device descendant would contradict
            # this node being a D-leaf, and D-leaves evict before ancestors.
            assert desc.evicted and desc.backuped, f"node {desc.id} not host-only"
            assert desc.write_through_pending_id is None
            self._release_all_component_layers(
                desc,
                StorageMedium.CPU,
                result.tracker,
                result.device_frees,
                result.host_frees,
            )
            self._remove_leaf_from_parent(desc)
        self._delete_unbacked_device_leaf(
            node,
            result.tracker,
            device_frees=result.device_frees,
            host_frees=result.host_frees,
        )
        result.is_dropped = True
        return result

    def _release_all_component_layers(
        self,
        node: UnifiedTreeNode,
        medium: StorageMedium,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free every component layer on the node and detach it from the LRU
        lists and evictable leaf sets."""
        self._record_remove_event(node, medium=medium)
        for comp in self.components:
            self._evict_component_and_detach_lru(
                node,
                comp,
                target=EvictLayer.ALL,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
        self.evictable_device_leaves.discard(node)
        self.evictable_host_leaves.discard(node)

    def _delete_unbacked_device_leaf(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Delete a device leaf that has no host backup, freeing all layers."""
        self._release_all_component_layers(
            node, StorageMedium.GPU, tracker, device_frees, host_frees
        )
        parent = node.parent
        self._remove_leaf_from_parent(node)
        self._update_evictable_leaf_sets(parent)
        self._iteratively_delete_tombstone_leaf(
            node, tracker, device_frees=device_frees, host_frees=host_frees
        )

    def drive_host_eviction(
        self, component_type: ComponentType, num_tokens: int
    ) -> DriveHostEvictionResult:
        """Evict a component's host-side resources; no-op if the component is absent."""
        result = DriveHostEvictionResult()
        comp = self.components_by_type.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(
                num_tokens,
                result.tracker,
                result.device_frees,
                result.host_frees,
            )
        return result

    def _evict_host_leaf(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        for comp in self.components:
            _, hf = self._evict_component_and_detach_lru(
                node,
                comp,
                target=EvictLayer.ALL,
                tracker=None,
                device_frees=device_frees,
                host_frees=host_frees,
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker, device_frees, host_frees)

    def demote(self, node_id: NodeId) -> DemoteResult:
        """Release a node's device KV once its host copy exists; the node stays in the
        tree, now host-only."""
        result = DemoteResult()
        self._demote(
            self.node_by_id(node_id),
            result.tracker,
            result.device_frees,
            result.host_frees,
        )
        return result

    def _demote(
        self,
        node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        assert not node.evicted and node.backuped
        trigger = self.components_by_type[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node,
            trigger,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self._cascade_evict(
            node, trigger, tracker, device_frees=device_frees, host_frees=host_frees
        )
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""

        is_leaf = False
        if target == EvictLayer.DEVICE:
            is_leaf = node in self.evictable_device_leaves
        elif target == EvictLayer.HOST:
            is_leaf = node in self.evictable_host_leaves

        trigger_priority = trigger.eviction_priority(is_leaf)
        base_evicted = False

        for comp in self.components:
            if comp.eviction_priority(is_leaf) <= trigger_priority:
                if comp is not trigger and comp.node_has_component_data(node, target):
                    cd = node.component_data[comp.component_type]
                    # A comp whose TRUE internal priority outranks the trigger
                    # is only in this loop because leaf-collapse flattened
                    # priorities; a lock on it is a legit pin and must be
                    # spared. A lock on a strictly-lower-priority tier is a
                    # real strand — fall through to the assert below.
                    if comp.eviction_priority(
                        is_leaf=False
                    ) >= trigger.eviction_priority(is_leaf=False):
                        if EvictLayer.DEVICE in target and cd.lock_ref != 0:
                            continue
                        if EvictLayer.HOST in target and cd.host_lock_ref != 0:
                            continue
                        if cd.session_ref > 0 and trigger.session_ref(node) == 0:
                            continue
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node,
                        comp,
                        target=target,
                        tracker=tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )
                    if comp.component_type == BASE_COMPONENT_TYPE:
                        base_evicted = True

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        if EvictLayer.DEVICE in target and base_evicted:
            node.component_data[BASE_COMPONENT_TYPE].value = None

        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        for component in self.components:
            component.discard_deleted_session_leaf(node)

        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node
        self._unregister_node(node)

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(
            node, target=target, device_frees=device_frees, host_frees=host_frees
        )
        if tracker is not None:
            if EvictLayer.DEVICE in target:
                tracker[comp.component_type] += device_freed
            elif EvictLayer.HOST in target:
                tracker[comp.component_type] += host_freed

        # Detach from the appropriate LRU list(s)
        ct = comp.component_type
        for layer, lru_lists in (
            (EvictLayer.DEVICE, self.lru_lists),
            (EvictLayer.HOST, self.host_lru_lists),
        ):
            if layer in target:
                lru = lru_lists[ct]
                if lru.in_list(node):
                    lru.remove_node(node)
        return device_freed, host_freed

    def _iteratively_delete_tombstone_leaf(
        self,
        deleted_node: UnifiedTreeNode,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ):
        """Walk up from *deleted_node* and cascade-delete childless ancestors.

        Only the Full (base) component decides whether a node survives:
          - Full device present  → keep as D-leaf
          - Full host present    → keep as H-leaf
          - neither              → evict all remaining data, delete, continue up
        """
        ct = BASE_COMPONENT_TYPE
        cur = deleted_node.parent
        while cur != self.root_node and len(cur.children) == 0:
            if any(
                cd.lock_ref > 0 or cd.host_lock_ref > 0 for cd in cur.component_data
            ):
                break

            has_device = cur.component_data[ct].value is not None
            has_host = cur.component_data[ct].host_value is not None

            if has_device:
                self._update_evictable_leaf_sets(cur)
                break

            # Full device absent — clean up orphaned aux device data.
            for comp in self.components_by_type.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur,
                        comp,
                        target=EvictLayer.DEVICE,
                        tracker=tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )

            if has_host:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components_by_type.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur,
                        comp,
                        target=EvictLayer.HOST,
                        tracker=tracker,
                        device_frees=device_frees,
                        host_frees=host_frees,
                    )

            self.evictable_host_leaves.discard(cur)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        if node is self.root_node or not node.evicted:
            return False
        if not node.backuped:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True

    # ==== HiCache ====

    def set_hicache_enabled(self) -> None:
        self.enable_hicache = True

    def insert_host(
        self,
        node_id: NodeId,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        """Insert a host-side (backuped) tree path descending from the given node."""
        node = self.node_by_id(node_id)
        total_len = len(key)
        self._touch_node(node)
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        matched_length = 0
        cache_actions: list[CacheAction | ComponentAction] = []
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                node, action = self._split_node(node.key, node, prefix_len)
                if action is not None:
                    cache_actions.append(action)

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(
            prefix_len=matched_length,
            total_len=total_len,
            cache_actions=cache_actions,
        )
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node.id
            return result

        # Drop the refill only under write-through (a non-write-back policy).
        if node is not self.root_node and not node.backuped and not self.is_write_back:
            logger.info(
                "HiCache prefetch dropped %d-token refill under un-backed-up node %d",
                len(host_value),
                node.id,
            )
            result.host_insert_dropped = True
            return result

        new_node = self._new_node(priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node.id
        return result

    def build_backup_spec(self, node_id: NodeId):
        """Read a node's device->host backup spec (device value + component transfers) now."""
        return self._build_backup_spec(self.node_by_id(node_id))

    def _build_backup_spec(self, node: UnifiedTreeNode):
        """Gather device value backup spec."""
        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        return device_value, comp_xfers

    def build_storage_backup_spec(
        self, node_id: NodeId, pass_prefix_keys: bool
    ) -> Optional[StorageBackupSpec]:
        """Gather a node's device->storage backup spec; None if the node is not backuped."""
        node = self.node_by_id(node_id)
        if not node.backuped:
            return None
        prefix_keys = None
        if pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node, CacheTransferPhase.BACKUP_STORAGE
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers
        return StorageBackupSpec(
            host_value=node.component_data[BASE_COMPONENT_TYPE].host_value,
            token_ids=node.key.token_ids,
            hash_value=node.hash_value,
            prefix_keys=prefix_keys,
            comp_xfers=comp_xfers,
        )

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
        """Route a build_hicache_transfers call to the component for the given type."""
        return self.components_by_type[component_type].build_hicache_transfers(
            self.node_by_id(node_id),
            phase,
            host_indices=host_indices,
            token_ids=token_ids,
            prefetch_tokens=prefetch_tokens,
            last_hash=last_hash,
        )

    def build_load_back_spec(
        self, node_id: NodeId, req: Optional[Req] = None
    ) -> tuple[PoolTransfer, dict[ComponentType, list[PoolTransfer]]]:
        """Build the H->D load-back KV transfer plus per-component aux transfers."""
        # Component hooks take primitives, not Req: extract its fields here.
        mamba_pool_idx = req.mamba_pool_idx if req is not None else None
        node = self.node_by_id(node_id)
        kv_xfer = self.components_by_type[BASE_COMPONENT_TYPE].build_hicache_transfers(
            node, CacheTransferPhase.LOAD_BACK
        )[0]
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self.components:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                node, CacheTransferPhase.LOAD_BACK, mamba_pool_idx=mamba_pool_idx
            )
            if t:
                comp_xfers[comp.component_type] = t
        return kv_xfer, comp_xfers

    def prefetch_anchor_info(self, node_id: NodeId) -> Optional[str]:
        """The anchor node's key extra_key."""
        node = self.node_by_id(node_id)
        return node.key.extra_key if node.key else None

    def _build_backup_kv_action(
        self, node: UnifiedTreeNode, write_back: bool = False
    ) -> BackupKV:
        """Build the backup action for a node and its unbacked ancestors."""
        chain = [node]
        if not write_back:
            ancestor = node.parent
            while (
                ancestor is not None
                and ancestor is not self.root_node
                and not ancestor.backuped
            ):
                chain.append(ancestor)
                ancestor = ancestor.parent
            # write_through: Ancestors first to preserve backup invariant
            chain.reverse()
        return BackupKV([target.id for target in chain])

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
        node = self.node_by_id(node_id)
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                phase,
                xfers,
                cache_actions=cache_actions,
                insert_result=insert_result,
                pool_storage_result=pool_storage_result,
            )

    def commit_backup(
        self,
        node_id: NodeId,
        host_indices: torch.Tensor,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> None:
        """Commit a successful backup to the node."""
        node = self.node_by_id(node_id)
        cache_actions: list[CacheAction | ComponentAction] = []
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
            cache_actions=cache_actions,
        )
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
                cache_actions=cache_actions,
            )
        assert not cache_actions  # BACKUP_HOST emits no actions

    def commit_load_back(
        self,
        node_id: NodeId,
        device_indices: torch.Tensor,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[CacheAction | ComponentAction]:
        """Commit a successful H->D load-back onto the node; the SWA full->swa mapping
        rebuild is deferred to the orchestration layer."""
        node = self.node_by_id(node_id)
        cache_actions: list[CacheAction | ComponentAction] = []
        kv_xfer.device_indices = device_indices
        self.components_by_type[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
            cache_actions=cache_actions,
        )
        for nid in kv_xfer.nodes_to_load or ():
            loaded = self.node_by_id(nid)
            self._record_store_event(loaded, medium=StorageMedium.GPU)
        for ct, xfers in comp_xfers.items():
            self.components_by_type[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
                cache_actions=cache_actions,
            )
        self._update_evictable_leaf_sets(node)
        return cache_actions

    def mark_write_through_pending(self, node_id: NodeId) -> None:
        """Mark a node as having an in-flight write-through backup."""
        node = self.node_by_id(node_id)
        node.write_through_pending_id = node_id

    def finish_write_through(self, node_ids: list[NodeId], ack_id: int) -> None:
        """Clear the write-through-pending mark (when it matches ack_id) and record the
        host store event for each acked node."""
        for node_id in node_ids:
            node = self.node_by_id(node_id)
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            self._record_store_event(node, medium=StorageMedium.CPU)

    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        """Store an auxiliary component's device value onto a node."""
        # Full uses leaf sets, not LRU; its stores go through the insert paths.
        assert component_type != BASE_COMPONENT_TYPE
        node = self.node_by_id(node_id)
        node.component_data[component_type].value = value
        host_lru = self.host_lru_lists[component_type]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.lru_lists[component_type].insert_mru(node)
        self.component_evictable_size_[component_type] += len(value)

    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        """The component's device value on the node, or None if evicted."""
        return self.node_by_id(node_id).component_data[component_type].value

    def component_has_host_value_only(
        self, node_id: NodeId, component_type: ComponentType
    ) -> bool:
        """Whether the component's data is device-evicted but host-backed."""
        cd = self.node_by_id(node_id).component_data[component_type]
        return cd.value is None and cd.host_value is not None

    # ==== Others ====

    def sanity_check(
        self,
        ongoing_write_through: list[tuple[int, NodeId]],
        ongoing_load_back: list[tuple[int, NodeId]],
    ) -> None:
        """Verify tree-structure, leaf-set, LRU, size, and ongoing-op invariants; raise
        AssertionError on any violation. ongoing_* args are (id, node_id) pairs.
        """
        errors: list[str] = []
        E = errors.append
        all_nodes = self._collect_all_nodes()
        all_node_set = set(all_nodes)
        FCT = BASE_COMPONENT_TYPE

        # ── PART 1: Tree Structure ──
        # Root state
        if self.root_node.component_data[FCT].value is None:
            E("[Root] root missing Full device value")
        if self.root_node.component_data[FCT].lock_ref <= 0:
            E(
                f"[Root] root Full lock_ref={self.root_node.component_data[FCT].lock_ref}"
            )
        if self.root_node.parent is not None:
            E("[Root] root has a parent pointer")
        # Parent ↔ child bidirectional consistency
        for node in all_nodes:
            for child in node.children.values():
                if child.parent is not node:
                    pid = child.parent.id if child.parent else None
                    E(f"[Tree] child {child.id} parent={pid}, expected {node.id}")
                if child.key is None:
                    E(f"[Tree] node {child.id} has no key")

        # ── PART 2: Per-node state machine and leaf qualification ──
        expected_dev_leaves: set[UnifiedTreeNode] = set()
        expected_hst_leaves: set[UnifiedTreeNode] = set()

        for node in all_nodes:
            if node is self.root_node:
                continue
            nid = node.id
            full_dev = node.component_data[FCT].value is not None
            full_hst = node.component_data[FCT].host_value is not None

            # Full is the tree backbone, so aux data requires Full data.
            for ct in self.component_types:
                if ct == FCT:
                    continue
                cd = node.component_data[ct]
                if cd.value is not None and not full_dev:
                    E(f"node {nid} {ct} device present but Full.value=None")
                if cd.host_value is not None and not full_hst:
                    E(f"node {nid} {ct} host present but Full.host_value=None")

            # Every node must keep Full data on at least one layer.
            if not full_dev and not full_hst:
                E(f"node {nid} dead: no Full device and no Full host")

            # Parent prefixes must keep data whenever the child does.
            if node.parent is not None and node.parent is not self.root_node:
                p_dev = node.parent.component_data[FCT].value is not None
                p_hst = node.parent.component_data[FCT].host_value is not None
                if full_dev and not p_dev:
                    E(f"node {nid} device present but parent {node.parent.id} evicted")
                if full_hst and not p_hst and not self.is_write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.component_types:
                cd = node.component_data[ct]
                if cd.lock_ref < 0:
                    E(f"node {nid} {ct} lock_ref={cd.lock_ref}")
                if cd.host_lock_ref < 0:
                    E(f"node {nid} {ct} host_lock_ref={cd.host_lock_ref}")
                if ct != FCT and fl < cd.lock_ref:
                    E(f"node {nid} full_lock={fl} < {ct}_lock={cd.lock_ref}")
                if cd.value is None and cd.lock_ref > 0:
                    E(f"node {nid} {ct} evicted but lock_ref={cd.lock_ref}")

            # Collect expected leaf qualification (single pass)
            if self._is_device_leaf(node):
                expected_dev_leaves.add(node)
            if self._is_host_leaf(node):
                expected_hst_leaves.add(node)

        # ── PART 3: Tracking structures ──

        # Device leaf set must match the expected leaves.
        if self.evictable_device_leaves != expected_dev_leaves:
            extra = self.evictable_device_leaves - expected_dev_leaves
            missing = expected_dev_leaves - self.evictable_device_leaves
            if extra:
                E(f"D-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"D-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # Host leaf set must match the expected leaves.
        if self.evictable_host_leaves != expected_hst_leaves:
            extra = self.evictable_host_leaves - expected_hst_leaves
            missing = expected_hst_leaves - self.evictable_host_leaves
            if extra:
                E(f"H-leaf extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

        if self.enable_session_radix_cache:
            for component in self.components:
                component.validate_session_state(all_node_set, E)

        # Stale nodes: leaf sets must only contain tree-reachable nodes
        stale = self.evictable_device_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in device_leaves: {[n.id for n in list(stale)[:5]]}"
            )
        stale = self.evictable_host_leaves - all_node_set
        if stale:
            E(
                f"{len(stale)} stale nodes in host_leaves: {[n.id for n in list(stale)[:5]]}"
            )

        # Per-component LRU tracking
        for ct in self.component_types:
            lru = self.lru_lists[ct]
            if ct == FCT:
                # Full uses leaf sets, not LRU
                if len(lru.cache) > 0:
                    E(f"Full device LRU not empty: {len(lru.cache)}")
                if len(self.host_lru_lists[ct].cache) > 0:
                    E(f"Full host LRU not empty: {len(self.host_lru_lists[ct].cache)}")
            else:
                # Aux device values must match the device LRU.
                tree_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is not None
                }
                lru_ids = set(lru.cache.keys())
                if tree_ids != lru_ids:
                    E(
                        f"{ct} device LRU: "
                        f"+tree={tree_ids - lru_ids}, +lru={lru_ids - tree_ids}"
                    )
                # Aux host-only states must match the host LRU.
                host_lru = self.host_lru_lists[ct]
                s3_ids = {
                    n.id
                    for n in all_nodes
                    if n is not self.root_node
                    and n.component_data[ct].value is None
                    and n.component_data[ct].host_value is not None
                }
                host_lru_ids = set(host_lru.cache.keys())
                if s3_ids != host_lru_ids:
                    E(
                        f"{ct} host LRU: "
                        f"+S3={s3_ids - host_lru_ids}, +lru={host_lru_ids - s3_ids}"
                    )
                # The same aux node must not appear in both device and host LRU.
                inv5_overlap = lru_ids & host_lru_ids
                if inv5_overlap:
                    E(f"{ct} in both device and host LRU: {inv5_overlap}")
                # Linked-list integrity
                self._check_lru_linked_list(lru, ct, "device", errors)
                self._check_lru_linked_list(host_lru, ct, "host", errors)

        # ── PART 4: Size Accounting ──
        for ct in self.component_types:
            evictable = 0
            protected = 0
            for n in all_nodes:
                if n is self.root_node:
                    continue
                cd = n.component_data[ct]
                if cd.value is not None:
                    toks = len(cd.value)
                    if cd.lock_ref > 0:
                        protected += toks
                    else:
                        evictable += toks
            if self.component_evictable_size_[ct] != evictable:
                E(
                    f"[Size] {ct} evictable={self.component_evictable_size_[ct]} "
                    f"!= recomputed={evictable}"
                )
            if self.component_protected_size_[ct] != protected:
                E(
                    f"[Size] {ct} protected={self.component_protected_size_[ct]} "
                    f"!= recomputed={protected}"
                )

        # ── PART 5: Ongoing Operations ──
        for nid, node_id in ongoing_write_through:
            n = self._node_arena.get(node_id)
            if n is None or n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, node_id in ongoing_load_back:
            n = self._node_arena.get(node_id)
            if n is None or n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )

        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def _check_lru_linked_list(
        self,
        lru: UnifiedLRUList,
        ct: ComponentType,
        label: str,
        errors: list[str],
    ) -> None:
        """Walk a LRU doubly-linked list, collect integrity errors."""
        pt = lru._pt  # use LRU's own pointer slot
        visited: set[int] = set()
        x = lru.head.lru_next[pt]
        prev = lru.head
        while x is not None and x != lru.tail:
            if x.lru_prev[pt] != prev:
                errors.append(f"[{label}][{ct}] broken prev at node {x.id}")
            if x is lru.mid or x is lru.cursor:
                prev = x
                x = x.lru_next[pt]
                continue
            if x.id not in lru.cache:
                errors.append(f"[{label}][{ct}] node {x.id} in list not cache")
            if x.id in visited:
                errors.append(f"[{label}][{ct}] cycle at node {x.id}")
                break
            visited.add(x.id)
            prev = x
            x = x.lru_next[pt]
        if x is None:
            errors.append(
                f"[{label}][{ct}] broken chain: lru_next is None "
                f"after node {prev.id if hasattr(prev, 'id') else 'head'}"
            )
        if len(visited) != len(lru.cache):
            errors.append(
                f"[{label}][{ct}] list={len(visited)} != cache={len(lru.cache)}"
            )

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            component_str = " ".join(
                f"{ct}={'yes' if node.component_data[ct].value is not None else 'no'}"
                for ct in self.component_types
            )
            print(
                " " * indent,
                f"[{node.id}]",
                len(node.key),
                f"full_lock={node.component_data[BASE_COMPONENT_TYPE].lock_ref}",
                component_str,
            )
            for child in node.children.values():
                stack.append((child, indent + 2))

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

    def component_evictable_size(self, component_type: ComponentType) -> int:
        """Evictable token count for one component (0 if the component is absent)."""
        return self.component_evictable_size_.get(component_type, 0)

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.SWA, 0)

    def mamba_evictable_size(self) -> int:
        return self.component_evictable_size_.get(ComponentType.MAMBA, 0)

    def swa_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.SWA, 0)

    def mamba_protected_size(self) -> int:
        return self.component_protected_size_.get(ComponentType.MAMBA, 0)

    def total_size(self) -> tuple[int, int]:
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.component_types:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

    def walk_for_kv_canary(
        self, unlocked_only: bool, swa_resident_only: bool
    ) -> RadixCacheWalkResult:
        """Flatten every FULL device slot into (slot, position, prev-slot) rows for the KV-canary sweep."""
        slots: list[int] = []
        positions: list[int] = []
        prev_slots: list[int] = []
        swa_filter = swa_resident_only and ComponentType.SWA in self.components_by_type

        def _dfs(node: UnifiedTreeNode, depth: int, parent_last_slot: int) -> None:
            value = node.component_data[BASE_COMPONENT_TYPE].value
            node_slots = value.tolist() if isinstance(value, torch.Tensor) else []

            emit = node is not self.root_node
            if unlocked_only:
                # Unified SWA owns an independent component lock. A node can still
                # hold Full KV for a running request while its SWA slots are unused.
                lock_ct = ComponentType.SWA if swa_filter else BASE_COMPONENT_TYPE
                emit = emit and node.component_data[lock_ct].lock_ref == 0
            if swa_filter:
                emit = emit and node.component_data[ComponentType.SWA].value is not None

            # Skipped nodes still advance the chain/depth so descendants stay consistent.
            chain_last_slot = parent_last_slot
            for j, slot in enumerate(node_slots):
                if emit:
                    slots.append(slot)
                    positions.append(depth + j)
                    prev_slots.append(parent_last_slot if j == 0 else node_slots[j - 1])
                chain_last_slot = slot

            # Device-evicted nodes hold no slots but still span their key length.
            if node is self.root_node or node.key is None:
                child_depth = depth + len(node_slots)
            else:
                child_depth = depth + len(node.key)
            for child in node.children.values():
                _dfs(child, child_depth, chain_last_slot)

        _dfs(self.root_node, 0, -1)
        return RadixCacheWalkResult(
            slot_indices=torch.tensor(slots, dtype=torch.int64),
            positions=torch.tensor(positions, dtype=torch.int64),
            prev_slot_indices=torch.tensor(prev_slots, dtype=torch.int64),
        )

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs(node: UnifiedTreeNode):
            for child in node.children.values():
                v = child.component_data[BASE_COMPONENT_TYPE].value
                if v is not None:
                    values.append(v)
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def _all_component_values_flatten(
        self, component_type: ComponentType
    ) -> torch.Tensor:
        if component_type not in self.components_by_type:
            return torch.tensor([], dtype=torch.int64, device=self.device)

        values = []

        def _dfs(node: UnifiedTreeNode):
            value = node.component_data[component_type].value
            if value is not None:
                values.append(value)
            for child in node.children.values():
                _dfs(child)

        _dfs(self.root_node)
        if values:
            return torch.cat(values)
        return torch.tensor([], dtype=torch.int64, device=self.device)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._all_component_values_flatten(ComponentType.MAMBA)
