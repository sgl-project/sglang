from __future__ import annotations

import hashlib
import logging
import math
import sys
import threading
import time
from array import array
from collections import defaultdict, deque
from functools import partial
from itertools import islice
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Optional, Sequence, TypeVar

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import (
    CheckpointRetryResource,
    L3StagingLease,
    PendingStorageCheckpoint,
    PrefetchOwnershipTransition,
    StorageCheckpointRegistry,
    StorageCheckpointRetryQueues,
    StorageCheckpointState,
    StoragePrefetchAdmissionPin,
    StoragePrefetchDeviceAdmissionPin,
    StoragePrefetchOwnership,
    StoragePrefetchState,
    StoragePrefetchTracker,
    _index_values,
    _take_indices,
    bounded_host_eviction_scan,
    get_host_eviction_scan_budget,
    l3_staging_allocation,
    make_storage_checkpoint_handle,
)
from sglang.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    LRURefreshPhase,
    MambaComponent,
    SWAComponent,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.session.streaming_session import StreamingSession

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.server_args import ServerArgs


T = TypeVar("T")

_CHECKPOINT_RETRY_QUANTUM = 8
_CHECKPOINT_SCAN_QUANTUM = 64
_ABORTED_PREFETCH_CLEANUP_QUANTUM = 8
_L3_STAGING_EVICTION_QUANTUM_PAGES = 64
_L3_STAGING_HOST_EVICTION_SCAN_QUANTUM = 256
_L3_STAGING_PROMOTION_QUANTUM_PAGES = 64
_L3_STAGING_DISCARD_QUANTUM_ENTRIES = 64
_L3_STAGING_DISCARD_SCAN_QUANTUM_NODES = 256
_L3_STAGING_RECLAIM_QUANTUM_LEASES = 64
_CHECKPOINT_STAGING_CAPACITY_EXCEEDED = "staging_capacity"

_CHECKPOINT_BLOCKED_RESOURCES = {
    "host": CheckpointRetryResource.HOST,
    "auxiliary": CheckpointRetryResource.AUXILIARY,
    "storage": CheckpointRetryResource.STORAGE,
    "continuation": CheckpointRetryResource.SCHEDULER,
}
_CAPACITY_RETRY_RESOURCES = frozenset(
    {
        CheckpointRetryResource.HOST,
        CheckpointRetryResource.AUXILIARY,
    }
)


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...], priority: int = 0):
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
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
        self._host_scan_cursor: UnifiedTreeNode | None = None

    def _add_node_after(self, prev_node: UnifiedTreeNode, new_node: UnifiedTreeNode):
        pt = self._pt
        new_node.lru_prev[pt] = prev_node
        new_node.lru_next[pt] = prev_node.lru_next[pt]
        prev_node.lru_next[pt].lru_prev[pt] = new_node
        prev_node.lru_next[pt] = new_node

    def _add_node(self, node: UnifiedTreeNode):
        self._add_node_after(self.head, node)

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

    def reset_node_and_parents_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        should_include,
    ):
        prev_node = self.head
        while node != root_node:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def reset_node_and_window_ancestors_mru(
        self,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
        window_size: int,
        should_include,
    ):
        prev_node = self.head
        accumulated = 0
        while node != root_node and accumulated < window_size:
            if should_include(node):
                assert node.id in self.cache
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
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
        budget = get_host_eviction_scan_budget()
        if budget is not None:
            if check_id:
                assert node.id in self.cache
            if node is self.tail:
                x = (
                    self._host_scan_cursor
                    if self.in_list(self._host_scan_cursor)
                    else node.lru_prev[self._pt]
                )
            else:
                x = node.lru_prev[self._pt]
            while x is not self.head:
                if budget.remaining_nodes == 0:
                    self._host_scan_cursor = x
                    return None
                budget.remaining_nodes -= 1
                if x.component_data[self.component_type].host_lock_ref == 0:
                    previous = x.lru_prev[self._pt]
                    self._host_scan_cursor = (
                        previous if previous is not self.head else None
                    )
                    return x
                x = x.lru_prev[self._pt]
            self._host_scan_cursor = None
            return None

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


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}

logger = logging.getLogger(__name__)


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node: UnifiedTreeNode
    lock_params: Optional[DecLockRefParams]
    publish_nodes: list[UnifiedTreeNode]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node: UnifiedTreeNode
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node: UnifiedTreeNode
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedRadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.kv_event_queue = []
        self.eviction_policy = params.eviction_policy.lower()
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.is_eagle = (
            params.is_eagle and ComponentType.MAMBA not in self.tree_components
        )
        component_registry = COMPONENT_REGISTRY
        if params.component_registry_override:
            component_registry = {
                **COMPONENT_REGISTRY,
                **params.component_registry_override,
            }
        self.components: dict[ComponentType, TreeComponent] = {
            ct: component_registry[ct](self, params) for ct in self.tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(
            self.components.values()
        )
        self.sidecar_pool_specs: list[SidecarPoolSpec] = []

        # Streaming session: embedded StreamingSession with self as inner.
        # Always on -- zero overhead when no streaming session is open (the
        # try_* entries short-circuit on non-streaming reqs / real TreeNodes).
        # Dispatch methods below pre-check conditions so the session's
        # internal fall-through to self.inner.xxx never fires -- no recursion.
        self.session = StreamingSession(inner=self)

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = (
            1
            if self.tp_group is None
            else torch.distributed.get_world_size(group=self.tp_group)
        )
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.work_list: list[torch.distributed.Work] = []

        # HiCache D↔H defaults (overridden by init_hicache)
        self.cache_controller: Optional[HybridCacheController] = None
        self.write_through_threshold = 256
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False

        self.reset()
        logger.info(f"Init Unified RadixTree with components {self.tree_components}")

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def synchronize_storage_prefetch_flag(self, flag: bool) -> bool:
        flag_tensor = torch.tensor(int(flag), dtype=torch.int)
        self._all_reduce_attn_groups(flag_tensor, torch.distributed.ReduceOp.MAX)
        return flag_tensor.item() != 0

    def synchronize_storage_prefetch_min(self, value: int) -> int:
        value_tensor = torch.tensor(value, dtype=torch.int)
        self._all_reduce_attn_groups(value_tensor, torch.distributed.ReduceOp.MIN)
        return int(value_tensor.item())

    def synchronize_storage_prefetch_min_values(self, values: list[int]) -> list[int]:
        value_tensor = torch.tensor(values, dtype=torch.int64)
        self._all_reduce_attn_groups(value_tensor, torch.distributed.ReduceOp.MIN)
        return [int(value) for value in value_tensor.tolist()]

    def _synchronize_storage_state_change_resources(
        self, resources: set[CheckpointRetryResource]
    ) -> set[CheckpointRetryResource]:
        ordered_resources = tuple(CheckpointRetryResource)
        flags = torch.tensor(
            [int(resource in resources) for resource in ordered_resources],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(flags, torch.distributed.ReduceOp.MAX)
        return {
            resource
            for resource, present in zip(ordered_resources, flags.tolist(), strict=True)
            if present
        }

    def _require_storage_prefetch_rank_agreement(
        self, phase: str, values: dict[str, int]
    ) -> None:
        local_values = list(values.values())
        reduced = self.synchronize_storage_prefetch_min_values(
            local_values + [-value for value in local_values]
        )
        count = len(local_values)
        minimums = reduced[:count]
        maximums = [-value for value in reduced[count:]]
        divergent = {
            name: (minimum, maximum)
            for name, minimum, maximum in zip(values, minimums, maximums, strict=True)
            if minimum != maximum
        }
        if divergent:
            raise RuntimeError(
                f"HiCache storage prefetch {phase} diverged across attention "
                f"ranks: {divergent}"
            )

    def _drain_async_work(self):
        """
        Block until all outstanding async sends are consumed, then clear.

        Called at the start of each event round, so work_list holds the sends
        accumulated since the last round. This bounds it and applies
        backpressure when a downstream PP rank lags. Scheduler thread only.
        """
        for work in self.work_list:
            work.wait()
        self.work_list.clear()

    def _all_reduce(self, data: torch.Tensor, tp_reduce_op: torch.distributed.ReduceOp):
        """
        Synchronize data across all TP and PP ranks.

        In particular, "tp_reduce_op" is performed on all TP ranks of the first PP rank,
        and then the result is propagated to all following PP ranks.

        Must be called in the scheduler thread.
        """
        if self.pp_rank == 0:
            self._all_reduce_attn_groups(data, tp_reduce_op)
        self._pp_sync(data)

    def _pp_sync(self, data: torch.Tensor) -> None:
        """
        Synchronize data across the PP pipeline, where PPn (n>0) will receive PP0's data.
        """
        if self.pp_size <= 1 or self.pp_group is None:
            return
        if self.pp_rank > 0:
            torch.distributed.recv(
                data,
                group_src=self.pp_rank - 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
        if self.pp_rank + 1 < self.pp_size:
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def reset(self) -> None:
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        if hasattr(self, "storage_pending_checkpoints"):
            for handle in tuple(self.storage_pending_checkpoints):
                self._release_storage_checkpoint(handle, release_staging=True)
        if hasattr(self, "storage_prefetch_admission_host_locks"):
            request_ids = set(self.storage_prefetch_admission_host_locks)
            request_ids.update(self.storage_prefetch_admission_device_locks)
            for request_id in request_ids:
                self.release_storage_prefetch_admission(request_id)

        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.priority = -sys.maxsize
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        self.root_node.hash_value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        self.component_evictable_size_ = {ct: 0 for ct in self.tree_components}
        self.component_protected_size_ = {ct: 0 for ct in self.tree_components}

        self.lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components) for ct in self.tree_components
        }
        self.session.slots.clear()

        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.evictable_host_scan_queue: dict[UnifiedTreeNode, None] = {}
        self.host_lru_lists = {
            ct: UnifiedLRUList(ct, self.tree_components, use_host_ptr=True)
            for ct in self.tree_components
        }
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.storage_prefetch_tracker = StoragePrefetchTracker()
        self.storage_checkpoint_registry = StorageCheckpointRegistry()
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        self.ongoing_backup: dict[int, tuple[UnifiedTreeNode, DecLockRefParams]] = {}
        self.storage_pending_checkpoints: dict[str, PendingStorageCheckpoint] = {}
        self.storage_checkpoint_generation = 0
        self.storage_checkpoint_owners_by_operation: dict[int, tuple[str, int]] = {}
        self.storage_checkpoint_retries = StorageCheckpointRetryQueues()
        self.storage_state_change_resources: set[CheckpointRetryResource] = set()
        self.storage_prefetch_ownership: dict[str, StoragePrefetchOwnership] = {}
        self.storage_prefetch_aborted_requests: set[str] = set()
        self.storage_prefetch_admission_host_locks: dict[
            str, StoragePrefetchAdmissionPin
        ] = {}
        self.storage_prefetch_admission_device_locks: dict[
            str, StoragePrefetchDeviceAdmissionPin
        ] = {}
        self.storage_checkpoint_dependency_results: dict[str, str] = {}
        self.l3_staging_pending_leases: deque[L3StagingLease] = deque()
        self.l3_staging_pending_keys: set[tuple[Any, ...]] = set()
        self.l3_staging_pending_tokens: dict[tuple[Any, ...], int] = {}
        self.l3_staging_pending_by_key: dict[tuple[Any, ...], L3StagingLease] = {}
        self.l3_staging_pending_token_total = 0
        self.l3_staging_reclaim_eligible = 0
        self.l3_staging_generation_seen = 0
        self.storage_radix_tree_generation = 0

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.cache_controller.l3_staging.reset_after_clear()
            self.l3_staging_generation_seen = (
                self.cache_controller.l3_staging.generation
            )
            self.enable_storage = self.cache_controller.enable_storage

        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        # Direct IO layout fixup (must happen before pool creation)
        if server_args.hicache_io_backend == "direct":
            if server_args.hicache_mem_layout == "page_first":
                server_args.override(
                    "hicache.mem_layout_force", hicache_mem_layout="page_first_direct"
                )
                logger.warning(
                    "Page first layout is not supported with direct IO backend, "
                    "switching to page first direct layout"
                )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = server_args.extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = server_args.hicache_storage_backend
        storage_extra_config = None
        storage_prefetch_threshold = 256
        prefetch_timeout_base = 1.0
        prefetch_timeout_per_ki_token = 0.25
        hicache_storage_pass_prefix_keys = False
        if storage_backend is not None:
            (
                storage_extra_config,
                storage_prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = HybridCacheController.parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )

        attach_hybrid_pool_to_unified_cache(
            self,
            params,
            server_args,
            load_cache_event=self.load_cache_event,
            storage_backend=storage_backend,
            storage_extra_config=storage_extra_config,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )
        self.cache_controller.scheduler_managed_prefetch = True
        if storage_backend is not None:
            try:
                self.cache_controller.l3_staging.install(
                    self.cache_controller.mem_pool_host,
                    envs.SGLANG_HICACHE_L3_STAGING_RESERVE_RATIO.get(),
                    self.cache_controller._synchronize_staging_values,
                )
            except Exception:
                self.cache_controller.detach_storage_backend()
                raise
            if self.cache_controller.l3_staging.reserve_tokens:
                logger.info(
                    "HiCache isolated L3 staging enabled ratio=%.3f pools=%s",
                    self.cache_controller.l3_staging.reserve_ratio,
                    self.cache_controller.l3_staging.usage(),
                )
        self.l3_staging_generation_seen = self.cache_controller.l3_staging.generation

        # State initialization
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        if storage_backend is not None:
            self._apply_storage_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )

    def register_sidecar_pool(self, spec: SidecarPoolSpec) -> None:
        self.sidecar_pool_specs.append(spec)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        result = self._insert_helper(self.root_node, key, value, params)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            if skip_swa and component.component_type == ComponentType.SWA:
                continue
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node: UnifiedTreeNode,
        swa_uuid_for_lock: Optional[int] = None,
    ) -> None:
        """Early-release the SWA portion of a request's tree lock, plus any
        strictly-lower-priority locks (e.g. Mamba) co-located on `node`.
        """
        if self.disable:
            return
        swa_component = self.components.get(ComponentType.SWA)
        if swa_component is None:
            return
        swa_component.release_window_lock(node, swa_uuid_for_lock)

        # Drop strictly-lower-priority locks (e.g. Mamba) co-located on `node`.
        swa_priority = swa_component.eviction_priority(is_leaf=False)
        dec_params = DecLockRefParams(swa_uuid_for_lock=swa_uuid_for_lock)
        for comp in self._components_tuple:
            if comp.eviction_priority(is_leaf=False) < swa_priority:
                comp.release_component_lock(node, dec_params)

    def inc_host_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(
                node=node, result=result, lock_host=True
            )

        self._update_evictable_leaf_sets(node)
        return result

    def dec_host_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params, lock_host=True)

        self._update_evictable_leaf_sets(node)
        return DecLockRefResult()

    def record_storage_checkpoint_dependency_result(
        self, request_id: str, result: str
    ) -> None:
        if result not in {"ready", "failed", "timeout"}:
            raise ValueError(f"Unknown storage checkpoint result: {result}")
        if request_id in self.storage_checkpoint_dependency_results:
            return
        self.storage_checkpoint_dependency_results[request_id] = result
        logger.info(
            "HiCache checkpoint dependency req=%s result=%s", request_id, result
        )
        if (
            self.enable_storage_metrics
            and self.storage_metrics_collector is not None
            and hasattr(
                self.storage_metrics_collector,
                "log_storage_checkpoint_dependency",
            )
        ):
            self.storage_metrics_collector.log_storage_checkpoint_dependency(result)

    def get_storage_checkpoint_dependency_state(
        self, request_id: str, checkpoint_dependency: str
    ) -> StorageCheckpointState:
        previous = self.storage_checkpoint_dependency_results.get(request_id)
        if previous == "ready":
            return StorageCheckpointState.READY
        if previous in {"failed", "timeout"}:
            return StorageCheckpointState.FAILED

        state = self.storage_checkpoint_registry.get_state(checkpoint_dependency)
        flags = torch.tensor(
            [
                int(state is StorageCheckpointState.FAILED),
                int(state is not StorageCheckpointState.READY),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(flags, torch.distributed.ReduceOp.MAX)
        if flags[0].item() != 0:
            self.record_storage_checkpoint_dependency_result(request_id, "failed")
            return StorageCheckpointState.FAILED
        if flags[1].item() != 0:
            return StorageCheckpointState.PENDING
        self.record_storage_checkpoint_dependency_result(request_id, "ready")
        return StorageCheckpointState.READY

    @staticmethod
    def _staging_lease_key(lease: L3StagingLease) -> tuple[Any, ...]:
        count = len(lease.indices)
        return (
            id(lease.quota),
            lease.owner_node.id if lease.owner_node is not None else -1,
            lease.component_type,
            count,
            int(lease.indices[0]) if count else -1,
            int(lease.indices[-1]) if count else -1,
            id(lease.indices) if lease.owner_node is None else 0,
        )

    @staticmethod
    def _staging_lease_pending_tokens(lease: L3StagingLease) -> int:
        return max(0, len(lease.indices) - lease.reclaim_index_cursor)

    @staticmethod
    def _staging_lease_logical_fingerprint(
        lease: L3StagingLease,
    ) -> tuple[int, ...]:
        component_value = (
            -1 if lease.component_type is None else int(lease.component_type)
        )
        digest = hashlib.sha256(lease.quota.name.encode())
        digest.update(b"\0")
        digest.update(str(component_value).encode())
        for hash_value in tuple(getattr(lease.owner_node, "hash_value", None) or ()):
            digest.update(b"\0")
            digest.update(hash_value.encode())
        return (
            int(lease.owner_node is not None),
            component_value,
            UnifiedRadixCache._staging_lease_pending_tokens(lease),
            lease.reclaim_index_cursor,
            *(
                int.from_bytes(digest.digest()[offset : offset + 4], "big") & 0x7FFFFFFF
                for offset in range(0, digest.digest_size, 4)
            ),
        )

    @staticmethod
    def _logical_fingerprint_digest(
        fingerprints: Sequence[tuple[int, ...]],
    ) -> tuple[int, ...]:
        digest = hashlib.sha256()
        for fingerprint in fingerprints:
            for value in fingerprint:
                digest.update(int(value).to_bytes(8, "big", signed=True))
        return tuple(
            int.from_bytes(digest.digest()[offset : offset + 4], "big") & 0x7FFFFFFF
            for offset in range(0, digest.digest_size, 4)
        )

    def _l3_staging_leases(
        self,
        host_indices: Optional[torch.Tensor],
        extra_pools: Optional[list[PoolTransfer]] = None,
        owner_node: Optional[UnifiedTreeNode] = None,
    ) -> tuple[L3StagingLease, ...]:
        controller = self.cache_controller
        if controller is None or not controller.l3_staging.quotas:
            return ()

        mem_pool_host = controller.mem_pool_host
        anchor_pool = getattr(
            getattr(mem_pool_host, "anchor_entry", None),
            "host_pool",
            mem_pool_host,
        )
        candidates = []
        if host_indices is not None:
            candidates.append((anchor_pool, host_indices, BASE_COMPONENT_TYPE))
        for transfer in extra_pools or ():
            if transfer.host_indices is None or transfer.indices_from_pool is not None:
                continue
            entry = mem_pool_host.entry_map.get(transfer.name)
            if entry is None:
                continue
            component_type = {
                PoolName.SWA: ComponentType.SWA,
                PoolName.MAMBA: ComponentType.MAMBA,
            }.get(transfer.name)
            candidates.append((entry.host_pool, transfer.host_indices, component_type))

        leases = []
        for pool, indices, component_type in candidates:
            quota = controller.l3_staging.quota_for_pool(pool)
            if quota is not None and quota.count_in_use(indices) > 0:
                leases.append(
                    L3StagingLease(
                        quota=quota,
                        indices=indices,
                        owner_node=owner_node,
                        component_type=component_type,
                    )
                )
        return tuple(leases)

    def _checkpoint_staging_leases(
        self, nodes: Sequence[UnifiedTreeNode]
    ) -> tuple[L3StagingLease, ...]:
        leases = []
        for node in nodes:
            extra_pools = [
                transfer
                for component in self._components_tuple
                if component.component_type != BASE_COMPONENT_TYPE
                for transfer in (
                    component.build_hicache_transfers(
                        node, CacheTransferPhase.BACKUP_STORAGE
                    )
                    or ()
                )
            ]
            leases.extend(
                self._l3_staging_leases(
                    node.component_data[BASE_COMPONENT_TYPE].host_value,
                    extra_pools,
                    owner_node=node,
                )
            )
        return tuple(leases)

    def _enqueue_staging_leases(
        self,
        leases: Sequence[L3StagingLease],
        *,
        schedule: bool = True,
        eligible: bool = True,
    ) -> None:
        live_leases = tuple(
            lease for lease in leases if self._staging_lease_pending_tokens(lease) > 0
        )
        for lease in live_leases:
            key = self._staging_lease_key(lease)
            canonical = self.l3_staging_pending_by_key.get(key)
            previous_tokens = self.l3_staging_pending_tokens.get(key)
            if previous_tokens is not None:
                if canonical is None:
                    raise RuntimeError("HiCache staging queue index is inconsistent")
                if lease.reclaim_index_cursor > canonical.reclaim_index_cursor:
                    canonical.reclaim_index_cursor = lease.reclaim_index_cursor
                    canonical.discard_scan_node = None
                    canonical.discard_scan_remaining = None
                current_tokens = self._staging_lease_pending_tokens(canonical)
                self.l3_staging_pending_token_total += current_tokens - previous_tokens
                self.l3_staging_pending_tokens[key] = current_tokens
                continue
            pending_tokens = self._staging_lease_pending_tokens(lease)
            self.l3_staging_pending_keys.add(key)
            self.l3_staging_pending_tokens[key] = pending_tokens
            self.l3_staging_pending_by_key[key] = lease
            self.l3_staging_pending_token_total += pending_tokens
            self.l3_staging_pending_leases.append(lease)
            if eligible:
                self.l3_staging_reclaim_eligible += 1
        if live_leases and schedule:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def _promote_l3_staging_leases(self, pending: list[L3StagingLease]) -> int:
        pending.sort(key=self._staging_lease_logical_fingerprint)
        fingerprints = [
            self._staging_lease_logical_fingerprint(lease) for lease in pending
        ]
        digest = self._logical_fingerprint_digest(fingerprints)
        self._require_storage_prefetch_rank_agreement(
            "staging promotion plan",
            {
                "entry_count": len(fingerprints),
                **{f"digest_{index}": value for index, value in enumerate(digest)},
            },
        )

        remaining_pages = _L3_STAGING_PROMOTION_QUANTUM_PAGES
        promoted_tokens = 0

        def promote(lease: L3StagingLease) -> None:
            nonlocal promoted_tokens, remaining_pages
            if remaining_pages == 0:
                return
            window_start = lease.reclaim_index_cursor
            window_end = min(
                len(lease.indices),
                window_start + remaining_pages * lease.quota.page_size,
            )
            if window_start >= window_end:
                return
            window_indices = lease.indices[window_start:window_end]
            live_positions = [
                window_start + position
                for position, value in enumerate(_index_values(window_indices))
                if value in lease.quota.in_use_indices
            ]
            remaining_pages -= (window_end - window_start) // lease.quota.page_size
            if not live_positions:
                lease.reclaim_index_cursor = window_end
                return
            selected_tokens = len(live_positions)
            selected_tokens -= selected_tokens % lease.quota.page_size
            if selected_tokens == 0:
                return
            promoted = lease.quota.promote(
                _take_indices(lease.indices, live_positions[:selected_tokens])
            )
            promoted_tokens += promoted
            if promoted == len(live_positions):
                lease.reclaim_index_cursor = window_end

        owner_groups: dict[int, list[L3StagingLease]] = {}
        unowned = []
        for lease in pending:
            if lease.owner_node is None:
                unowned.append(lease)
            else:
                owner_groups.setdefault(id(lease.owner_node), []).append(lease)

        for lease in unowned:
            promote(lease)
        for leases in owner_groups.values():
            base_leases = [
                lease for lease in leases if lease.component_type == BASE_COMPONENT_TYPE
            ]
            for lease in base_leases:
                promote(lease)
            if any(
                self._staging_lease_pending_tokens(lease) > 0 for lease in base_leases
            ):
                continue
            for lease in leases:
                if lease.component_type != BASE_COMPONENT_TYPE:
                    promote(lease)
        return promoted_tokens

    def _staging_discard_plan(
        self,
        lease: L3StagingLease,
        scan_budget: list[int],
    ) -> tuple[tuple[tuple[UnifiedTreeNode, TreeComponent, Any, int], ...], bool]:
        if lease.owner_node is None or lease.component_type is None:
            return (), False

        window_start = lease.reclaim_index_cursor
        window_end = min(
            len(lease.indices),
            window_start + _L3_STAGING_PROMOTION_QUANTUM_PAGES * lease.quota.page_size,
        )
        window_indices = lease.indices[window_start:window_end]
        live_indices = {
            value
            for value in _index_values(window_indices)
            if value in lease.quota.in_use_indices
        }
        if not live_indices:
            lease.reclaim_index_cursor = window_end
            lease.discard_scan_node = None
            lease.discard_scan_remaining = None
            return (), window_end < len(lease.indices)
        window_deferred = window_end < len(lease.indices)

        component = self.components.get(lease.component_type)
        if component is None:
            return (), False
        if lease.discard_scan_remaining is None:
            remaining = live_indices
            node = lease.owner_node
        else:
            remaining = set(lease.discard_scan_remaining).intersection(live_indices)
            node = lease.discard_scan_node
            if node is None:
                remaining = live_indices
                node = lease.owner_node

        plan = []
        while node is not self.root_node and remaining:
            if scan_budget[0] == 0:
                if plan:
                    lease.discard_scan_node = None
                    lease.discard_scan_remaining = None
                    return tuple(plan), True
                lease.discard_scan_node = node
                lease.discard_scan_remaining = frozenset(remaining)
                return (), True
            scan_budget[0] -= 1
            component_data = node.component_data[lease.component_type]
            host_value = component_data.host_value
            if host_value is not None:
                host_values = set(_index_values(host_value))
                overlap = remaining.intersection(host_values)
                if overlap:
                    if (
                        component_data.host_lock_ref != 0
                        or component_data.value is None
                        or not host_values.issubset(lease.quota.in_use_indices)
                    ):
                        lease.discard_scan_node = None
                        lease.discard_scan_remaining = None
                        return (), False
                    plan.append((node, component, lease.quota, len(host_values)))
                    remaining.difference_update(overlap)
            node = node.parent
            if len(plan) == _L3_STAGING_DISCARD_QUANTUM_ENTRIES:
                lease.discard_scan_node = None
                lease.discard_scan_remaining = None
                return tuple(plan), bool(remaining) or window_deferred

        lease.discard_scan_node = None
        lease.discard_scan_remaining = None
        return tuple(plan), window_deferred and not remaining

    @staticmethod
    def _staging_discard_fingerprint(
        entry: tuple[UnifiedTreeNode, TreeComponent, Any, int],
    ) -> tuple[int, ...]:
        node, component, quota, token_count = entry
        hash_values = tuple(node.hash_value or ())
        digest = hashlib.sha256()
        digest.update(quota.name.encode())
        digest.update(b"\0")
        digest.update(str(int(component.component_type)).encode())
        for hash_value in hash_values:
            digest.update(b"\0")
            digest.update(hash_value.encode())
        return (
            int(bool(hash_values)),
            int(component.component_type),
            token_count,
            len(hash_values),
            *(
                int.from_bytes(digest.digest()[offset : offset + 4], "big") & 0x7FFFFFFF
                for offset in range(0, digest.digest_size, 4)
            ),
        )

    def _discard_releasable_l3_staging(
        self, pending: list[L3StagingLease]
    ) -> tuple[int, bool]:
        load_back_active = int(bool(self.ongoing_load_back))
        load_back_extrema = self.synchronize_storage_prefetch_min_values(
            [load_back_active, -load_back_active]
        )
        if -load_back_extrema[1] != 0:
            return 0, False

        planned = {}
        scan_budget = [_L3_STAGING_DISCARD_SCAN_QUANTUM_NODES]
        scan_deferred = False
        for lease in pending:
            lease_plan, lease_deferred = self._staging_discard_plan(lease, scan_budget)
            scan_deferred = scan_deferred or lease_deferred
            for node, component, quota, token_count in lease_plan:
                planned[(node.id, component.component_type)] = (
                    node,
                    component,
                    quota,
                    token_count,
                )

        blocked_base_entries = set()
        for key, (node, component, _quota, _token_count) in planned.items():
            if component.component_type != BASE_COMPONENT_TYPE:
                continue
            if any(
                node.component_data[component_type].host_value is not None
                and (node.id, component_type) not in planned
                for component_type in self.components
                if component_type != BASE_COMPONENT_TYPE
            ):
                blocked_base_entries.add(key)
        for key in blocked_base_entries:
            del planned[key]

        entries = sorted(
            planned.values(),
            key=lambda entry: (
                entry[1].component_type == BASE_COMPONENT_TYPE,
                self._staging_discard_fingerprint(entry),
            ),
        )
        discard_deferred = (
            scan_deferred or len(entries) > _L3_STAGING_DISCARD_QUANTUM_ENTRIES
        )
        entries = entries[:_L3_STAGING_DISCARD_QUANTUM_ENTRIES]
        self._require_storage_prefetch_rank_agreement(
            "staging replica discard count", {"entries": len(entries)}
        )
        if not entries:
            return 0, discard_deferred

        fingerprints = [self._staging_discard_fingerprint(entry) for entry in entries]
        self._require_storage_prefetch_rank_agreement(
            "staging replica discard plan",
            {
                f"entry_{entry_index}_field_{field_index}": value
                for entry_index, fingerprint in enumerate(fingerprints)
                for field_index, value in enumerate(fingerprint)
            },
        )
        if any(fingerprint[0] == 0 for fingerprint in fingerprints):
            raise RuntimeError(
                "HiCache staging replica discard owner has no storage hashes"
            )

        quotas = self.cache_controller.l3_staging.quotas.values()
        discard_counts = {quota.name: 0 for quota in quotas}
        for _node, _component, quota, token_count in entries:
            discard_counts[quota.name] += token_count
        self._require_storage_prefetch_rank_agreement(
            "staging replica discard", discard_counts
        )
        for node, component, _quota, token_count in entries:
            _device_freed, host_freed = self._evict_component_and_detach_lru(
                node, component, target=EvictLayer.HOST
            )
            if host_freed != token_count:
                raise RuntimeError(
                    "HiCache staging replica discard freed an unexpected "
                    f"number of tokens: expected={token_count} actual={host_freed}"
                )
            self._update_evictable_leaf_sets(node)
        return len(entries), discard_deferred

    def _reclaim_l3_staging_headroom(self) -> int:
        controller = self.cache_controller
        reserve_tokens = (
            int(getattr(controller.l3_staging, "reserve_tokens", 0))
            if controller
            else 0
        )
        quota_count = (
            len(getattr(controller.l3_staging, "quotas", {})) if controller else 0
        )
        self._require_storage_prefetch_rank_agreement(
            "staging configuration",
            {"reserve_tokens": reserve_tokens, "quota_count": quota_count},
        )
        if reserve_tokens == 0:
            if controller is not None:
                controller.l3_staging.shortfall_tokens = 0
            return 0
        if controller is None or quota_count == 0:
            raise RuntimeError("HiCache L3 staging reserve has no installed quota")

        quotas = sorted(
            controller.l3_staging.quotas.values(), key=lambda quota: quota.name
        )
        head_count = min(
            len(self.l3_staging_pending_leases),
            self.l3_staging_reclaim_eligible,
            _L3_STAGING_RECLAIM_QUANTUM_LEASES,
        )
        head_fingerprints = [
            self._staging_lease_logical_fingerprint(lease)
            for lease in islice(self.l3_staging_pending_leases, head_count)
        ]
        head_digest = self._logical_fingerprint_digest(head_fingerprints)
        self._require_storage_prefetch_rank_agreement(
            "staging reclaim head",
            {
                "queue_length": len(self.l3_staging_pending_leases),
                "eligible": self.l3_staging_reclaim_eligible,
                "head_count": head_count,
                **{f"digest_{index}": value for index, value in enumerate(head_digest)},
            },
        )
        pending = []
        for _ in range(head_count):
            lease = self.l3_staging_pending_leases.popleft()
            self.l3_staging_reclaim_eligible -= 1
            key = self._staging_lease_key(lease)
            self.l3_staging_pending_keys.discard(key)
            previous_tokens = self.l3_staging_pending_tokens.pop(key, 0)
            self.l3_staging_pending_by_key.pop(key, None)
            self.l3_staging_pending_token_total -= previous_tokens
            if self._staging_lease_pending_tokens(lease) > 0:
                pending.append(lease)

        pending_by_quota = {
            id(quota): min(
                quota.used_tokens,
                sum(
                    self._staging_lease_pending_tokens(lease)
                    for lease in pending
                    if lease.quota is quota
                ),
            )
            for quota in quotas
        }
        self._require_storage_prefetch_rank_agreement(
            "staging quota",
            {
                value_name: value
                for quota in quotas
                for value_name, value in (
                    (f"{quota.name}.capacity", quota.capacity_tokens),
                    (f"{quota.name}.used", quota.used_tokens),
                    (f"{quota.name}.available", quota.available_tokens),
                    (
                        f"{quota.name}.ordinary_available",
                        quota.original_available_size(),
                    ),
                    (f"{quota.name}.pending", pending_by_quota[id(quota)]),
                )
            },
        )

        promoted_tokens = self._promote_l3_staging_leases(pending)
        pending = [
            lease for lease in pending if self._staging_lease_pending_tokens(lease) > 0
        ]
        pending_by_quota = defaultdict(int)
        for lease in pending:
            pending_by_quota[id(lease.quota)] += self._staging_lease_pending_tokens(
                lease
            )
        pending_by_quota = {
            id(quota): min(quota.used_tokens, pending_by_quota[id(quota)])
            for quota in quotas
        }

        mem_pool_host = controller.mem_pool_host
        anchor_pool = getattr(
            getattr(mem_pool_host, "anchor_entry", None),
            "host_pool",
            mem_pool_host,
        )
        evicted_values = {}
        remaining_eviction_pages = _L3_STAGING_EVICTION_QUANTUM_PAGES
        continue_eviction = False
        with bounded_host_eviction_scan(
            _L3_STAGING_HOST_EVICTION_SCAN_QUANTUM
        ) as host_scan_budget:
            for quota in quotas:
                requested = pending_by_quota.get(id(quota), 0)
                eviction_needed = max(0, requested - quota.original_available_size())
                if eviction_needed == 0:
                    continue
                if remaining_eviction_pages == 0:
                    continue_eviction = True
                    continue
                eviction_pages = min(
                    math.ceil(eviction_needed / quota.page_size),
                    remaining_eviction_pages,
                )
                eviction_request = eviction_pages * quota.page_size
                remaining_eviction_pages -= eviction_pages
                entry = next(
                    (
                        candidate
                        for candidate in getattr(mem_pool_host, "entries", ())
                        if candidate.host_pool is quota.pool
                    ),
                    None,
                )
                if quota.pool is anchor_pool:
                    evicted = self.evict_host(eviction_request)
                else:
                    evicted = (
                        entry.host_evict_fn(eviction_request)
                        if entry is not None and entry.host_evict_fn is not None
                        else 0
                    )
                evicted = int(evicted or 0)
                evicted_values[quota.name] = evicted
                if 0 < evicted < eviction_needed:
                    continue_eviction = True
                if evicted < eviction_needed and host_scan_budget.remaining_nodes == 0:
                    continue_eviction = True

        if evicted_values:
            self._require_storage_prefetch_rank_agreement(
                "staging promotion eviction", evicted_values
            )
            self._require_storage_prefetch_rank_agreement(
                "staging post-eviction quota",
                {
                    value_name: value
                    for quota in quotas
                    for value_name, value in (
                        (f"{quota.name}.used", quota.used_tokens),
                        (f"{quota.name}.available", quota.available_tokens),
                        (
                            f"{quota.name}.ordinary_available",
                            quota.original_available_size(),
                        ),
                    )
                },
            )
            promoted_tokens += self._promote_l3_staging_leases(pending)

        pending = [
            lease for lease in pending if self._staging_lease_pending_tokens(lease) > 0
        ]
        discarded_entries, discard_deferred = self._discard_releasable_l3_staging(
            pending
        )
        pending = [
            lease for lease in pending if self._staging_lease_pending_tokens(lease) > 0
        ]
        made_progress = (
            promoted_tokens > 0
            or discarded_entries > 0
            or any(evicted > 0 for evicted in evicted_values.values())
        )
        if made_progress:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)
        self._require_storage_prefetch_rank_agreement(
            "staging continuation",
            {
                "made_progress": int(made_progress),
                "continue_eviction": int(continue_eviction),
                "discard_deferred": int(discard_deferred),
            },
        )
        self._enqueue_staging_leases(
            pending,
            schedule=False,
            eligible=made_progress or continue_eviction or discard_deferred,
        )
        shortfall = self.l3_staging_pending_token_total
        controller.l3_staging.shortfall_tokens = shortfall
        if shortfall and (
            self.l3_staging_reclaim_eligible > 0
            or continue_eviction
            or discard_deferred
        ):
            self.storage_state_change_resources.add(CheckpointRetryResource.SCHEDULER)
        return shortfall

    def pin_storage_prefetch_admission(
        self,
        request_id: str,
        node: UnifiedTreeNode,
        occupied_tokens: int,
        staging_leases: tuple[L3StagingLease, ...] = (),
    ) -> None:
        existing = self.storage_prefetch_admission_host_locks.get(request_id)
        if (
            existing is not None
            and existing.node is node
            and existing.occupied_tokens == occupied_tokens
            and tuple(map(self._staging_lease_key, existing.staging_leases))
            == tuple(map(self._staging_lease_key, staging_leases))
        ):
            return
        if (
            existing is not None
            and self.cache_controller is not None
            and existing.occupied_tokens
            > self.cache_controller.prefetch_tokens_occupied
        ):
            raise RuntimeError(
                "HiCache prefetch admission accounting underflow while replacing "
                f"request {request_id}"
            )
        lock_params = (
            existing.lock_params
            if existing is not None and existing.node is node
            else self.inc_host_lock_ref(node).to_dec_params()
        )
        self.storage_prefetch_admission_host_locks[request_id] = (
            StoragePrefetchAdmissionPin(
                node=node,
                lock_params=lock_params,
                occupied_tokens=occupied_tokens,
                staging_leases=staging_leases,
            )
        )
        if existing is not None:
            if existing.node is not node:
                self.dec_host_lock_ref(existing.node, existing.lock_params)
            self.cache_controller.prefetch_tokens_occupied -= existing.occupied_tokens
            self._enqueue_staging_leases(existing.staging_leases)
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def pin_storage_prefetch_device_admission(
        self, request_id: str, node: UnifiedTreeNode
    ) -> None:
        existing = self.storage_prefetch_admission_device_locks.get(request_id)
        if existing is not None and existing.node is node:
            return
        lock_params = self.inc_lock_ref(node).to_dec_params()
        self.storage_prefetch_admission_device_locks[request_id] = (
            StoragePrefetchDeviceAdmissionPin(node=node, lock_params=lock_params)
        )
        if existing is not None:
            self.dec_lock_ref(existing.node, existing.lock_params)

    def release_storage_prefetch_admission(self, request_id: str) -> None:
        host_pin = self.storage_prefetch_admission_host_locks.get(request_id)
        if (
            host_pin is not None
            and self.cache_controller is not None
            and host_pin.occupied_tokens
            > self.cache_controller.prefetch_tokens_occupied
        ):
            raise RuntimeError(
                "HiCache prefetch admission accounting underflow while releasing "
                f"request {request_id}"
            )
        device_pin = self.storage_prefetch_admission_device_locks.pop(request_id, None)
        if device_pin is not None:
            self.dec_lock_ref(device_pin.node, device_pin.lock_params)
        host_pin = self.storage_prefetch_admission_host_locks.pop(request_id, None)
        if host_pin is not None:
            self.dec_host_lock_ref(host_pin.node, host_pin.lock_params)
            if self.cache_controller is not None:
                self.cache_controller.prefetch_tokens_occupied -= (
                    host_pin.occupied_tokens
                )
            self._enqueue_staging_leases(host_pin.staging_leases)
        if device_pin is not None or host_pin is not None:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def complete_storage_prefetch_admission(self, request_id: str) -> None:
        self.release_storage_prefetch_admission(request_id)
        self.storage_checkpoint_dependency_results.pop(request_id, None)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            if req.storage_checkpoint_handle is not None:
                self.storage_checkpoint_registry.fail(req.storage_checkpoint_handle)
                self._release_storage_checkpoint(
                    req.storage_checkpoint_handle, release_staging=True
                )
            return

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            if req.storage_checkpoint_handle is not None:
                self.storage_checkpoint_registry.fail(req.storage_checkpoint_handle)
                self._release_storage_checkpoint(
                    req.storage_checkpoint_handle, release_staging=True
                )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        result = None
        insert_params = None
        radix_key = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.cache_protected_len,
                priority=getattr(req, "priority", 0) or 0,
            )

            # components prepare insert data + return effective cache_len
            effective_cache_len = len(token_ids)
            for comp in self._components_tuple:
                cl = comp.prepare_for_caching_req(
                    req=req,
                    insert_params=insert_params,
                    token_ids_len=len(token_ids),
                    is_finished=True,
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

            # Truncate if needed
            if effective_cache_len < len(token_ids):
                free_start = max(effective_cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[free_start:])
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
            skip_swa=getattr(req, "swa_prefix_lock_released", False),
        )

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )
        if req.storage_checkpoint_handle is not None:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            if isinstance(req.finished_reason, FINISH_ABORT):
                self.storage_checkpoint_registry.fail(req.storage_checkpoint_handle)
                self._release_storage_checkpoint(
                    req.storage_checkpoint_handle, release_staging=True
                )
            else:
                self._create_storage_checkpoint(
                    req.storage_checkpoint_handle, radix_key
                )

    def reserve_storage_checkpoint(self, handle: str) -> None:
        self._release_storage_checkpoint(handle, release_staging=True)
        self.storage_checkpoint_registry.reserve(handle)

    def _release_storage_checkpoint(
        self, handle: str, *, release_staging: bool = False
    ) -> None:
        checkpoint = self.storage_pending_checkpoints.pop(handle, None)
        if checkpoint is None:
            return
        released_pin = False
        if checkpoint.device_pin is not None:
            self.dec_lock_ref(*checkpoint.device_pin)
            released_pin = True
        if checkpoint.host_pin is not None:
            self.dec_host_lock_ref(*checkpoint.host_pin)
            released_pin = True
        if release_staging or (
            self.storage_checkpoint_registry.get_state(handle)
            is not StorageCheckpointState.PENDING
        ):
            self._enqueue_staging_leases(checkpoint.staging_leases)
        if released_pin:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def _checkpoint_path(self, radix_key: RadixKey) -> Optional[list[UnifiedTreeNode]]:
        target_node = self.match_prefix(
            MatchPrefixParams(key=radix_key)
        ).best_match_node
        path = []
        node = target_node
        while node is not self.root_node:
            path.append(node)
            node = node.parent
        path.reverse()
        if not path or any(node.hash_value is None for node in path):
            return None
        matched_tokens = sum(len(node.hash_value) * self.page_size for node in path)
        if matched_tokens != len(radix_key):
            return None
        return self._align_checkpoint_swa_boundary(path)

    def _checkpoint_swa_prefix_tokens(self, total_tokens: int) -> Optional[int]:
        component = self.components.get(ComponentType.SWA)
        if component is None:
            return None
        window_tokens = (
            (component.sliding_window_size + self.page_size - 1) // self.page_size
        ) * self.page_size
        return max(total_tokens - window_tokens, 0)

    def _align_checkpoint_swa_boundary(
        self, path: list[UnifiedTreeNode]
    ) -> list[UnifiedTreeNode]:
        prefix_tokens = self._checkpoint_swa_prefix_tokens(
            sum(len(node.key) for node in path)
        )
        if not prefix_tokens:
            return path
        node_start = 0
        for index, node in enumerate(path):
            node_end = node_start + len(node.key)
            if node_start < prefix_tokens < node_end:
                prefix_node = self._split_node(
                    node.key, node, prefix_tokens - node_start
                )
                return [*path[:index], prefix_node, node, *path[index + 1 :]]
            node_start = node_end
        return path

    @staticmethod
    def _checkpoint_path_hashes(path: Sequence[UnifiedTreeNode]) -> tuple[str, ...]:
        return tuple(
            page_hash for node in path for page_hash in (node.hash_value or ())
        )

    def _pin_storage_checkpoint_path(
        self,
        checkpoint: PendingStorageCheckpoint,
        path: list[UnifiedTreeNode],
    ) -> None:
        target = path[-1]
        existing_device = checkpoint.device_pin
        existing_host = checkpoint.host_pin
        released_pin = False
        if target.backuped:
            if checkpoint.host_pin is None or checkpoint.host_pin[0] is not target:
                checkpoint.host_pin = (
                    target,
                    self.inc_host_lock_ref(target).to_dec_params(),
                )
            if (checkpoint.requires_device_pin or existing_device is not None) and (
                checkpoint.device_pin is None or checkpoint.device_pin[0] is not target
            ):
                checkpoint.device_pin = (
                    target,
                    self.inc_lock_ref(target).to_dec_params(),
                )
        elif checkpoint.device_pin is None or checkpoint.device_pin[0] is not target:
            checkpoint.device_pin = (
                target,
                self.inc_lock_ref(target).to_dec_params(),
            )

        if existing_host is not None and existing_host != checkpoint.host_pin:
            self.dec_host_lock_ref(*existing_host)
            released_pin = True
        if existing_device is not None and existing_device != checkpoint.device_pin:
            self.dec_lock_ref(*existing_device)
            released_pin = True
        if not target.backuped and existing_host is not None:
            checkpoint.host_pin = None
        checkpoint.path = tuple(path)
        if released_pin:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def _capture_checkpoint_staging(
        self,
        checkpoint: PendingStorageCheckpoint,
        nodes: Sequence[UnifiedTreeNode],
    ) -> None:
        if not checkpoint.requires_device_pin:
            checkpoint.requires_device_pin = any(
                node.component_data[component.component_type].value is not None
                and node.component_data[component.component_type].host_value is None
                for node in nodes
                for component in self._components_tuple
                if component.component_type != BASE_COMPONENT_TYPE
            )
        if not checkpoint.staging_lease_keys and checkpoint.staging_leases:
            checkpoint.staging_lease_keys.update(
                self._staging_lease_key(lease) for lease in checkpoint.staging_leases
            )
        for lease in self._checkpoint_staging_leases(nodes):
            key = self._staging_lease_key(lease)
            if key not in checkpoint.staging_lease_keys:
                checkpoint.staging_leases.append(lease)
                checkpoint.staging_lease_keys.add(key)

    def _l3_staging_capacity_state(
        self,
        base_tokens: int,
        transfers: Sequence[PoolTransfer],
        extra_requirements: Optional[dict[PoolName, int]] = None,
    ) -> Optional[str]:
        controller = self.cache_controller
        if controller is None or not controller.l3_staging.quotas:
            return None
        mem_pool_host = controller.mem_pool_host
        anchor_pool = getattr(
            getattr(mem_pool_host, "anchor_entry", None),
            "host_pool",
            mem_pool_host,
        )
        requirements: dict[int, int] = defaultdict(int)
        if base_tokens:
            requirements[id(anchor_pool)] = base_tokens
        for transfer in transfers:
            if transfer.indices_from_pool is not None:
                continue
            if transfer.device_indices is None:
                continue
            entry = mem_pool_host.entry_map.get(transfer.name)
            if entry is not None:
                requirements[id(entry.host_pool)] += len(transfer.device_indices)
        for pool_name, required in (extra_requirements or {}).items():
            entry = mem_pool_host.entry_map.get(pool_name)
            if entry is not None:
                requirements[id(entry.host_pool)] += required

        values = {
            value_name: value
            for pool_id, quota in controller.l3_staging.quotas.items()
            for value_name, value in (
                (f"{quota.name}.required", requirements.get(pool_id, 0)),
                (f"{quota.name}.capacity", quota.capacity_tokens),
                (f"{quota.name}.available", quota.available_tokens),
            )
        }
        values["missing_pool_count"] = sum(
            pool_id not in controller.l3_staging.quotas for pool_id in requirements
        )
        self._require_storage_prefetch_rank_agreement("staging reservation", values)
        if any(
            pool_id not in controller.l3_staging.quotas
            or required > controller.l3_staging.quotas[pool_id].capacity_tokens
            for pool_id, required in requirements.items()
        ):
            return _CHECKPOINT_STAGING_CAPACITY_EXCEEDED
        if any(
            required > controller.l3_staging.quotas[pool_id].available_tokens
            for pool_id, required in requirements.items()
        ):
            return "host"
        return None

    def _prefetch_staging_capacity_state(self, prefetch_tokens: int) -> Optional[str]:
        controller = self.cache_controller
        if controller is None or not controller.l3_staging.quotas:
            return None
        mem_pool_host = controller.mem_pool_host
        extra_requirements = {}
        swa_component = self.components.get(ComponentType.SWA)
        if swa_component is not None:
            window_tokens = (
                (swa_component.sliding_window_size + self.page_size - 1)
                // self.page_size
            ) * self.page_size
            if prefetch_tokens >= window_tokens:
                extra_requirements[PoolName.SWA] = window_tokens
        if ComponentType.MAMBA in self.components:
            entry = mem_pool_host.entry_map.get(PoolName.MAMBA)
            if entry is not None:
                extra_requirements[PoolName.MAMBA] = entry.host_pool.page_size
        return self._l3_staging_capacity_state(prefetch_tokens, (), extra_requirements)

    def _queue_storage_checkpoint_retry(
        self,
        checkpoint: PendingStorageCheckpoint,
        blocked_reason: str,
    ) -> None:
        resource = _CHECKPOINT_BLOCKED_RESOURCES.get(blocked_reason)
        if resource is None:
            raise ValueError(
                f"Unknown storage checkpoint block reason: {blocked_reason}"
            )
        if checkpoint.retry_queued and checkpoint.blocked_resource is resource:
            return
        checkpoint.blocked_resource = resource
        checkpoint.blocked_generation = self.storage_checkpoint_retries.generations[
            resource
        ]
        checkpoint.retry_ticket += 1
        checkpoint.retry_queued = True
        self.storage_checkpoint_retries.queues[resource].append(
            (checkpoint.handle, checkpoint.generation, checkpoint.retry_ticket)
        )
        if resource is CheckpointRetryResource.SCHEDULER:
            self.storage_state_change_resources.add(resource)

    def _checkpoint_skip_components(
        self, node_end: int, total_tokens: int
    ) -> frozenset[ComponentType]:
        swa_prefix_tokens = self._checkpoint_swa_prefix_tokens(total_tokens)
        if swa_prefix_tokens is not None and node_end <= swa_prefix_tokens:
            return frozenset({ComponentType.SWA})
        return frozenset()

    def _drive_storage_checkpoint(
        self,
        checkpoint: PendingStorageCheckpoint,
        path: Sequence[UnifiedTreeNode],
    ) -> str:
        write_tracker = self.cache_controller.storage_write_tracker
        page_offset = checkpoint.cursor_pages
        scanned_nodes = 0
        for node in path:
            node_hashes = node.hash_value or []
            node_end = page_offset + len(node_hashes)
            if scanned_nodes == _CHECKPOINT_SCAN_QUANTUM:
                self.storage_state_change_resources.add(
                    CheckpointRetryResource.SCHEDULER
                )
                return "continuation"
            scanned_nodes += 1
            expected = checkpoint.path_hashes[page_offset:node_end]
            if tuple(node_hashes) != expected:
                self.storage_checkpoint_registry.fail(checkpoint.handle)
                return "storage"
            durable_hashes = write_tracker.get_durable_hashes(node_hashes)
            has_pending_write = write_tracker.has_pending(node_hashes)
            self._require_storage_prefetch_rank_agreement(
                "checkpoint node state",
                {
                    "durable_hashes": len(durable_hashes),
                    "pending_write": int(has_pending_write),
                    "write_through_pending": int(
                        node.write_through_pending_id is not None
                    ),
                    "host_resident": int(node.backuped),
                    "node_pages": len(node_hashes),
                },
            )
            if len(durable_hashes) == len(set(node_hashes)):
                checkpoint.cursor_pages = node_end
                page_offset = node_end
                continue
            if has_pending_write or node.write_through_pending_id is not None:
                return "storage"

            skip_components = self._checkpoint_skip_components(
                node_end * self.page_size, len(checkpoint.radix_key)
            )
            if not node.backuped:
                preview_transfers = [
                    transfer
                    for component in self._components_tuple
                    if component.component_type != BASE_COMPONENT_TYPE
                    and component.component_type not in skip_components
                    for transfer in (
                        component.build_hicache_transfers(
                            node, CacheTransferPhase.BACKUP_HOST
                        )
                        or ()
                    )
                ]
                device_value = node.component_data[BASE_COMPONENT_TYPE].value
                capacity_state = self._l3_staging_capacity_state(
                    len(device_value) if device_value is not None else 0,
                    preview_transfers,
                )
                if capacity_state is not None:
                    return capacity_state
                with l3_staging_allocation():
                    written = self.write_backup(node, skip_components=skip_components)
                self._require_storage_prefetch_rank_agreement(
                    "checkpoint host backup", {"written_tokens": written}
                )
                self._capture_checkpoint_staging(checkpoint, (node,))
                return "storage" if written > 0 else "host"

            with l3_staging_allocation():
                auxiliary_started = self._write_missing_aux_backup(
                    node, skip_components=skip_components
                )
            self._require_storage_prefetch_rank_agreement(
                "checkpoint auxiliary backup",
                {
                    "state": (
                        1
                        if auxiliary_started is True
                        else (
                            0
                            if auxiliary_started is False
                            else (
                                -2
                                if auxiliary_started
                                == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED
                                else -1
                            )
                        )
                    )
                },
            )
            self._capture_checkpoint_staging(checkpoint, (node,))
            if auxiliary_started == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED:
                return _CHECKPOINT_STAGING_CAPACITY_EXCEEDED
            if auxiliary_started is False:
                return "auxiliary"
            if auxiliary_started is True:
                return "storage"

            operation_id = self.write_backup_storage(node)
            self._require_storage_prefetch_rank_agreement(
                "checkpoint storage submission",
                {"submitted": int(operation_id is not None)},
            )
            if operation_id is None:
                return "storage"
            checkpoint.active_operation_ids.add(operation_id)
            self.storage_checkpoint_owners_by_operation[operation_id] = (
                checkpoint.handle,
                checkpoint.generation,
            )
            self._capture_checkpoint_staging(checkpoint, (node,))
            return "storage"
        return "storage"

    @staticmethod
    def _reset_checkpoint_path_scan(checkpoint: PendingStorageCheckpoint) -> None:
        checkpoint.path_scan_node = None
        checkpoint.path_scan_reverse_nodes = deque()
        checkpoint.path_scan_drive_reverse_nodes = deque()
        checkpoint.path_scan_hash_cursor = 0
        checkpoint.path_scan_key_tokens = 0
        checkpoint.path_scan_valid = True
        checkpoint.path_scan_tree_generation = -1

    def _scan_storage_checkpoint_path(
        self, checkpoint: PendingStorageCheckpoint
    ) -> tuple[Optional[list[UnifiedTreeNode]], Optional[deque[UnifiedTreeNode]], bool]:
        if (
            checkpoint.path_scan_reverse_nodes
            and checkpoint.path_scan_tree_generation
            != self.storage_radix_tree_generation
        ):
            self._reset_checkpoint_path_scan(checkpoint)
        if not checkpoint.path_scan_reverse_nodes:
            if not checkpoint.path:
                return None, None, True
            checkpoint.path_scan_node = checkpoint.path[-1]
            checkpoint.path_scan_hash_cursor = len(checkpoint.path_hashes)
            checkpoint.path_scan_key_tokens = 0
            checkpoint.path_scan_valid = True
            checkpoint.path_scan_tree_generation = self.storage_radix_tree_generation

        node = checkpoint.path_scan_node
        scanned_nodes = []
        while (
            node is not self.root_node and len(scanned_nodes) < _CHECKPOINT_SCAN_QUANTUM
        ):
            if node is None:
                checkpoint.path_scan_valid = False
                break
            scanned_nodes.append(node)
            node_hashes = tuple(node.hash_value or ())
            node_end = checkpoint.path_scan_hash_cursor
            hash_start = checkpoint.path_scan_hash_cursor - len(node_hashes)
            if (
                hash_start < 0
                or checkpoint.path_hashes[hash_start : checkpoint.path_scan_hash_cursor]
                != node_hashes
            ):
                checkpoint.path_scan_valid = False
            else:
                checkpoint.path_scan_hash_cursor = hash_start
            if node.key is None:
                checkpoint.path_scan_valid = False
            else:
                checkpoint.path_scan_key_tokens += len(node.key)
            checkpoint.path_scan_reverse_nodes.appendleft(node)
            if node_end > checkpoint.cursor_pages:
                checkpoint.path_scan_drive_reverse_nodes.appendleft(node)
            node = node.parent

        self._capture_checkpoint_staging(checkpoint, scanned_nodes)
        checkpoint.path_scan_node = node
        complete = node is self.root_node or node is None
        if complete:
            checkpoint.path_scan_valid = (
                checkpoint.path_scan_valid
                and node is self.root_node
                and checkpoint.path_scan_hash_cursor == 0
                and checkpoint.path_scan_key_tokens == len(checkpoint.radix_key)
            )
        self._require_storage_prefetch_rank_agreement(
            "checkpoint path scan",
            {
                "complete": int(complete),
                "valid": int(checkpoint.path_scan_valid),
                "scanned_nodes": len(scanned_nodes),
                "accumulated_nodes": len(checkpoint.path_scan_reverse_nodes),
                "remaining_hashes": checkpoint.path_scan_hash_cursor,
                "key_tokens": checkpoint.path_scan_key_tokens,
                "tree_generation": self.storage_radix_tree_generation,
                **{
                    f"path_digest_{index}": value
                    for index, value in enumerate(checkpoint.path_hash_digest)
                },
            },
        )
        if not complete:
            self.storage_state_change_resources.add(CheckpointRetryResource.SCHEDULER)
            return None, None, False

        valid = checkpoint.path_scan_valid
        path = list(checkpoint.path_scan_reverse_nodes)
        drive_path = checkpoint.path_scan_drive_reverse_nodes
        checkpoint.path_scan_drive_reverse_nodes = deque()
        self._reset_checkpoint_path_scan(checkpoint)
        if not valid:
            return None, None, True
        self._pin_storage_checkpoint_path(checkpoint, path)
        return path, drive_path, True

    def retry_storage_checkpoint(self, handle: str) -> None:
        state = self.storage_checkpoint_registry.get_state(handle)
        if state is not StorageCheckpointState.PENDING:
            self._release_storage_checkpoint(handle)
            return
        checkpoint = self.storage_pending_checkpoints.get(handle)
        if checkpoint is None:
            self.storage_checkpoint_registry.fail(handle)
            return
        path, drive_path, scan_complete = self._scan_storage_checkpoint_path(checkpoint)
        if not scan_complete:
            self._queue_storage_checkpoint_retry(checkpoint, "continuation")
            return
        if path is None or drive_path is None:
            self.storage_checkpoint_registry.fail(handle)
            self._release_storage_checkpoint(handle, release_staging=True)
            return
        blocked_reason = self._drive_storage_checkpoint(checkpoint, drive_path)
        if blocked_reason == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED:
            self.storage_checkpoint_registry.fail(handle)
            self._release_storage_checkpoint(handle, release_staging=True)
            return
        if (
            self.storage_checkpoint_registry.get_state(handle)
            is not StorageCheckpointState.PENDING
        ):
            self._release_storage_checkpoint(handle)
            return
        self._queue_storage_checkpoint_retry(checkpoint, blocked_reason)

    def _drive_storage_checkpoint_retries(
        self, changed_resources: set[CheckpointRetryResource]
    ) -> None:
        retries = self.storage_checkpoint_retries
        for resource in changed_resources:
            retries.generations[resource] += 1
        continuation_resources = set(retries.continuation_resources)
        retries.continuation_resources.clear()
        eligible_resources = changed_resources | continuation_resources
        resources = tuple(CheckpointRetryResource)

        def ticket_is_eligible(
            resource: CheckpointRetryResource,
            ticket: tuple[str, int, int],
        ) -> bool:
            checkpoint = self.storage_pending_checkpoints.get(ticket[0])
            stale = (
                checkpoint is None
                or checkpoint.generation != ticket[1]
                or checkpoint.retry_ticket != ticket[2]
                or checkpoint.blocked_resource is not resource
            )
            return stale or (
                checkpoint.blocked_generation < retries.generations[resource]
            )

        for _ in range(_CHECKPOINT_RETRY_QUANTUM):
            selected_resource = None
            selected_ticket = None
            for offset in range(len(resources)):
                index = (retries.cursor + offset) % len(resources)
                resource = resources[index]
                queue = retries.queues[resource]
                if resource not in eligible_resources or not queue:
                    continue
                ticket = queue[0]
                if not ticket_is_eligible(resource, ticket):
                    continue
                selected_resource = resource
                selected_ticket = ticket
                retries.cursor = (index + 1) % len(resources)
                break
            handle = selected_ticket[0] if selected_ticket is not None else ""
            digest = hashlib.sha256(handle.encode()).digest()
            self._require_storage_prefetch_rank_agreement(
                "checkpoint retry ticket",
                {
                    "present": int(selected_ticket is not None),
                    "resource": (
                        resources.index(selected_resource)
                        if selected_resource is not None
                        else -1
                    ),
                    "generation": (
                        selected_ticket[1] if selected_ticket is not None else -1
                    ),
                    "sequence": (
                        selected_ticket[2] if selected_ticket is not None else -1
                    ),
                    "handle_0": int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
                    "handle_1": int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
                },
            )
            if selected_resource is None or selected_ticket is None:
                break
            retries.queues[selected_resource].popleft()
            checkpoint = self.storage_pending_checkpoints.get(selected_ticket[0])
            if (
                checkpoint is None
                or checkpoint.generation != selected_ticket[1]
                or checkpoint.retry_ticket != selected_ticket[2]
            ):
                continue
            checkpoint.retry_queued = False
            self.retry_storage_checkpoint(checkpoint.handle)

        remaining_resources = {
            resource
            for resource in eligible_resources
            if retries.queues[resource]
            and ticket_is_eligible(resource, retries.queues[resource][0])
        }
        self._require_storage_prefetch_rank_agreement(
            "checkpoint retry continuation",
            {
                resource.name: int(resource in remaining_resources)
                for resource in resources
            },
        )
        retries.continuation_resources.update(remaining_resources)
        if remaining_resources:
            self.storage_state_change_resources.add(CheckpointRetryResource.SCHEDULER)

    def _create_storage_checkpoint(
        self, handle: str, radix_key: Optional[RadixKey]
    ) -> None:
        if (
            radix_key is None
            or not self.enable_storage
            or self.cache_controller is None
        ):
            self._release_storage_checkpoint(handle, release_staging=True)
            self.storage_checkpoint_registry.fail(handle)
            return

        path = self._checkpoint_path(radix_key)
        if path is None:
            self._release_storage_checkpoint(handle, release_staging=True)
            self.storage_checkpoint_registry.fail(handle)
            return

        self._release_storage_checkpoint(handle, release_staging=True)
        self.storage_checkpoint_generation += 1
        page_hashes = self._checkpoint_path_hashes(path)
        pending = PendingStorageCheckpoint(
            handle=handle,
            generation=self.storage_checkpoint_generation,
            radix_key=radix_key,
            path=tuple(path),
            path_hashes=page_hashes,
        )
        self.storage_pending_checkpoints[handle] = pending
        self._capture_checkpoint_staging(pending, path)
        self._pin_storage_checkpoint_path(pending, path)
        self._reclaim_l3_staging_headroom()

        write_tracker = self.cache_controller.storage_write_tracker
        checkpoint = self.storage_checkpoint_registry.create(
            handle,
            page_hashes,
            durable_hashes=write_tracker.get_durable_hashes(page_hashes),
            has_pending=write_tracker.has_pending,
        )
        if checkpoint.state is StorageCheckpointState.READY:
            self._release_storage_checkpoint(handle)
            return

        blocked_reason = self._drive_storage_checkpoint(pending, path)
        if blocked_reason == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED:
            self.storage_checkpoint_registry.fail(handle)
            self._release_storage_checkpoint(handle, release_staging=True)
            logger.error(
                "HiCache checkpoint failed handle=%s reason=staging-capacity",
                handle,
            )
            return
        self._queue_storage_checkpoint_retry(pending, blocked_reason)
        logger.info(
            "HiCache checkpoint deferred handle=%s reason=%s staging=%s",
            handle,
            blocked_reason,
            self.cache_controller.l3_staging.usage(),
        )

    def _write_missing_aux_backup(
        self,
        node: UnifiedTreeNode,
        skip_components: frozenset[ComponentType] = frozenset(),
    ) -> bool | str | None:
        if self.cache_controller is None:
            return False

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            if comp.component_type in skip_components:
                continue
            component_data = node.component_data[comp.component_type]
            if component_data.value is None or component_data.host_value is not None:
                continue
            transfers = comp.build_hicache_transfers(
                node, CacheTransferPhase.BACKUP_HOST
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        if not comp_xfers:
            return None

        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        if device_value is None:
            return False
        aux_xfers = [transfer for xfers in comp_xfers.values() for transfer in xfers]
        capacity_state = self._l3_staging_capacity_state(0, aux_xfers)
        if capacity_state is not None:
            return (
                capacity_state
                if capacity_state == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED
                else False
            )
        host_indices = self.cache_controller.write(
            device_value[:0], node_id=node.id, extra_pools=aux_xfers
        )
        if host_indices is None:
            return False
        assert host_indices.numel() == 0

        for component_type, transfers in comp_xfers.items():
            self.components[component_type].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=transfers,
            )

        lock_params = self.inc_lock_ref(node).to_dec_params()
        self._track_write_through_node(node, lock_params)
        return True

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.cache_protected_len,
            chunked=chunked,
            priority=getattr(req, "priority", 0) or 0,
        )
        effective_cache_len = len(token_ids)
        for comp in self._components_tuple:
            cl = comp.prepare_for_caching_req(
                req=req,
                insert_params=insert_params,
                token_ids_len=len(token_ids),
                is_finished=False,
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(
                    req, effective_cache_len - 1, insert_params
                )

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        lock_result = self.inc_lock_ref(new_last_node)

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        # The rematch acquired a new SWA prefix lock.
        req.swa_prefix_lock_released = False

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req,
                is_finished=False,
                insert_result=result,
                insert_params=insert_params,
            )

    # ---- Internal Helpers ----

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[torch.Tensor], UnifiedTreeNode, UnifiedTreeNode, int]:
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
        separate_device_match = self.cache_controller is not None
        if separate_device_match:
            validators = tuple(
                comp.create_match_validator() for comp in self._components_tuple
            )
            device_validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
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
            if prefix_len < len(child.key):
                node = self._split_node(child.key, child, prefix_len)
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
        )

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[torch.Tensor],
        best_match_node: UnifiedTreeNode,
        best_match_device_node: UnifiedTreeNode,
        best_match_device_value_len: int,
    ) -> MatchResult:
        node_update = best_match_node
        for comp in self._components_tuple:
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
            best_match_node
            if self.cache_controller is not None
            else best_match_device_node
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
        )

        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_match_device_value_len,
            )
        return result

    def _split_node(
        self, key: RadixKey, child: UnifiedTreeNode, split_len: int
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=child.priority)
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

        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])

        self._for_each_component_lru(
            new_node, UnifiedLRUList.insert_mru, skip_existing=True
        )
        self._for_each_component_lru(
            child, UnifiedLRUList.insert_mru, skip_existing=True
        )
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        self.storage_radix_tree_generation += 1
        return new_node

    def _touch_node(self, node: UnifiedTreeNode):
        node.last_access_time = get_and_increase_time_counter()
        if node != self.root_node:
            for comp in self._components_tuple:
                if comp.component_type == BASE_COMPONENT_TYPE:
                    continue
                comp.refresh_lru(LRURefreshPhase.WALKDOWN, node, self.root_node)

    def _add_new_node(
        self,
        parent: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components, priority=priority)
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
        self.storage_radix_tree_generation += 1
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

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        params: InsertParams,
    ) -> InsertResult:
        priority = params.priority
        if priority is None:
            priority = 0
        self._touch_node(node)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)
            node.priority = max(node.priority, priority)

            if node.evicted:
                self._unevict_node_on_insert(node, value[:prefix_len])
                # FULL was restored from the request's fresh KV. Aux
                # components (e.g. SWA) may still hold tombstones and need
                # to rebuild their value from the same slice.
                for component in self._components_tuple:
                    if component.component_type == BASE_COMPONENT_TYPE:
                        continue
                    component.recover_after_unevict(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        params=params,
                    )
            else:
                value_slice = value[:prefix_len]
                consumed_from = prefix_len
                # Let each component claim ownership of overlapping KV slots
                for component in self._components_tuple:
                    comp_consumed_from = component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )
                    consumed_from = min(consumed_from, comp_consumed_from)

                dup_start = max(0, params.prev_prefix_len - total_prefix_length)
                if dup_start < consumed_from:
                    self.token_to_kv_pool_allocator.free(
                        value_slice[dup_start:consumed_from]
                    )

            self._inc_hit_count(node, params.chunked)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

        is_new_leaf = False
        # Create new leaf for remaining suffix. A leaf survives on its Full
        # value alone; auxiliary components (SWA, Mamba) may legitimately hold
        # only a tombstone for this span (e.g. the whole leaf is outside the SWA
        # window). Materialize it anyway so the Full KV stays cacheable.
        if len(key):
            target_node = self._add_new_node(node, key, value, priority=priority)
            is_new_leaf = True
        else:
            target_node = node

        # Finalize: let each component attach its data to the target node.
        # e.g. Mamba attaches mamba_value to the leaf node
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        if target_node is not self.root_node:
            for component in self._components_tuple:
                if component.component_type == BASE_COMPONENT_TYPE:
                    continue
                component.refresh_lru(
                    LRURefreshPhase.INSERT_END, target_node, self.root_node
                )

        if is_new_leaf:
            self._inc_hit_count(target_node, params.chunked)
        return result

    def _insert_helper_host(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        host_value: torch.Tensor,
        hash_value: list[str],
    ) -> InsertResult:
        total_len = len(key)
        self._touch_node(node)
        if total_len == 0:
            return InsertResult(prefix_len=0, mamba_exist=True)

        child_key = key.child_key(self.page_size)
        matched_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            self._touch_node(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            if len(key):
                child_key = key.child_key(self.page_size)

        result = InsertResult(prefix_len=matched_length, total_len=total_len)
        if len(key) == 0:
            if (
                node is not self.root_node
                and node.component_data[BASE_COMPONENT_TYPE].host_value is not None
            ):
                result.inserted_host_node = node
            return result

        new_node = UnifiedTreeNode(self.tree_components, priority=node.priority)
        new_node.parent = node
        new_node.key = key
        new_node.hash_value = hash_value
        new_node.component_data[BASE_COMPONENT_TYPE].host_value = host_value.clone()
        node.children[child_key] = new_node
        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(node)
        result.inserted_host_node = new_node
        self.storage_radix_tree_generation += 1
        return result

    # ---- Evict Helpers ----

    def _cascade_evict(
        self,
        node: UnifiedTreeNode,
        trigger: TreeComponent,
        tracker: dict[ComponentType, int],
        target: EvictLayer = EvictLayer.DEVICE,
    ):
        """Cascade eviction from trigger to lower-or-equal priority components."""

        is_leaf = False
        if target == EvictLayer.DEVICE:
            is_leaf = node in self.evictable_device_leaves
        elif target == EvictLayer.HOST:
            is_leaf = node in self.evictable_host_leaves

        trigger_priority = trigger.eviction_priority(is_leaf)

        for comp in self._components_tuple:
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
                    if EvictLayer.DEVICE in target:
                        assert cd.lock_ref == 0
                    if EvictLayer.HOST in target:
                        assert cd.host_lock_ref == 0
                    self._evict_component_and_detach_lru(
                        node, comp, target=target, tracker=tracker
                    )

        # Now that all components (including SWA which depends on Full.value)
        # have been freed, we can safely tombstone Full.value.
        # This is deferred from evict_component because free_swa needs it.
        if (
            target is EvictLayer.DEVICE
            and trigger.component_type == BASE_COMPONENT_TYPE
        ):
            node.component_data[trigger.component_type].value = None

        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node
        self.storage_radix_tree_generation += 1

    def _evict_component_and_detach_lru(
        self,
        node: UnifiedTreeNode,
        comp: TreeComponent,
        target: EvictLayer = EvictLayer.DEVICE,
        tracker: Optional[dict[ComponentType, int]] = None,
    ) -> tuple[int, int]:
        device_freed, host_freed = comp.evict_component(node, target=target)
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
        self, deleted_node: UnifiedTreeNode, tracker: dict[ComponentType, int]
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
            for comp in self.components.values():
                if comp.node_has_component_data(cur):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.DEVICE, tracker=tracker
                    )

            if has_host:
                self._update_evictable_leaf_sets(cur)
                break

            # Full absent on both layers — evict remaining host data, delete.
            for comp in self.components.values():
                if comp.node_has_component_data(cur, target=EvictLayer.HOST):
                    self._evict_component_and_detach_lru(
                        cur, comp, target=EvictLayer.HOST, tracker=tracker
                    )

            self.evictable_host_leaves.discard(cur)
            self.evictable_host_scan_queue.pop(cur, None)
            self._remove_leaf_from_parent(cur)
            parent = cur.parent
            self._update_evictable_leaf_sets(parent)
            cur = parent

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
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue  # Full uses leaf sets, not LRU
            cd = node.component_data[ct]
            if (cd.host_value if target is EvictLayer.HOST else cd.value) is not None:
                lru = lru_dict[ct]
                if skip_existing and lru.in_list(node):
                    continue
                lru_op(lru, node)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        comp = self.components.get(component_type)
        if comp is not None:
            comp.drive_host_eviction(num_tokens, tracker)
        return tracker[component_type]

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

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        """Update both device and host leaf sets for a node."""
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

        if self._is_host_leaf(node):
            self.evictable_host_leaves.add(node)
            self.evictable_host_scan_queue.setdefault(node, None)
        else:
            self.evictable_host_leaves.discard(node)
            self.evictable_host_scan_queue.pop(node, None)

    def _evict_to_host(
        self, node: UnifiedTreeNode, tracker: Optional[dict[ComponentType, int]] = None
    ) -> None:
        """GPU→CPU demotion: release all device resources, node stays in tree."""
        assert not node.evicted and node.backuped
        trigger = self.components[BASE_COMPONENT_TYPE]
        self._evict_component_and_detach_lru(
            node, trigger, target=EvictLayer.DEVICE, tracker=tracker
        )
        self._cascade_evict(node, trigger, tracker)
        self._record_remove_event(node, medium=StorageMedium.GPU)

        # after device eviction, insert aux components into host LRU.
        self._for_each_component_lru(
            node, UnifiedLRUList.insert_mru, target=EvictLayer.HOST, skip_existing=True
        )
        self._update_evictable_leaf_sets(node.parent)
        self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    def _evict_device_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict a device leaf node, choosing the right strategy:

        - backuped: demote to host via _evict_to_host (node stays in tree)
        - not backuped + write_back: write_backup first, then demote
        - not backuped + write_through: Cascade evict all components

        All freed device tokens are accumulated into *tracker*.
        """
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"
        if not node.backuped:
            if (
                self.cache_controller is not None
                and self.cache_controller.write_policy == "write_back"
            ):
                written = self.write_backup(node, write_back=True)
                if written == 0:
                    return
                self.writing_check(write_back=True)
                self._evict_to_host(node, tracker)
                return
            else:
                # Write-through: node has no backup, delete entirely.
                self._record_remove_event(node, medium=StorageMedium.GPU)
                for comp in self._components_tuple:
                    self._evict_component_and_detach_lru(
                        node, comp, target=EvictLayer.ALL, tracker=tracker
                    )
                self.evictable_device_leaves.discard(node)
                parent = node.parent
                self._remove_leaf_from_parent(node)
                self._update_evictable_leaf_sets(parent)
                self._iteratively_delete_tombstone_leaf(node, tracker)
                return
        self._evict_to_host(node, tracker)

    def _evict_host_leaf(
        self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]
    ) -> None:
        """Atomically evict all components on a host leaf.

        All freed tokens are accumulated into *tracker*."""
        assert self._is_host_leaf(node), f"node {node.id} is not an H-leaf"

        self._record_remove_event(node, medium=StorageMedium.CPU)
        for comp in self._components_tuple:
            _, hf = self._evict_component_and_detach_lru(
                node, comp, target=EvictLayer.ALL, tracker=None
            )
            tracker[comp.component_type] += hf
        self.evictable_host_leaves.discard(node)
        self.evictable_host_scan_queue.pop(node, None)
        self._remove_leaf_from_parent(node)
        self._iteratively_delete_tombstone_leaf(node, tracker)
        self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    # ---- HiCache: Backup / LoadBack ----

    def write_backup(
        self,
        node: UnifiedTreeNode,
        write_back: bool = False,
        skip_components: frozenset[ComponentType] = frozenset(),
    ) -> int:
        """Backup a node's data from device to host (D->H)."""
        if self.cache_controller is None:
            return 0

        # Backup invariant (write-through): parent must be backuped first
        if not write_back and (
            node.parent is not self.root_node and not node.parent.backuped
        ):
            if (
                self.write_backup(
                    node.parent,
                    skip_components=skip_components,
                )
                <= 0
            ):
                return 0

        device_value = node.component_data[BASE_COMPONENT_TYPE].value
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            if comp.component_type in skip_components:
                continue
            t = comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

        # Pre-evict host if insufficient
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            evicted = self.evict_host(needed)
            if evicted < needed:
                return 0

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        host_indices = self.cache_controller.write(
            device_value, node_id=node.id, extra_pools=aux_xfers or None
        )
        if host_indices is None:
            return 0

        # Commit
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[kv_xfer],
        )
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                node,
                CacheTransferPhase.BACKUP_HOST,
                transfers=xfers,
            )

        lock_params = None
        if not write_back:
            lock_params = self.inc_lock_ref(node).to_dec_params()
        self._track_write_through_node(node, lock_params)
        return len(host_indices)

    def _track_write_through_node(
        self,
        node: UnifiedTreeNode,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = _OngoingWriteThrough(
            node, lock_params, [node]
        )

    def _replace_pending_write_through_node(
        self, old_node: UnifiedTreeNode, new_nodes: list[UnifiedTreeNode]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, lock_params, publish_nodes = pending
        updated_nodes = []
        replaced = False
        for node in publish_nodes:
            if node is old_node:
                updated_nodes.extend(new_nodes)
                replaced = True
            else:
                updated_nodes.append(node)

        if not replaced:
            return

        for node in new_nodes:
            node.write_through_pending_id = ack_id
        self.ongoing_write_through[ack_id] = _OngoingWriteThrough(
            lock_node,
            lock_params,
            updated_nodes,
        )

    def _finish_write_through_ack(self, ack_id: int) -> None:
        lock_node, lock_params, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            self._record_store_event(node, medium=StorageMedium.CPU)
        if lock_params is not None:
            self.dec_lock_ref(lock_node, lock_params)
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node in publish_nodes:
                self.write_backup_storage(node)

    def load_back(
        self,
        best_match_node: UnifiedTreeNode,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return False

        host_anchor_params = self.inc_host_lock_ref(best_match_node).to_dec_params()
        # Build KV transfer
        kv_xfer = self.components[BASE_COMPONENT_TYPE].build_hicache_transfers(
            best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]

        # Lock path & pre-evict if device pool is insufficient
        result = self.inc_lock_ref(best_match_node)
        ancestor_lock_params = result.to_dec_params()
        kv_tokens = len(kv_xfer.host_indices)

        # Build aux transfers, keyed per component.
        comp_xfers: dict[ComponentType, list] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            t = comp.build_hicache_transfers(
                best_match_node, CacheTransferPhase.LOAD_BACK, req=req
            )
            if t:
                comp_xfers[comp.component_type] = t
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding.
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(best_match_node, ancestor_lock_params)
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        if self.supports_swa():
            avail = self.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = self.token_to_kv_pool_allocator.available_size()
        if avail < kv_tokens:
            needed = kv_tokens - avail
            result = self.evict(EvictParams(num_tokens=needed))
            if result.num_tokens_evicted < needed:
                self.dec_lock_ref(best_match_node, ancestor_lock_params)
                self.dec_host_lock_ref(best_match_node, host_anchor_params)
                return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=best_match_node.id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(best_match_node, ancestor_lock_params)
        if device_indices is None:
            self.dec_host_lock_ref(best_match_node, host_anchor_params)
            return False

        # Commit: each component gets only its own transfers
        kv_xfer.device_indices = device_indices
        self.components[BASE_COMPONENT_TYPE].commit_hicache_transfer(
            best_match_node,
            CacheTransferPhase.LOAD_BACK,
            [kv_xfer],
        )
        for node in kv_xfer.nodes_to_load or ():
            self._record_store_event(node, medium=StorageMedium.GPU)
        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                best_match_node,
                CacheTransferPhase.LOAD_BACK,
                xfers,
            )

        self._update_evictable_leaf_sets(best_match_node)
        self.ongoing_load_back[best_match_node.id] = _OngoingLoadBack(
            best_match_node,
            self.inc_lock_ref(best_match_node).to_dec_params(),
            host_anchor_params,
        )

        return True

    def _build_sidecar_transfers(
        self,
        phase: CacheTransferPhase,
        kv_xfer: PoolTransfer,
        comp_xfers: dict[ComponentType, list[PoolTransfer]],
    ) -> list[PoolTransfer]:
        transfers: list[PoolTransfer] = []
        for spec in self.sidecar_pool_specs:
            if spec.indices_from_pool == PoolName.KV:
                indices_source = kv_xfer
            else:
                source_component = {
                    PoolName.SWA: ComponentType.SWA,
                    PoolName.MAMBA: ComponentType.MAMBA,
                }.get(spec.indices_from_pool)
                if source_component is None:
                    raise AssertionError(
                        f"Unsupported sidecar indices source pool "
                        f"{spec.indices_from_pool}."
                    )
                matching_sources = comp_xfers.get(source_component, ())
                if not matching_sources:
                    continue
                indices_source = matching_sources[0]
                if indices_source.name != spec.indices_from_pool:
                    raise AssertionError(
                        f"Sidecar indices source pool {spec.indices_from_pool} "
                        f"resolved to {indices_source.name} during {phase}."
                    )

            indices = (
                indices_source.device_indices
                if phase == CacheTransferPhase.BACKUP_HOST
                else indices_source.host_indices
            )
            defer_kv_sidecar = (
                phase == CacheTransferPhase.PREFETCH
                and spec.indices_from_pool == PoolName.KV
            )
            if (indices is None or len(indices) == 0) and not defer_kv_sidecar:
                continue
            transfers.append(
                PoolTransfer(
                    name=spec.pool_name,
                    keys=indices_source.keys,
                    hit_policy=spec.hit_policy,
                    indices_from_pool=spec.indices_from_pool,
                )
            )
        return transfers

    def _inc_hit_count(self, node: UnifiedTreeNode, chunked: bool = False) -> None:
        """Increment hit count; trigger write_backup when threshold reached."""
        if node.evicted or chunked:
            return
        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            return
        node.hit_count += 1
        if (
            self.cache_controller is not None
            and not node.backuped
            and node.hit_count >= self.write_through_threshold
        ):
            self.write_backup(node)

    def write_backup_storage(self, node: UnifiedTreeNode) -> Optional[int]:
        if (
            not self.enable_storage
            or self.cache_controller is None
            or not node.backuped
        ):
            return None

        prefix_keys = None
        if self.hicache_storage_pass_prefix_keys:
            prefix_keys = node.get_prefix_hash_values(node.parent)

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            transfers = comp.build_hicache_transfers(
                node,
                CacheTransferPhase.BACKUP_STORAGE,
            )
            if transfers:
                comp_xfers[comp.component_type] = transfers

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=node.component_data[BASE_COMPONENT_TYPE].host_value,
            keys=node.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, comp_xfers
        )
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.cache_controller.write_storage(
            node.component_data[BASE_COMPONENT_TYPE].host_value,
            node.key.token_ids,
            node.hash_value,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_backup[operation_id] = (
            node,
            self.inc_host_lock_ref(node).to_dec_params(),
        )
        return operation_id

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: UnifiedTreeNode,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        checkpoint_dependency: Optional[str] = None,
    ) -> StoragePrefetchState:
        if req_id in self.ongoing_prefetch or req_id in self.storage_prefetch_ownership:
            raise RuntimeError(f"HiCache prefetch {req_id} already owns host memory")
        if not self.enable_storage or self.cache_controller is None:
            state = StoragePrefetchState.MISS
            self.storage_prefetch_tracker.set(req_id, state)
            return state

        extra_key = last_host_node.key.extra_key if last_host_node.key else None
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        if checkpoint_dependency is not None:
            checkpoint_state = self.get_storage_checkpoint_dependency_state(
                req_id, checkpoint_dependency
            )
            if checkpoint_state is StorageCheckpointState.FAILED:
                state = StoragePrefetchState.FAILED
                self.storage_prefetch_tracker.set(req_id, state)
                return state
            if checkpoint_state is not StorageCheckpointState.READY:
                state = StoragePrefetchState.DEFERRED
                self.storage_prefetch_tracker.set(req_id, state)
                return state

        prefetch_length = len(prefetch_key)
        if prefetch_length < self.prefetch_threshold:
            state = StoragePrefetchState.MISS
            self.storage_prefetch_tracker.set(req_id, state)
            return state
        pending_write = torch.tensor(
            int(
                self.cache_controller.has_pending_storage_write(prefetch_key, last_hash)
            ),
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(pending_write, torch.distributed.ReduceOp.MAX)
        if pending_write.item() != 0:
            state = StoragePrefetchState.DEFERRED
            self.storage_prefetch_tracker.set(req_id, state)
            return state
        if self.synchronize_storage_prefetch_flag(
            self.cache_controller.prefetch_rate_limited()
        ):
            state = StoragePrefetchState.DEFERRED
            self.storage_prefetch_tracker.set(req_id, state)
            return state

        anchor_lock_params = self.inc_host_lock_ref(last_host_node).to_dec_params()
        quotas_enabled = bool(self.cache_controller.l3_staging.quotas)
        allocation_length = prefetch_length
        if quotas_enabled:
            capacity_state = self._prefetch_staging_capacity_state(allocation_length)
            if capacity_state is not None and checkpoint_dependency is None:
                with l3_staging_allocation():
                    available_size = (
                        self.cache_controller.mem_pool_host.available_size()
                    )
                available_size = self.synchronize_storage_prefetch_min(available_size)
                allocation_length = min(
                    prefetch_length,
                    available_size - available_size % self.page_size,
                )
                if allocation_length >= self.prefetch_threshold:
                    capacity_state = self._prefetch_staging_capacity_state(
                        allocation_length
                    )
            if (
                capacity_state is not None
                or allocation_length < self.prefetch_threshold
            ):
                self.dec_host_lock_ref(last_host_node, anchor_lock_params)
                state = (
                    StoragePrefetchState.FAILED
                    if capacity_state == _CHECKPOINT_STAGING_CAPACITY_EXCEEDED
                    and checkpoint_dependency is not None
                    else StoragePrefetchState.DEFERRED
                )
                self.storage_prefetch_tracker.set(req_id, state)
                return state

        with l3_staging_allocation():
            host_indices = self.cache_controller.mem_pool_host.alloc(allocation_length)
            if host_indices is None and not quotas_enabled:
                self.evict_host(allocation_length)
                host_indices = self.cache_controller.mem_pool_host.alloc(
                    allocation_length
                )
        if quotas_enabled and host_indices is None:
            raise RuntimeError("HiCache L3 staging capacity changed after reservation")
        synchronized_length = self.synchronize_storage_prefetch_min(
            len(host_indices) if host_indices is not None else 0
        )
        if synchronized_length < self.prefetch_threshold:
            if host_indices is not None:
                self.cache_controller.mem_pool_host.free(host_indices)
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            state = StoragePrefetchState.DEFERRED
            self.storage_prefetch_tracker.set(req_id, state)
            return state
        assert host_indices is not None
        if len(host_indices) > synchronized_length:
            self.cache_controller.mem_pool_host.free(host_indices[synchronized_length:])
            host_indices = host_indices[:synchronized_length]
        prefetch_key = prefetch_key[:synchronized_length]

        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for comp in self._components_tuple:
            if comp.component_type == BASE_COMPONENT_TYPE:
                continue
            with l3_staging_allocation():
                transfers = comp.build_hicache_transfers(
                    last_host_node,
                    CacheTransferPhase.PREFETCH,
                    token_ids=prefetch_key.token_ids,
                    prefetch_tokens=len(prefetch_key),
                    last_hash=last_hash,
                )
            if transfers == []:
                alloc_failed = True
                break
            if transfers:
                comp_xfers[comp.component_type] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        alloc_failed = self.synchronize_storage_prefetch_flag(alloc_failed)
        if alloc_failed:
            self.cache_controller.mem_pool_host.free(host_indices)
            self._release_prefetch_aux_allocations(comp_xfers)
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            state = StoragePrefetchState.DEFERRED
            self.storage_prefetch_tracker.set(req_id, state)
            return state

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
            host_indices=host_indices,
        )
        self.ongoing_prefetch[req_id] = _OngoingPrefetch(
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.storage_prefetch_ownership[req_id] = StoragePrefetchOwnership(
            operation_id=operation.id,
            generation=operation.id,
            host_indices=host_indices,
            owned_tokens=len(host_indices),
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)
        state = StoragePrefetchState.QUERYING
        self.storage_prefetch_tracker.set(req_id, state)
        return state

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def storage_prefetch_timeout(self, num_tokens: int) -> float:
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        return self.prefetch_timeout_base + num_pages * self.prefetch_timeout_per_page

    def can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self._prefetch_timeout_check_linear_func(
                operation
            )
        else:
            return True
        if (
            completed
            and getattr(operation, "pool_transfers", None)
            and not getattr(operation, "pool_transfers_done", True)
        ):
            can_terminate = False

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        return can_terminate or operation_terminated

    @staticmethod
    def _prefetch_transition_fingerprint(
        transition: PrefetchOwnershipTransition,
    ) -> dict[str, int]:
        request_digest = hashlib.sha256(transition.request_id.encode()).digest()
        hash_digest = hashlib.sha256()
        for value in transition.hash_values:
            encoded = value.encode()
            hash_digest.update(len(encoded).to_bytes(8, "big"))
            hash_digest.update(encoded)
        return {
            "request_id_0": int.from_bytes(request_digest[:4], "big") & 0x7FFFFFFF,
            "request_id_1": int.from_bytes(request_digest[4:8], "big") & 0x7FFFFFFF,
            "generation": transition.generation,
            "state": transition.state.value,
            "retained_tokens": transition.retained_tokens,
            "hash_count": len(transition.hash_values),
            **{
                f"hash_digest_{index}": int.from_bytes(
                    hash_digest.digest()[offset : offset + 4], "big"
                )
                & 0x7FFFFFFF
                for index, offset in enumerate(range(0, hash_digest.digest_size, 4))
            },
        }

    def _truncate_prefetch_ownership(
        self, request_id: str, retained_tokens: int
    ) -> StoragePrefetchOwnership:
        ownership = self.storage_prefetch_ownership.get(request_id)
        if ownership is None:
            raise RuntimeError(f"HiCache prefetch {request_id} has no ownership")
        if retained_tokens % self.page_size != 0:
            raise RuntimeError(
                f"HiCache prefetch {request_id} retained unaligned memory"
            )
        if not 0 <= retained_tokens <= ownership.owned_tokens:
            raise RuntimeError(
                f"HiCache prefetch {request_id} ownership cannot move from "
                f"{ownership.owned_tokens} to {retained_tokens}"
            )
        if retained_tokens < ownership.owned_tokens:
            self.cache_controller.mem_pool_host.free(
                ownership.host_indices[retained_tokens : ownership.owned_tokens]
            )
            ownership.owned_tokens = retained_tokens
            self.storage_state_change_resources.add(CheckpointRetryResource.HOST)
        return ownership

    def _release_prefetch_aux_allocations(
        self, comp_xfers: dict[ComponentType, list[PoolTransfer]]
    ) -> None:
        self._release_prefetch_pool_allocations(
            [transfer for transfers in comp_xfers.values() for transfer in transfers]
        )

    def _release_prefetch_pool_allocations(
        self, transfers: Sequence[PoolTransfer]
    ) -> None:
        controller = self.cache_controller
        mem_pool_host = controller.mem_pool_host
        anchor_entry = getattr(mem_pool_host, "anchor_entry", None)
        released = False
        for transfer in transfers:
            host_indices = transfer.host_indices
            entry = mem_pool_host.entry_map.get(transfer.name)
            if (
                host_indices is None
                or len(host_indices) == 0
                or entry is None
                or entry is anchor_entry
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            entry.host_pool.free(host_indices)
            released = True
        if released:
            self.storage_state_change_resources.add(CheckpointRetryResource.AUXILIARY)

    def _apply_prefetch_ownership_transition(
        self, transition: PrefetchOwnershipTransition
    ) -> None:
        self._require_storage_prefetch_rank_agreement(
            "ownership transition",
            self._prefetch_transition_fingerprint(transition),
        )
        info = self.ongoing_prefetch.get(transition.request_id)
        ownership = self.storage_prefetch_ownership.get(transition.request_id)
        if info is None or ownership is None:
            return
        operation = info.operation
        if (
            ownership.operation_id != operation.id
            or ownership.generation != transition.generation
        ):
            return
        if transition.state is not StoragePrefetchState.READING:
            self._revoke_pending_prefetch(transition.request_id, transition.state)
            return
        if transition.request_id in self.storage_prefetch_aborted_requests:
            self._revoke_pending_prefetch(
                transition.request_id, StoragePrefetchState.FAILED
            )
            return
        if (
            transition.retained_tokens < self.prefetch_threshold
            or len(transition.hash_values) * self.page_size
            != transition.retained_tokens
        ):
            raise RuntimeError(
                f"HiCache prefetch {transition.request_id} published an "
                "inconsistent ownership transition"
            )
        ownership = self._truncate_prefetch_ownership(
            transition.request_id, transition.retained_tokens
        )
        operation.hash_value = list(transition.hash_values)
        operation.host_indices = ownership.host_indices[: transition.retained_tokens]
        operation.storage_prefetch_state = StoragePrefetchState.READING
        operation.worker_done.clear()
        self.storage_prefetch_tracker.set(
            transition.request_id, StoragePrefetchState.READING
        )
        self.cache_controller.prefetch_buffer.put(operation)

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return self.storage_prefetch_tracker.get(req_id).is_terminal

        (
            last_host_node,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        state = operation.storage_prefetch_state
        query_in_progress = self.synchronize_storage_prefetch_flag(
            state is StoragePrefetchState.QUERYING
        )
        if query_in_progress:
            self.storage_prefetch_tracker.set(req_id, state)
            return False
        if state is StoragePrefetchState.READING:
            read_in_progress = self.synchronize_storage_prefetch_flag(
                not operation.worker_done.is_set()
            )
            if read_in_progress:
                self.storage_prefetch_tracker.set(req_id, state)
                return False
            self._truncate_prefetch_ownership(req_id, int(operation.completed_tokens))
        self.storage_prefetch_tracker.set(req_id, state)
        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        ownership = self.storage_prefetch_ownership.get(req_id)
        if ownership is None or ownership.owned_tokens != completed_tokens:
            raise RuntimeError(
                f"HiCache prefetch {req_id} completion disagrees with "
                "scheduler ownership"
            )

        min_completed_tokens = self._sync_and_check_hybrid_prefetch_result(
            req_id,
            operation,
            completed_tokens,
            hash_value,
            last_host_node,
            anchor_lock_params,
            prefetch_key,
        )
        if min_completed_tokens is None:
            # Hybrid all-or-nothing check failed; result already discarded.
            return True

        fetched_key = prefetch_key[:min_completed_tokens]
        insert_result = self._insert_helper_host(
            last_host_node,
            fetched_key,
            host_indices[:min_completed_tokens],
            hash_value[: min_completed_tokens // self.page_size],
        )
        self._require_storage_prefetch_rank_agreement(
            "host insertion",
            {
                "matched_tokens": insert_result.prefix_len,
                "has_final_host_node": int(
                    insert_result.inserted_host_node is not None
                ),
            },
        )

        for ct, xfers in comp_xfers.items():
            self.components[ct].commit_hicache_transfer(
                last_host_node,
                CacheTransferPhase.PREFETCH,
                xfers,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )

        loaded_from_storage = min_completed_tokens - insert_result.prefix_len
        extra_transfers = [
            transfer for transfers in comp_xfers.values() for transfer in transfers
        ]
        staging_leases = self._l3_staging_leases(
            host_indices[insert_result.prefix_len : min_completed_tokens],
            extra_transfers,
            owner_node=insert_result.inserted_host_node,
        )
        admission_occupied_tokens = 0
        hold_staging = False
        if self.storage_checkpoint_dependency_results.get(req_id) == "ready":
            final_host_node = insert_result.inserted_host_node
            if (
                final_host_node is None
                and last_host_node is not self.root_node
                and last_host_node.backuped
            ):
                final_host_node = last_host_node
            if final_host_node is not None:
                admission_occupied_tokens = loaded_from_storage
                self.pin_storage_prefetch_admission(
                    req_id,
                    final_host_node,
                    admission_occupied_tokens,
                    staging_leases,
                )
                hold_staging = True
        if not hold_staging:
            self._enqueue_staging_leases(staging_leases)

        if insert_result.prefix_len > 0:
            self.cache_controller.mem_pool_host.free(
                host_indices[: insert_result.prefix_len]
            )
        if min_completed_tokens < completed_tokens:
            self.cache_controller.mem_pool_host.free(
                host_indices[min_completed_tokens:completed_tokens]
            )
        self.storage_prefetch_ownership.pop(req_id)
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)
        released_tokens = len(prefetch_key) - admission_occupied_tokens
        if released_tokens > self.cache_controller.prefetch_tokens_occupied:
            raise RuntimeError(
                f"HiCache prefetch {req_id} occupancy accounting underflow"
            )
        self.cache_controller.prefetch_tokens_occupied -= released_tokens

        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        self.storage_prefetch_tracker.set(
            req_id,
            (
                StoragePrefetchState.READY
                if loaded_from_storage > 0
                else StoragePrefetchState.MISS
            ),
        )
        logger.info(
            "HiCache durable prefetch req=%s state=%s completed_local=%d "
            "completed_synced=%d matched=%d loaded=%d occupied=%d staging=%s",
            req_id,
            self.storage_prefetch_tracker.get(req_id).name,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            self.cache_controller.prefetch_tokens_occupied,
            self.cache_controller.l3_staging.usage(),
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return True

    def _sync_and_check_hybrid_prefetch_result(
        self,
        req_id: str,
        operation: PrefetchOperation,
        completed_tokens: int,
        hash_value: list[str],
        last_host_node: UnifiedTreeNode,
        anchor_lock_params: DecLockRefParams,
        prefetch_key: RadixKey,
    ) -> Optional[int]:
        """Sync prefetch results across ATTN groups and decide the usable prefix.

        Two strategies depending on the hybrid layout:

        * DSA-style (Full attention + KV-derived ALL_PAGES sidecar such as the
          DSA / MiniMax indexer): *clamp* to the minimum fetched prefix shared by
          the Full KV pool and every sidecar. A partial prefix is still usable
          because the sidecar is page-aligned with KV and required for every page.
        * Everything else (SWA / Mamba components, mixed DeepSeekV4 stacks):
          *all-or-nothing*. Their pools only cover a window / tail and cannot be
          truncated page by page, so any shortfall discards the whole prefetch.

        Returns the synced usable token count (possibly clamped, possibly 0), or
        ``None`` when an all-or-nothing prefetch was discarded (the caller should
        then treat the prefetch as finished).
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        pool_transfers = operation.pool_transfers or []
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        packed = torch.tensor([completed_tokens, *pool_hit_pages], dtype=torch.int)
        self._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
        min_completed_tokens = int(packed[0].item())
        pool_hit_pages = list(map(int, packed[1:].tolist()))
        for transfer, count in zip(pool_transfers, pool_hit_pages):
            hit_pages[transfer.name] = count

        # DSA-style clamp: every sidecar is KV-derived and required for the whole
        # prefix (ALL_PAGES), so the usable length is simply the shared minimum of
        # the Full KV completion and each sidecar hit.
        clampable = bool(pool_transfers) and all(
            t.hit_policy == PoolHitPolicy.ALL_PAGES
            and t.indices_from_pool == PoolName.KV
            for t in pool_transfers
        )
        if clampable:
            usable_pages = min(min_completed_tokens // self.page_size, *pool_hit_pages)
            return usable_pages * self.page_size

        # Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
        # must cover the same fetched prefix. If any pool falls short the whole
        # prefetch result is unusable, so discard it and release everything.
        expected_tokens = len(hash_value) * self.page_size
        all_succeeded = min_completed_tokens == expected_tokens and all(
            transfer.keys is not None and count == len(transfer.keys)
            for transfer, count in zip(pool_transfers, pool_hit_pages)
        )
        if pool_transfers and not all_succeeded:
            self._truncate_prefetch_ownership(req_id, 0)
            self.storage_prefetch_ownership.pop(req_id)
            self._release_prefetch_pool_allocations(pool_transfers)
            self.dec_host_lock_ref(last_host_node, anchor_lock_params)
            del self.ongoing_prefetch[req_id]
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)
            if len(prefetch_key) > self.cache_controller.prefetch_tokens_occupied:
                raise RuntimeError(
                    f"HiCache prefetch {req_id} occupancy accounting underflow"
                )
            self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
            self.prefetch_loaded_tokens_by_reqid[req_id] = 0
            self.storage_prefetch_tracker.set(req_id, StoragePrefetchState.MISS)
            logger.warning(
                "HiCache hybrid prefetch discarded req=%s completed=%d requested=%d",
                req_id,
                completed_tokens,
                expected_tokens,
            )
            return None
        return min_completed_tokens

    def terminate_prefetch(self, req_id: str) -> None:
        if req_id not in self.ongoing_prefetch:
            return
        operation = self.ongoing_prefetch[req_id].operation
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        self.storage_prefetch_tracker.forget(req_id)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_request(self, rid: str) -> None:
        checkpoint_handle = make_storage_checkpoint_handle(rid)
        self.storage_checkpoint_registry.fail_if_present(checkpoint_handle)
        self._release_storage_checkpoint(checkpoint_handle, release_staging=True)
        self.release_storage_prefetch_admission(rid)
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self.storage_checkpoint_dependency_results.pop(rid, None)
        if rid not in self.ongoing_prefetch:
            self.storage_prefetch_tracker.forget(rid)
            return
        self.ongoing_prefetch[rid].operation.mark_terminate()
        self.storage_prefetch_aborted_requests.add(rid)

    def _revoke_pending_prefetch(
        self,
        req_id: str,
        state: StoragePrefetchState = StoragePrefetchState.MISS,
    ) -> None:
        was_aborted = req_id in self.storage_prefetch_aborted_requests
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return
        (
            last_host_node,
            prefetch_key,
            _host_indices,
            _operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        cc = self.cache_controller
        self._truncate_prefetch_ownership(req_id, 0)
        self.storage_prefetch_ownership.pop(req_id)
        self._release_prefetch_aux_allocations(comp_xfers)
        self.dec_host_lock_ref(last_host_node, anchor_lock_params)
        self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)
        cc.prefetch_tokens_occupied = max(
            0, cc.prefetch_tokens_occupied - len(prefetch_key)
        )
        self.storage_prefetch_aborted_requests.discard(req_id)
        if was_aborted:
            self.storage_prefetch_tracker.forget(req_id)
        else:
            self.storage_prefetch_tracker.set(req_id, state)

    def _cleanup_aborted_prefetches(self) -> None:
        aborted = sorted(self.storage_prefetch_aborted_requests)
        for index in range(_ABORTED_PREFETCH_CLEANUP_QUANTUM):
            request_id = aborted[index] if index < len(aborted) else None
            digest = hashlib.sha256((request_id or "").encode()).digest()
            self._require_storage_prefetch_rank_agreement(
                "aborted prefetch cleanup",
                {
                    "present": int(request_id is not None),
                    "request_id_0": int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
                    "request_id_1": int.from_bytes(digest[4:8], "big") & 0x7FFFFFFF,
                },
            )
            if request_id is None:
                return
            info = self.ongoing_prefetch.get(request_id)
            if info is None:
                self.storage_prefetch_aborted_requests.discard(request_id)
                continue
            operation = info.operation
            if operation.storage_prefetch_state is StoragePrefetchState.QUERYING:
                continue
            if (
                operation.storage_prefetch_state is StoragePrefetchState.READING
                and self.synchronize_storage_prefetch_flag(
                    not operation.worker_done.is_set()
                )
            ):
                continue
            self._revoke_pending_prefetch(request_id, StoragePrefetchState.FAILED)

    @staticmethod
    def _backup_sequence_fingerprint(operations: Sequence[Any]) -> dict[str, int]:
        digest = hashlib.sha256()
        for operation in operations:
            digest.update(int(operation.id).to_bytes(8, "big", signed=True))
            for value in tuple(operation.hash_value or ()):
                encoded = value.encode()
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            for transfer in getattr(operation, "pool_transfers", None) or ():
                encoded_name = str(transfer.name).encode()
                digest.update(len(encoded_name).to_bytes(8, "big"))
                digest.update(encoded_name)
                for value in tuple(transfer.keys or ()):
                    encoded = value.encode()
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
        return {
            "count": len(operations),
            **{
                f"digest_{index}": int.from_bytes(
                    digest.digest()[offset : offset + 4], "big"
                )
                & 0x7FFFFFFF
                for index, offset in enumerate(range(0, digest.digest_size, 4))
            },
        }

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_storage_hit: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], limit: Optional[int]) -> Iterator[T]:
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        changed_resources = set(self.storage_state_change_resources)
        self.storage_state_change_resources.clear()

        transitions = list(_drain_queue(cc.prefetch_revoke_queue, n_revoke))
        for transition in transitions:
            if not isinstance(transition, PrefetchOwnershipTransition):
                raise RuntimeError(
                    "Unified HiCache received a legacy prefetch transition"
                )
            self._apply_prefetch_ownership_transition(transition)
        if transitions:
            changed_resources.update(
                {
                    CheckpointRetryResource.HOST,
                    CheckpointRetryResource.AUXILIARY,
                }
            )

        if self.synchronize_storage_prefetch_flag(
            bool(self.storage_prefetch_aborted_requests)
        ):
            self._cleanup_aborted_prefetches()

        operations = list(_drain_queue(cc.ack_backup_queue, n_backup))
        if operations:
            self._require_storage_prefetch_rank_agreement(
                "storage backup completion sequence",
                self._backup_sequence_fingerprint(operations),
            )
            changed_resources.update(CheckpointRetryResource)
            durable_pages = torch.tensor(
                [
                    operation.durable_page_count(
                        self.page_size, backup_skip=cc.backup_skip
                    )
                    for operation in operations
                ],
                dtype=torch.int,
            )
            self._all_reduce_attn_groups(durable_pages, torch.distributed.ReduceOp.MIN)
            for operation, durable_page_count in zip(
                operations, durable_pages.tolist(), strict=True
            ):
                completion = cc.storage_write_tracker.complete(
                    operation.id, int(durable_page_count)
                )
                if completion is not None:
                    self.storage_checkpoint_registry.record_write_completion(
                        completion, cc.storage_write_tracker.has_pending
                    )
                checkpoint_owner = self.storage_checkpoint_owners_by_operation.pop(
                    operation.id, None
                )
                if checkpoint_owner is not None:
                    checkpoint = self.storage_pending_checkpoints.get(
                        checkpoint_owner[0]
                    )
                    if (
                        checkpoint is not None
                        and checkpoint.generation == checkpoint_owner[1]
                    ):
                        checkpoint.active_operation_ids.discard(operation.id)
                entry = self.ongoing_backup.pop(operation.id, None)
                owner_node = entry[0] if entry is not None else None
                if entry is not None:
                    node, lock_params = entry
                    self.dec_host_lock_ref(node, lock_params)
                self._enqueue_staging_leases(
                    self._l3_staging_leases(
                        operation.host_indices,
                        getattr(operation, "pool_transfers", None),
                        owner_node=owner_node,
                    )
                )
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        host_indices_list = list(_drain_queue(cc.host_mem_release_queue, n_release))
        if host_indices_list:
            cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            changed_resources.add(CheckpointRetryResource.HOST)

        if extra_release_counts:
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                indices = list(_drain_queue(release_queue, limit))
                if not indices:
                    continue
                entry = cc.mem_pool_host.entry_map.get(pool_name)
                if entry is not None:
                    entry.host_pool.free(torch.cat(indices, dim=0))
                    changed_resources.add(CheckpointRetryResource.AUXILIARY)

        changed_resources.update(self.storage_state_change_resources)
        self.storage_state_change_resources.clear()
        staging_generation = int(
            getattr(getattr(cc, "l3_staging", None), "generation", 0)
        )
        if staging_generation != getattr(
            self, "l3_staging_generation_seen", staging_generation
        ):
            changed_resources.update(_CAPACITY_RETRY_RESOURCES)
        changed_resources = self._synchronize_storage_state_change_resources(
            changed_resources
        )
        if changed_resources.intersection(_CAPACITY_RETRY_RESOURCES):
            self.l3_staging_reclaim_eligible = len(self.l3_staging_pending_leases)
        if changed_resources.intersection(
            _CAPACITY_RETRY_RESOURCES | {CheckpointRetryResource.SCHEDULER}
        ):
            self._reclaim_l3_staging_headroom()
            changed_resources.update(self.storage_state_change_resources)
            self.storage_state_change_resources.clear()
            changed_resources = self._synchronize_storage_state_change_resources(
                changed_resources
            )
        self._drive_storage_checkpoint_retries(changed_resources)
        self.l3_staging_generation_seen = int(
            getattr(getattr(cc, "l3_staging", None), "generation", 0)
        )

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_revoke_queue.qsize(),
            cc.prefetch_hit_queue.qsize(),
            cc.ack_backup_queue.qsize(),
            cc.host_mem_release_queue.qsize(),
            *[
                extra_release_queues[pool_name].qsize()
                for pool_name in extra_pool_names
            ],
        ]
        qsizes = torch.tensor(
            local_qsize_list,
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_revoke, n_storage_hit, n_backup, n_release = qsize_list[:4]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[4:])
        }
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_storage_hit=n_storage_hit,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            attn_cp_rank, attn_cp_size = (
                self.cache_controller.get_attn_cp_rank_and_size()
            )
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": attn_cp_rank,
                "attn_cp_size": attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = self.storage_metrics_collector
            if existing_collector is None:
                from sglang.srt.runtime_context import get_server_args

                storage_cls = resolve_collector_class(
                    get_server_args(),
                    STAT_LOGGER_ROLE_STORAGE,
                    StorageMetricsCollector,
                )
                self.storage_metrics_collector = storage_cls(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )
        else:
            self.storage_metrics_collector = None

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage attach yet. "
            "Configure hicache_storage_backend at startup instead.",
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        return (
            False,
            "UnifiedRadixCache does not support runtime HiCache storage detach yet. "
            "Restart without hicache_storage_backend to disable it.",
        )

    def clear_storage_backend(self) -> bool:
        try:
            ok = self.cache_controller.clear_storage_backend()
        except Exception as e:
            logger.error("Failed to clear hierarchical cache storage backend: %s", e)
            return False
        if ok:
            for handle in tuple(self.storage_pending_checkpoints):
                self.storage_checkpoint_registry.fail(handle)
                self._release_storage_checkpoint(handle, release_staging=True)
            request_ids = set(self.storage_prefetch_admission_host_locks)
            request_ids.update(self.storage_prefetch_admission_device_locks)
            for request_id in request_ids:
                self.release_storage_prefetch_admission(request_id)
            self._reclaim_l3_staging_headroom()
            self.cache_controller.storage_write_tracker.clear()
            self.storage_checkpoint_registry.clear()
            self.storage_checkpoint_retries.clear()
            self.storage_checkpoint_owners_by_operation.clear()
            self.storage_checkpoint_dependency_results.clear()
            self.storage_state_change_resources.intersection_update(
                _CAPACITY_RETRY_RESOURCES | {CheckpointRetryResource.SCHEDULER}
            )
            self.l3_staging_generation_seen = (
                self.cache_controller.l3_staging.generation
            )
            logger.info("Hierarchical cache storage backend cleared successfully!")
        return ok

    # ---- HiCache: Async Event Management ----

    def writing_check(self, write_back: bool = False) -> None:
        """Poll write-through completions."""
        cc = self.cache_controller
        if cc is None:
            return

        if write_back:
            # Blocking: wait for all pending write-backs
            while self.ongoing_write_through:
                for ack in cc.ack_write_queue:
                    ack.finish_event.synchronize()
                    for ack_id in ack.node_ids:
                        if ack_id in self.ongoing_write_through:
                            self._finish_write_through_ack(ack_id)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # Every rank must enter the all_reduce below; ongoing_write_through can
        # diverge across ranks (e.g. write_backup returning 0 on a subset).
        finish_count = 0
        if self.pp_rank == 0:
            for ack in cc.ack_write_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1

        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()
        had_completions = finish_count > 0

        # Process completed acks
        while finish_count > 0:
            ack = cc.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id)
            finish_count -= 1
        if had_completions:
            self.storage_state_change_resources.update(
                {
                    CheckpointRetryResource.HOST,
                    CheckpointRetryResource.AUXILIARY,
                    CheckpointRetryResource.STORAGE,
                }
            )

    def loading_check(self) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        # Every rank must enter the all_reduce below; ongoing_load_back can
        # diverge across ranks.
        finish_count = 0
        if self.pp_rank == 0:
            for ack in cc.ack_load_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1
        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()
        had_completions = finish_count > 0

        while finish_count > 0:
            ack = cc.ack_load_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)

            if self.metrics_collector is not None:
                self.metrics_collector.increment_load_back_num_tokens(ack.num_tokens)
                if ack.timing_enabled:
                    duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                    self.metrics_collector.observe_load_back_duration(
                        duration_ms / 1000.0
                    )
            finish_count -= 1
        if had_completions:
            self.storage_state_change_resources.update(_CAPACITY_RETRY_RESOURCES)

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, UnifiedTreeNode]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        best_match_node = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        last_best_match_device_node = req.last_node

        def _collect_new_prefix_indices() -> torch.Tensor:
            prefix_chunks: list[torch.Tensor] = []
            node = best_match_node
            while node is not last_best_match_device_node:
                value = node.component_data[BASE_COMPONENT_TYPE].value
                assert value is not None
                prefix_chunks.append(value)
                node = node.parent
            if not prefix_chunks:
                return self._empty_match_result.device_indices
            prefix_chunks.reverse()
            return torch.cat(prefix_chunks)

        if (
            best_match_node.evicted
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(best_match_node, mem_quota, req=req):
                new_indices = _collect_new_prefix_indices()
                if new_indices.numel() == 0:
                    return (
                        self._empty_match_result.device_indices,
                        last_best_match_device_node,
                    )

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node.id,
                )
                return new_indices, best_match_node

        return (
            self._empty_match_result.device_indices,
            last_best_match_device_node,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def flush_write_through_acks(self) -> None:
        """Flush pending write-through acknowledgements."""
        self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        swa = self.components.get(ComponentType.SWA)
        return swa.sliding_window_size if swa else None

    def swa_reprefill_tail_tokens(self) -> int:
        """
        Only unified_kv + HiCache needs this: SWA lives in a per-request ring
        (state_slot/pos), not content-stable and never offloaded to host, so a
        reused prefix's trailing sliding window would read another request's
        stale ring slots. Re-prefilling that window rewrites this request's ring
        (what plain radix reuse does via its SWA match gate). 0 for every other
        layout.
        """
        swa = self.components.get(ComponentType.SWA)
        unified_compress_only_hicache = (
            self.cache_controller is not None
            and swa is not None
            and swa._swa_kv_pool_host is None
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def supports_swa(self) -> bool:
        return ComponentType.SWA in self.components

    def supports_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    # ---- Streaming session API (delegates to composed StreamingSession) ----

    def supports_streaming_session(self) -> bool:
        return True

    def release_session(self, session_id: str) -> None:
        self.session.release_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_tokens(active_pool_idxs)

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_full_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_swa_tokens(active_pool_idxs)

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_req_count(active_pool_idxs)

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session.session_held_mamba_slots(active_pool_idxs)

    def evictable_size(self) -> int:
        return self.component_evictable_size_.get(BASE_COMPONENT_TYPE, 0)

    def protected_size(self) -> int:
        return self.component_protected_size_.get(BASE_COMPONENT_TYPE, 0)

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

    def total_size(self):
        total_size = 0
        total_aux_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            for ct in self.tree_components:
                if ct == BASE_COMPONENT_TYPE:
                    continue
                value = node.component_data[ct].value
                if value is not None:
                    total_aux_size += len(value)
            for child in node.children.values():
                stack.append(child)
        return total_size, total_aux_size

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
        if component_type not in self.components:
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

    def available_and_evictable_str(self) -> str:
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.component_evictable_size_[BASE_COMPONENT_TYPE]
        lines = [
            f"Available full tokens: {full_available_size + full_evictable} "
            f"(full_available_size={full_available_size} + full_evictable_size_={full_evictable})"
        ]
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            if ct.is_swa:
                available_size = self.token_to_kv_pool_allocator.swa_available_size()
            elif ct.is_mamba:
                available_size = self.req_to_token_pool.mamba_allocator.available_size()
            else:
                continue

            lines.append(
                f"Available {ct}: {available_size + self.component_evictable_size_[ct]} "
                f"(available_size={available_size} + component_evictable_size_={self.component_evictable_size_[ct]})"
            )
        return "\n".join(lines) + "\n"

    def _collect_all_nodes(self) -> list[UnifiedTreeNode]:
        nodes = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        return nodes

    def sanity_check(self):
        """Verify tree invariants.

        TODO(hzh): This method has relatively high latency; simplify the
        check logic once the tree implementation stabilizes.
        """
        # Skip when streaming sessions hold tree locks: the check asserts
        # all nodes are unlocked during idle, which streaming sessions break
        # by design (they hold a first-turn lock across turns).
        if self.session.any_holding_kv():
            return

        write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )

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
            for ct in self.tree_components:
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
                if full_hst and not p_hst and not write_back:
                    E(f"node {nid} backed up but parent {node.parent.id} not backed up")

            # Lock hierarchy and counters must stay sane.
            fl = node.component_data[FCT].lock_ref
            for ct in self.tree_components:
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
        queued_hst_leaves = set(self.evictable_host_scan_queue)
        if queued_hst_leaves != expected_hst_leaves:
            extra = queued_hst_leaves - expected_hst_leaves
            missing = expected_hst_leaves - queued_hst_leaves
            if extra:
                E(f"H-leaf scan queue extra: {[n.id for n in list(extra)[:5]]}")
            if missing:
                E(f"H-leaf scan queue missing: {[n.id for n in list(missing)[:5]]}")

        # D-leaf ∩ H-leaf = ∅
        overlap = self.evictable_device_leaves & self.evictable_host_leaves
        if overlap:
            E(
                f"[Leaf] {len(overlap)} in both sets: {[n.id for n in list(overlap)[:5]]}"
            )

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
        for ct in self.tree_components:
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
        for ct in self.tree_components:
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
        for nid, (n, _, _) in self.ongoing_write_through.items():
            if n not in all_node_set:
                E(f"[Ongoing] write_through node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] write_through node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )
        for nid, (n, _, _) in self.ongoing_load_back.items():
            if n not in all_node_set:
                E(f"[Ongoing] load_back node {nid} not in tree")
            elif n.component_data[FCT].lock_ref <= 0:
                E(
                    f"[Ongoing] load_back node {nid} lock_ref={n.component_data[FCT].lock_ref}"
                )

        # ── Result ──
        if errors:
            msg = (
                f"Sanity check FAILED ({len(errors)} violations "
                f"across {len(all_nodes)} nodes):\n"
                + "\n".join(f"  {e}" for e in errors)
            )
            logger.error(msg)
            self.pretty_print()
            raise AssertionError(msg)

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
                for ct in self.tree_components
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
