from __future__ import annotations

import atexit
import logging
import threading
import time
from dataclasses import replace
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, Sequence, TypeVar

import torch

from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation
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
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    validate_buffer_only_stack,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.common import RetractionBackup
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeComponentDeviceSlot,
    FreeDeviceKV,
    FreeDeviceKVFullOnly,
    ReplaceWriteThroughOnNodeSplit,
)

# UnifiedTreeNode / UnifiedLRUList live on the tree core; re-exported here
# because other modules and tests import them from this module.
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    FullComponent,
    MambaComponent,
    PrepareLoadBackResult,
    SWAComponent,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    UnifiedSessionRefTracker,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.tree_core_registry import create_tree_core
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: F401
    NodeId,
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.observability.metrics_collector import (
    StorageMetrics,
    StorageMetricsCollector,
)
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_observability,
)
from sglang.srt.session.streaming_session import StreamingSession
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheAck
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        PrefetchOperation,
    )
    from sglang.srt.mem_cache.pool_host import PoolEntry
    from sglang.srt.server_args import ServerArgs

from sglang.srt.utils.rank_consensus_checker import rank_consensus

T = TypeVar("T")

# Metric label per component, matching the host pool names used by
# hicache_backup_tokens_total and the host occupancy gauges.
_COMPONENT_POOL_LABEL = {
    ComponentType.FULL: PoolName.KV.value,
    ComponentType.SWA: PoolName.SWA.value,
    ComponentType.MAMBA: PoolName.MAMBA.value,
}


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}


logger = logging.getLogger(__name__)


class _OngoingWriteThrough(NamedTuple):
    """Tracks an in-flight D→H write-through operation."""

    node_id: NodeId
    lock_params: Optional[DecLockRefParams]
    publish_node_ids: list[NodeId]


class _OngoingLoadBack(NamedTuple):
    """Tracks an in-flight H→D load-back operation."""

    node_id: NodeId
    lock_params: DecLockRefParams
    host_lock_params: DecLockRefParams


class _OngoingPrefetch(NamedTuple):
    """Tracks an in-flight storage→host prefetch operation."""

    anchor_node_id: NodeId
    prefetch_key: RadixKey
    host_indices: torch.Tensor
    operation: PrefetchOperation
    anchor_lock_params: DecLockRefParams
    comp_xfers: dict[ComponentType, list[PoolTransfer]]


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        params: CacheInitParams,
    ):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.disable = params.disable

        if params.enable_metrics:
            self.init_metrics_collector()
        self._enable_metrics_flag = params.enable_metrics
        self.enable_storage_metrics = False
        self.storage_metrics_collector: Optional[StorageMetricsCollector] = None
        self.extra_metric_labels = None

        assert params.tree_components is not None
        self.tree_components = tuple(params.tree_components)
        self.enable_session_radix_cache = params.enable_session_radix_cache
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
        # Whether SWA is enabled.
        self.is_swa_enabled = ComponentType.SWA in params.tree_components
        # Whether Mamba is enabled.
        self.is_mamba_enabled = ComponentType.MAMBA in params.tree_components
        # Whether the mamba extra (ping-pong) buffer is enabled.
        self.enable_mamba_extra_buffer = (
            params.enable_mamba_extra_buffer if self.is_mamba_enabled else False
        )
        # SWA window size (None when SWA is not enabled).
        self._sliding_window_size = (
            params.sliding_window_size if self.is_swa_enabled else None
        )
        # The TreeCore owns the tree member-var state (structure, LRUs, sizes,
        # evictable leaves) and drives the components' tree-level hooks.
        self._tree_core_backend = envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()
        self.tree_core = create_tree_core(
            name=self._tree_core_backend,
            params=params,
            components=self.components,
        )
        # Components execute boundary actions through the tree core.
        for component in self.components.values():
            component.tree_core = self.tree_core

        # Session ref tracking (--enable-session-radix-cache).
        self.session_refs = UnifiedSessionRefTracker(
            components=self._components_tuple,
            tree_core=self.tree_core,
            enable_session_radix_cache=self.enable_session_radix_cache,
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
        self.host_pool_group = None  # set by attach_hybrid_pool_to_unified_cache
        # Owns the storage backend lifecycle; built by init_hicache.
        self._storage_attachment: Optional[StorageAttachment] = None
        self.linker: Optional[UnifiedCacheLinkerWrapper] = None
        self.prefetch_stop_policy = "best_effort"
        self.prefetch_threshold = 256
        self.prefetch_timeout_base = 1.0
        self.prefetch_timeout_per_page = 0.25
        self.hicache_storage_pass_prefix_keys = False
        # Buffer-only host memory mode (host RAM as transient GPU↔storage
        # staging, not an L2 tier); resolved in init_hicache, which also
        # constructs the pipeline collaborator (None = cache mode).
        self.host_memory_mode = "cache"
        self.buffer_pipeline: Optional[BufferModePipeline] = None
        # Write-side dedupe: beliefs about what storage already holds, so
        # re-inserts of hot prefixes skip the redundant backup.
        self.storage_existence_cache = StorageExistenceCache()
        # Cumulative prefetch-outcome counters, exported through the
        # log_storage_metrics flow.
        self._prefetch_outcome_stats: dict[str, float] = {
            "attempts": 0,
            "issued": 0,
            "declined_too_short": 0,
            "declined_rate_limited": 0,
            "declined_anchor_lost": 0,
            "declined_device_covered": 0,
            "revoked_insufficient": 0,
            "revoked_full_miss": 0,
            "l3_demand_requests": 0,
            "l3_miss_tokens": 0,
            "l1l2_miss_tokens": 0,
            "l3_demand_total_tokens": 0,
            "l3_sum_rate_all": 0.0,
            "l3_sum_rate_main_weighted": 0.0,
        }

        self.reset()
        logger.info(
            f"Init Unified Radix Cache. Components: {self.tree_components}. "
            f"Tree Core: {type(self.tree_core).__name__}"
        )

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

    def init_cache_linker(self, cache_linker: UnifiedCacheLinker) -> None:
        """Attach an external KV store directly to the device pools."""
        self.linker = UnifiedCacheLinkerWrapper(self, cache_linker)

    def reset(self) -> None:
        if self.linker is not None:
            self.linker.reset()
        self._reset_full()

    def _reset_full(self) -> None:
        """Full reset: destroy entire tree and all state."""
        self.tree_core.reset()
        self.session_refs.reset()

        # Reset Controller.
        self.session.slots.clear()
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        # Rids whose storage prefetch resolved without a usable result;
        # popped by the scheduler to pace availability-check retries.
        self._storage_prefetch_missed_rids: set[str] = set()
        self.ongoing_backup: dict[int, tuple[NodeId, DecLockRefParams]] = {}
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.reset()

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self.tree_core.kv_events.record_all_cleared()

    def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
        """Initialize HiCache infrastructure."""
        self.host_memory_mode = get_memory().hicache_host_memory_mode
        if self.host_memory_mode == "buffer_only":
            # TODO(Jialin): Extend buffer-only state handoff to Mamba in a
            # follow-up to #34798 and #35769.
            # FULL and FULL+SWA only: Mamba has no state-handoff channel on
            # the admission-time load-back read path and is not layer-gated.
            # Lifting the fence also needs the admission charge: a staged
            # state slot is request-pinned at consumption and must ride
            # req.mamba_host_hit_length the way the SWA window does.
            supported = {ComponentType.FULL, ComponentType.SWA}
            if not set(self.tree_components) <= supported:
                raise ValueError(
                    "--hicache-host-memory-mode buffer_only supports only "
                    "FULL/SWA unified trees; got components "
                    f"{sorted(ct.name for ct in self.tree_components)}."
                )
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            attach_hybrid_pool_to_unified_cache,
        )

        self.load_cache_event = threading.Event()
        self.sidecar_pool_specs.clear()
        self.extra_metric_labels = get_observability().extra_metric_labels

        # Parse storage config once, share with assembler and tree
        storage_backend = get_memory().hicache_storage_backend
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
                get_memory().hicache_storage_backend_extra_config
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
        # Tag HiCache enablement on the TreeCore.
        if self.cache_controller is not None:
            self.tree_core.set_hicache_enabled()
            if self.supports_swa():
                swa = self.components[ComponentType.SWA]
                self.tree_core.has_swa_host_pool = swa._swa_kv_pool_host is not None

        if self.host_memory_mode == "buffer_only":
            swa = self.components.get(ComponentType.SWA)
            validate_buffer_only_stack(
                sidecar_pool_specs=self.sidecar_pool_specs, swa_component=swa
            )
            self.buffer_pipeline = BufferModePipeline(
                cache=self,
                max_context_len=get_model().context_length or 0,
                swa_window_pages=(
                    swa.full_window_pages
                    if swa is not None and self.tree_core.has_swa_host_pool
                    else 0
                ),
                # Leak backstop only: live queued tokens are intrinsically
                # bounded by the FULL device pool (one intent per node, stale
                # intents swept per tick), so a cap that binds on live
                # content would drop-newest and punch storage holes.
                write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
            )
            self.cache_controller.host_write_staged_tokens_fn = lambda: (
                self.buffer_pipeline.write_staged_tokens_
            )

        # State initialization
        self.write_through_threshold = (
            1 if get_memory().hicache_write_policy == "write_through" else 2
        )
        self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        )
        # Pre-seed the dropped-tokens series at 0 per pool
        if self.metrics_collector is not None and self.cache_controller is not None:
            for ct in self.tree_components:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=0,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )
        self.load_back_threshold = 10
        self.prefetch_stop_policy = get_memory().hicache_storage_prefetch_policy

        # Runtime attach/detach of the L3 backend (startup, admin API, atexit).
        self._storage_attachment = StorageAttachment(self)
        atexit.register(self.shutdown)

        if storage_backend is not None:
            self._storage_attachment.apply_runtime_config(
                storage_backend=storage_backend,
                prefetch_threshold=storage_prefetch_threshold,
                prefetch_timeout_base=prefetch_timeout_base,
                prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
                enable_storage=self.cache_controller.enable_storage,
                enable_storage_metrics=self._enable_metrics_flag,
                extra_metric_labels=self.extra_metric_labels,
            )

    def register_sidecar_pool(
        self, spec: SidecarPoolSpec, entry: Optional[PoolEntry] = None
    ) -> None:
        if entry is not None:
            if self.cache_controller is None:
                raise RuntimeError("HiCache controller is not attached.")
            self.cache_controller.register_host_pool_entry(entry)
        self.sidecar_pool_specs.append(spec)

    def release_host_resources(self) -> None:
        if self.linker is not None:
            self.linker.close()
        if self.host_pool_group is not None:
            self.host_pool_group.destroy()

    @rank_consensus(
        same_params=["params"],
        same_results=["result.full_kv_hit_length", "result.swa_host_hit_length"],
    )
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        if self.disable:
            return self.tree_core.empty_match_result
        result = self.tree_core.match_prefix(params)
        # Apply the walk's actions (e.g. a pending write-through relocation on
        # a split) before the finalizers, which can evict or raise.
        self._apply_cache_actions(result.cache_actions)
        for component in self._components_tuple:
            result = component.finalize_match_result_in_cache(params, result)
        # Finalizers must not emit actions; the walk's were applied above.
        assert not result.cache_actions
        if self.linker is not None and params.req is not None:
            result = self.linker.match(params.key, params.req, result)
        return result

    def is_chunk_cache(self) -> bool:
        return self.disable

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        # Fail fast on re-entrancy without touching the in-flight walk.
        assert not self.tree_core.has_ongoing_insert(), "re-entrant insert"
        # Pump the resumable insert, applying each step's actions at its barrier.
        try:
            step = self.tree_core.begin_insert(params)
            while True:
                self._apply_cache_actions(step.actions)
                if step.result is not None:
                    # Walk actions flow through the steps; the result is action-free.
                    assert not step.result.cache_actions
                    return step.result
                step = self.tree_core.resume_insert()
        finally:
            # Drain still-pending actions so frees reach the allocator on abort.
            self._apply_cache_actions(self.tree_core.end_insert())

    def evict(self, params: EvictParams) -> EvictResult:
        return self._evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict until the requested component allocations become feasible.

        ``params`` contains allocator shortfalls, not absolute eviction quotas.
        A component eviction can cascade to its peers; with a shared memory pool,
        those collateral frees can satisfy the original allocation before the
        triggering component's requested count is reached.
        """
        if self.disable:
            return EvictResult()

        request_by_type = self._evict_request_by_type(params)
        available_size_targets = {
            ct: self._component_available_size(ct) + request_cnt
            for ct, request_cnt in request_by_type.items()
            if request_cnt > 0
        }
        return self._evict(params, available_size_targets)

    @staticmethod
    def _evict_request_by_type(params: EvictParams) -> dict[ComponentType, int]:
        return {
            ComponentType.FULL: params.num_tokens,
            ComponentType.SWA: params.swa_num_tokens,
            ComponentType.MAMBA: params.mamba_num,
            ComponentType.C128: 0,
        }

    def _component_available_size(self, component_type: ComponentType) -> int:
        """Return capacity usable by the component's next allocation.

        Shared allocators expose schedulable capacity, which includes peer holes
        that an urgent allocator flush can reclaim without further eviction.
        """
        if component_type == ComponentType.FULL:
            if self.supports_swa():
                return self.token_to_kv_pool_allocator.full_available_size()
            return self.token_to_kv_pool_allocator.available_size()
        if component_type == ComponentType.SWA:
            return self.token_to_kv_pool_allocator.swa_available_size()
        if component_type == ComponentType.MAMBA:
            return self.req_to_token_pool.mamba_allocator.schedulable_available_size()
        raise ValueError(f"Unsupported cache component: {component_type}")

    def _evict(
        self,
        params: EvictParams,
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        request_by_type = self._evict_request_by_type(params)
        self._evict_components(
            request_by_type,
            tracker,
            available_size_targets=available_size_targets,
        )

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        # Report full-layer tokens only
        self.update_eviction_metrics(tracker[BASE_COMPONENT_TYPE], start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )

    def _free_values(
        self,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Free a tree-side step's returned device and host values right away."""
        # Both drains must run even if one raises.
        try:
            self._drain_device_frees(device_frees)
        finally:
            self._drain_host_frees(host_frees)

    def _accumulate_tracker(
        self,
        tracker: dict[ComponentType, int],
        delta: dict[ComponentType, int],
    ) -> None:
        """Fold a step result's evicted delta into the running totals."""
        for ct, n in delta.items():
            tracker[ct] += n

    def _evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> tuple[Optional[NodeId], bool]:
        """Advance the eviction walk one node, consuming its step result."""
        result = self.tree_core.evict_device_next_node(component_type, tracker)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.node_id, result.made_progress

    def _evict_device_leaf(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> Optional[BackupKV]:
        """Evict one device leaf, consuming its step result; returns the
        deferred write-back BackupKV when one must run before the demote."""
        result = self.tree_core.evict_device_leaf(node_id, self.is_write_back)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.backup_kv

    def _demote(self, node_id: NodeId, tracker: dict[ComponentType, int]) -> None:
        """Demote a backed-up node, consuming its step result."""
        result = self.tree_core.demote(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)

    def _drop_subtree_no_host(
        self, node_id: NodeId, tracker: dict[ComponentType, int]
    ) -> bool:
        """Run the write-back drop fallback, consuming its step result."""
        result = self.tree_core.drop_subtree_no_host(node_id)
        self._free_values(result.device_frees, result.host_frees)
        self._accumulate_tracker(tracker, result.tracker)
        return result.is_dropped

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        available_size_targets: Optional[dict[ComponentType, int]] = None,
    ) -> None:
        # Buffer mode: eviction always wins over queued backup intents — a
        # destroyed victim's intent is stale-swept and the content rewrites
        # after its recompute.

        def target_reached(component_type: ComponentType) -> bool:
            if available_size_targets is None:
                return False
            target = available_size_targets.get(component_type)
            # Do not compact on every eviction step. Shared allocators include
            # drainable peer holes here and flush the peer once in alloc().
            return (
                target is not None
                and self._component_available_size(component_type) >= target
            )

        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # A preceding component may have cascade-evicted this component or,
            # on a shared pool, released enough bytes to satisfy its allocation.
            if tracker[ct] >= request_cnt or target_reached(ct):
                continue
            self.tree_core.evict_device_start(ct, request_cnt)
            try:
                while not target_reached(ct):
                    node_id, made_progress = self._evict_device_next_node(ct, tracker)
                    if node_id is None:
                        if made_progress:
                            # Internal tombstone frees are now allocator-visible;
                            # recheck the allocation target before walking again.
                            continue
                        break
                    backup_kv = self._evict_device_leaf(node_id, tracker)
                    if backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        written = self._execute_and_commit_kv_backup(
                            backup_kv, write_back=True
                        )
                        freed_before_drop = dict(tracker)
                        if written > 0:
                            self.writing_check(write_back=True)
                            self._demote(node_id, tracker)
                        elif self._drop_subtree_no_host(node_id, tracker):
                            self._record_dropped_tokens(tracker, freed_before_drop)
                            logger.warning(
                                "write_back: KV subtree dropped without backup "
                                "due to host memory pressure, root node %d",
                                node_id,
                            )
                        else:
                            logger.warning(
                                "write_back: backup failed under host memory "
                                "pressure but subtree drop declined (node "
                                "locked); root node %d stays device-resident "
                                "until host space frees",
                                node_id,
                            )
            finally:
                self.tree_core.evict_device_end(ct)

    def _record_dropped_tokens(
        self,
        tracker: dict[ComponentType, int],
        freed_before_drop: dict[ComponentType, int],
    ) -> None:
        """Record per-pool tokens dropped without backup under host pressure."""
        if self.metrics_collector is None:
            return
        for ct, freed in tracker.items():
            dropped = freed - freed_before_drop[ct]
            if dropped > 0:
                self.metrics_collector.increment_dropped_tokens(
                    num_tokens=dropped,
                    reason="host_pressure",
                    pool=_COMPONENT_POOL_LABEL[ct],
                )

    def inc_lock_ref(
        self, node_id: NodeId, skip_lock_components: Sequence[ComponentType] = ()
    ) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node_id)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id, skip_lock_components)

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node_id, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_lock_ref(node_id, params, skip_swa)

    def _dec_req_lock(self, req: Req, *, skip_swa: bool = False) -> None:
        """Release the tree lock a request holds on its last_node, honoring the
        components it skipped locking so it never drops a lock it never took."""
        self.dec_lock_ref(
            req.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=req.swa_uuid_for_lock,
                skip_lock_node_ids=req.skip_lock_node_ids,
            ),
            skip_swa=skip_swa,
        )

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int] = None,
        skip_lock_node_ids: Optional[dict] = None,
    ) -> None:
        if self.disable:
            return
        result = self.tree_core.dec_swa_lock_only(
            node_id, swa_uuid_for_lock, skip_lock_node_ids
        )
        self._free_values(result.device_frees, result.host_frees)

    def inc_host_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_host_lock_ref(node_id)

    def dec_host_lock_ref(
        self, node_id: NodeId, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult()
        return self.tree_core.dec_host_lock_ref(node_id, params)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if self.session.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return

        if self.disable:
            self.free_kv_row(req.kv, [(0, kv_len_to_handle)])
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :kv_len_to_handle
        ]

        result = None
        insert_params = None

        if is_insert:
            insert_params = InsertParams(
                prev_prefix_len=req.kv.cache_protected_len,
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

            # Truncate if needed; the tail free is deferred and batched with
            # the unaligned tail below so a shared boundary page is emitted once.
            kv_indices_full = kv_indices
            tail_free_start = None
            if effective_cache_len < len(token_ids):
                tail_free_start = max(effective_cache_len, req.kv.cache_protected_len)
                token_ids = token_ids[:effective_cache_len]
                kv_indices = kv_indices[:effective_cache_len]

            radix_key = RadixKey(
                token_ids,
                req.extra_key,
                is_bigram=self.tree_core.is_eagle,
                cache_salt=req.cache_salt,
            ).page_aligned(self.page_size)
            page_aligned_len = len(radix_key)
            values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

            insert_params.key = radix_key
            insert_params.value = values
            result = self.insert(insert_params)

            # Free unaligned tail (+ deferred truncation tail)
            ranges = [(page_aligned_len, len(kv_indices))]
            if tail_free_start is not None:
                ranges.append((tail_free_start, len(kv_indices_full)))
            self.free_kv_row(req.kv, ranges)
        else:
            self.free_kv_row(req.kv, [(req.kv.cache_protected_len, kv_len_to_handle)])

        # Synthetic profiling requests may own KV without locking a tree node.
        if req.last_node is not None:
            self._dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)

        if is_insert and result is not None and result.last_device_node is not None:
            req.last_node = result.last_device_node

        # cleanup
        for comp in self._components_tuple:
            comp.cleanup_after_caching_req(
                req, is_finished=True, insert_result=result, insert_params=insert_params
            )

        if self.enable_session_radix_cache and result is not None:
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            if req.finished_reason is not None and not isinstance(
                req.finished_reason, FINISH_ABORT
            ):
                self.session_refs.register_session_ref(req)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        if self.session.try_cache_unfinished_req(req, chunked=chunked, **kwargs):
            return

        token_ids = req.get_fill_ids()

        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, : len(token_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : len(token_ids)
        ]

        # components prepare insert data + return effective cache_len
        insert_params = InsertParams(
            prev_prefix_len=req.kv.cache_protected_len,
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

        radix_key = RadixKey(
            token_ids[:effective_cache_len],
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        )

        if envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get():
            # The frontier lands a page below page_floor(pre_len + 1), which has to
            # be where the insert stops, or the leaf it creates keeps less than a
            # sliding window of live SWA and the match after the insert rejects it.
            # The insert stops at page_floor(len(radix_key)), and a bigram key is
            # one shorter than the tokens it spans, so measure the key.
            for comp in self._components_tuple:
                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)

        if effective_cache_len <= 0:
            req.prefix_indices = kv_indices_orig.to(dtype=torch.int64, copy=True)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(
                    req, is_finished=False, insert_params=insert_params
                )
            return

        kv_indices = kv_indices_orig[:effective_cache_len]

        radix_key = radix_key.page_aligned(self.page_size)
        page_aligned_len = len(radix_key)
        values = kv_indices[:page_aligned_len].to(dtype=torch.int64, copy=True)

        insert_params.key = radix_key
        insert_params.value = values
        result = self.insert(insert_params)

        # Match prefix. SWA insertion retains one extra window before the
        # page-aligned boundary, so the normal match remains safe to repoint.
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key, req=req))
        new_indices = match_result.device_indices
        new_last_node = match_result.last_device_node
        new_prefix_len = result.prefix_len
        assert req.kv.cache_protected_len <= len(new_indices) + self.page_size - 1, (
            f"{req.kv.cache_protected_len=}, {len(new_indices)=}, {page_aligned_len=}"
        )
        assert new_prefix_len <= len(new_indices), (
            f"{new_prefix_len=}, {len(new_indices)=}"
        )
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(req.kv.cache_protected_len, len(new_indices))),
            new_indices[req.kv.cache_protected_len :],
        )

        self._dec_req_lock(req)
        # Opt-in: leave the matched-prefix mamba evictable during decode (it is
        # already COW'd to the request's own slot, never read from this node again).
        # Safe only because any future COW source is the COWing request's own
        # admission-locked last_node (recorded only if still present, locked before
        # the next alloc) -- not this evictable node. A scheduler that matched a
        # whole batch before locking would break that. Off = original full lock.
        skip_lock_components = (
            (ComponentType.MAMBA,)
            if envs.SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK.get()
            else ()
        )
        lock_result = self.inc_lock_ref(
            new_last_node, skip_lock_components=skip_lock_components
        )

        # Update req fields
        if len(new_indices) < len(kv_indices_orig):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices_orig[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.kv.cache_protected_len = len(new_indices)
        req.last_node = new_last_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        # carry the skip set so this node's dec releases only what we locked
        req.skip_lock_node_ids = lock_result.skip_lock_node_ids
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

    def _apply_cache_actions(
        self, actions: list[CacheAction | ComponentAction]
    ) -> None:
        # Apply and consume one at a time: a spent list cannot be double-applied.
        actions.reverse()
        try:
            while actions:
                self._apply_cache_action(actions.pop())
        finally:
            actions.reverse()

    def _apply_cache_action(self, action: CacheAction | ComponentAction) -> None:
        # Component actions route to their component class; the rest are
        # cache-owned and handled here by type.
        if isinstance(action, ComponentAction):
            self.components[action.component_type].apply_component_action(action)
        elif isinstance(action, ReplaceWriteThroughOnNodeSplit):
            self._replace_pending_write_through_node(
                action.ack_id,
                action.old_node_id,
                [action.new_node_id, action.new_child_node_id],
            )
            if self.linker is not None:
                self.linker.replace_pending_offload_node(
                    action.ack_id,
                    action.old_node_id,
                    [action.new_node_id, action.new_child_node_id],
                )
        elif isinstance(action, FreeDeviceKV):
            # tree values are page-aligned copies of a kv row: page-exact segments
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_segment(indices, start_pos=0)
        elif isinstance(action, FreeDeviceKVFullOnly):
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free_full(indices)
        elif isinstance(action, BackupKV):
            if self.linker is not None:
                self.linker.offload_nodes(action.node_ids)
            else:
                self._execute_and_commit_kv_backup(action)
        else:
            raise AssertionError(f"unhandled CacheAction: {type(action).__name__}")

    def _drain_device_frees(
        self, device_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component device slots, consuming each entry as it frees.
        for ct in list(device_frees):
            self._apply_cache_action(
                FreeComponentDeviceSlot(device_frees.pop(ct), component_type=ct)
            )

    def _drain_host_frees(
        self, host_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component host-pool slots, consuming each entry as it frees.
        for ct in list(host_frees):
            self.components[ct].free_host_values(host_frees.pop(ct))

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        if self.host_memory_mode == "buffer_only":
            # The tree never holds host values in buffer mode, and staging
            # is operation-owned (freed at each ack): nothing is evictable.
            return 0
        result = self.tree_core.drive_host_eviction(component_type, num_tokens)
        self._free_values(result.device_frees, result.host_frees)
        return result.tracker.get(component_type, 0)

    # ---- Decode retraction ----

    def supports_retraction_backup(self) -> bool:
        if self.cache_controller is None or self.host_pool_group is None:
            return False
        if self.supports_mamba():
            return False

        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, SWAKVPool):
            return (
                self.supports_swa()
                and {
                    PoolName.KV,
                    PoolName.SWA,
                }
                <= self.host_pool_group.entry_map.keys()
            )
        return isinstance(kv_cache, MHATokenToKVPool) and (
            PoolName.KV in self.host_pool_group.entry_map
        )

    def validate_retraction_host_capacity(self) -> None:
        if not self.supports_retraction_backup():
            raise ValueError(
                "--disaggregation-decode-retraction-backup=host_pool requires "
                "an MHA or hybrid-SWA HiCache host stack."
            )

        for spec in self.sidecar_pool_specs:
            source_size = self.host_pool_group.entry_map[
                spec.indices_from_pool
            ].host_pool.logical_size
            sidecar_size = self.host_pool_group.entry_map[
                spec.pool_name
            ].host_pool.logical_size
            if sidecar_size < source_size:
                raise ValueError(
                    "Retraction sidecar host pool is smaller than its index source: "
                    f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                    f"source={spec.indices_from_pool}, source_slots={source_size}."
                )

    @staticmethod
    def _pad_retraction_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
        aligned_len = ceil_align(len(indices), page_size)
        if aligned_len == len(indices):
            return indices
        tail = indices[-1] + torch.arange(
            1,
            aligned_len - len(indices) + 1,
            dtype=torch.int64,
            device=indices.device,
        )
        return torch.cat([indices, tail])

    def _retraction_device_transfers(
        self, req: Req
    ) -> tuple[torch.Tensor, list[PoolTransfer]]:
        num_tokens = req.seqlen - 1
        full_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :num_tokens
        ].to(torch.int64)
        full_indices = self._pad_retraction_indices(full_indices, self.page_size)

        component_transfers: dict[ComponentType, list[PoolTransfer]] = {}
        if self.supports_swa():
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            assert self.sliding_window_size is not None
            window_start = max(0, num_tokens - self.sliding_window_size)
            window_start = window_start // self.page_size * self.page_size
            window_indices = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, window_start:num_tokens
            ].to(torch.int64)
            swa_indices = kv_cache.translate_loc_from_full_to_swa(window_indices)
            assert bool((swa_indices > 0).all()), (
                f"unmapped SWA window positions for request {req.rid}"
            )
            component_transfers[ComponentType.SWA] = [
                PoolTransfer(
                    name=PoolName.SWA,
                    device_indices=self._pad_retraction_indices(
                        swa_indices, self.page_size
                    ),
                )
            ]

        kv_transfer = PoolTransfer(name=PoolName.KV, device_indices=full_indices)
        extra_transfers = [
            transfer
            for transfers in component_transfers.values()
            for transfer in transfers
        ]
        extra_transfers.extend(
            self._build_sidecar_transfers(
                CacheTransferPhase.BACKUP_HOST,
                kv_transfer,
                component_transfers,
            )
        )
        return full_indices, extra_transfers

    def _reclaim_retraction_host(self, num_tokens: int) -> int:
        if self.disable:
            return 0
        return self.evict_host(num_tokens)

    def retraction_backup(self, req: Req) -> Optional[RetractionBackup]:
        """Back up device KV to the host pool; None when it cannot fit after reclaim."""
        assert req.seqlen > 1

        device_indices, extra_transfers = self._retraction_device_transfers(req)
        host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            self._reclaim_retraction_host(len(device_indices))
            host_indices = self.host_pool_group.alloc(len(device_indices))
        if host_indices is None:
            return None

        resolved = self.host_pool_group.resolve_host_transfers(
            extra_transfers or None,
            primary_device_indices=device_indices,
            primary_host_indices=host_indices,
        )
        if resolved is None and extra_transfers:
            self.host_pool_group.free(host_indices)
            return None

        backup = RetractionBackup(
            host_indices=host_indices,
            pool_transfers=[replace(x, device_indices=None) for x in resolved or []]
            or None,
        )
        operation = CacheOperation(
            host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        try:
            write_host, write_device, write_pools = (
                self.cache_controller._move_write_operation(operation)
            )
            completion = self.cache_controller.l2_transfer_engine.submit_device_to_host(
                self.cache_controller._l2_transfers(
                    write_host, write_device, write_pools
                )
            )
            completion.finish_event.synchronize()
        except Exception:
            self.retraction_discard(backup)
            raise
        return backup

    def retraction_restore(self, req: Req, backup: RetractionBackup) -> None:
        device_indices, current_transfers = self._retraction_device_transfers(req)
        assert len(backup.host_indices) == len(device_indices), (
            f"Host backup has {len(backup.host_indices)} slots, but restore has "
            f"{len(device_indices)}"
        )

        current_by_name = {transfer.name: transfer for transfer in current_transfers}
        saved_by_name = {
            transfer.name: transfer for transfer in backup.pool_transfers or []
        }
        assert current_by_name.keys() == saved_by_name.keys(), (
            f"Host backup pools {set(saved_by_name)} do not match restore pools "
            f"{set(current_by_name)}"
        )
        restored_transfers = [
            replace(
                saved,
                device_indices=current_by_name[name].device_indices,
            )
            for name, saved in saved_by_name.items()
        ]
        resolved = self.cache_controller._resolve_device_transfers(
            restored_transfers or None,
            kv_device_indices=device_indices,
            kv_host_indices=backup.host_indices,
        )
        assert resolved is not None or not restored_transfers

        operation = CacheOperation(
            backup.host_indices,
            device_indices,
            node_id=-1,
            pool_transfers=resolved,
        )
        load_host, load_device, load_pools = self.cache_controller._move_op_indices(
            operation
        )
        completion = self.cache_controller.l2_transfer_engine.submit_host_to_device(
            self.cache_controller._l2_load_transfers(
                load_host, load_device, load_pools
            ),
            layer_num=self.cache_controller.layer_num,
        )
        completion.finish_event.synchronize()
        self.retraction_discard(backup)

    def retraction_discard(self, backup: RetractionBackup) -> None:
        self.host_pool_group.free(backup.host_indices)
        self.host_pool_group.release_transfers(backup.pool_transfers)

    # ---- HiCache: Backup / LoadBack ----

    def _execute_and_commit_kv_backup(
        self, action: BackupKV, write_back: bool = False
    ) -> int:
        """Run a backup action top-down, stopping at the first failed backup."""
        if self.buffer_pipeline is not None:
            # Buffer mode bypasses the host-backup contiguity below: nothing
            # is ever host-backuped here. Contiguity comes from end-to-end
            # FIFO ordering instead (BackupKV chains are parent-before-child
            # and every pipeline stage drains in order).
            for node_id in action.node_ids:
                self.buffer_pipeline.enqueue_backup_intent(node_id)
            return 0
        written = 0
        for node_id in action.node_ids:
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
            # Overlapping chain actions may revisit nodes with Full KV already
            # backed up. Skip only when no transfer remains.
            if device_value.numel() == 0 and not comp_xfers:
                continue
            sidecar_xfers = self._build_backup_sidecar(device_value, comp_xfers)
            host_indices = self._execute_kv_backup(
                node_id, device_value, comp_xfers, sidecar_xfers
            )
            if host_indices is None:
                return 0
            self.tree_core.commit_backup(node_id, host_indices, comp_xfers)
            lock_params = None
            if not write_back:
                lock_params = self.inc_lock_ref(node_id).to_dec_params()
            self._track_write_through_node(node_id, lock_params)
            written = len(host_indices)
        return written

    def _build_backup_sidecar(self, device_value, comp_xfers):
        """Gather sidecar transfer spec."""
        kv_xfer = PoolTransfer(name=PoolName.KV, device_indices=device_value)
        return self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_HOST, kv_xfer, comp_xfers
        )

    def _execute_kv_backup(self, node_id, device_value, comp_xfers, sidecar_xfers):
        """Execute Backup action."""
        kv_tokens = len(device_value)
        host_avail = self.cache_controller.mem_pool_host.available_size()
        if host_avail < kv_tokens:
            needed = kv_tokens - host_avail
            if self.evict_host(needed) < needed:
                return None
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        return self.cache_controller.write(
            device_value, node_id=node_id, extra_pools=aux_xfers or None
        )

    def _track_write_through_node(
        self,
        node_id: NodeId,
        lock_params: Optional[DecLockRefParams],
    ) -> None:
        self.tree_core.mark_write_through_pending(node_id)
        self.ongoing_write_through[node_id] = _OngoingWriteThrough(
            node_id, lock_params, [node_id]
        )

    def _replace_pending_write_through_node(
        self, ack_id: int, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node_id, lock_params, publish_node_ids = pending
        updated_node_ids = []
        replaced = False
        for node_id in publish_node_ids:
            if node_id == old_node_id:
                updated_node_ids.extend(new_node_ids)
                replaced = True
            else:
                updated_node_ids.append(node_id)

        if not replaced:
            return

        self.ongoing_write_through[ack_id] = _OngoingWriteThrough(
            lock_node_id,
            lock_params,
            updated_node_ids,
        )

    def _finish_write_through_ack(self, ack_id: int) -> None:
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.finish_backup_ack(ack_id)
            return

        lock_node_id, lock_params, publish_node_ids = self.ongoing_write_through.pop(
            ack_id
        )
        self.tree_core.finish_write_through(publish_node_ids, ack_id)
        if lock_params is not None:
            self.dec_lock_ref(lock_node_id, lock_params)
        if self.enable_storage:
            # Back up each fragment: after a split, lock_node only holds the
            # suffix; the prefix fragment must be persisted as well.
            for node_id in publish_node_ids:
                self.write_backup_storage(node_id)

    def load_back(
        self,
        node_id: NodeId,
        mem_quota: Optional[int] = None,
        req=None,
    ) -> bool:
        """Load evicted KV data from host back to device (H→D)."""
        if self.cache_controller is None:
            return False

        host_anchor_params = self.inc_host_lock_ref(node_id).to_dec_params()

        # Lock the path before building transfers (the aux build can evict).
        result = self.inc_lock_ref(node_id)
        ancestor_lock_params = result.to_dec_params()

        # Let each component pre-allocate per-request state for the load-back;
        # the finally below lets components recover it unless the load succeeds.
        preps: dict[ComponentType, PrepareLoadBackResult] = {
            comp.component_type: comp.prepare_load_back(node_id, req=req)
            for comp in self._components_tuple
        }
        success = False
        try:
            success = self._load_back_transfers(
                node_id=node_id,
                mem_quota=mem_quota,
                req=req,
                result=result,
                ancestor_lock_params=ancestor_lock_params,
                host_anchor_params=host_anchor_params,
            )
            return success
        finally:
            for comp in self._components_tuple:
                comp.finalize_load_back(req, preps[comp.component_type], success)

    def _load_back_transfers(
        self,
        *,
        node_id: NodeId,
        mem_quota: Optional[int],
        req,
        result: IncLockRefResult,
        ancestor_lock_params: DecLockRefParams,
        host_anchor_params: DecLockRefParams,
    ) -> bool:
        # Build the KV + per-component aux transfers.
        kv_xfer, comp_xfers = self.tree_core.build_load_back_spec(node_id, req=req)
        kv_tokens = len(kv_xfer.host_indices)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.LOAD_BACK, kv_xfer, comp_xfers
        )

        # Skip if there is nothing to load, or if the Full-KV transfer is too
        # small / exceeds memory quota. Aux transfers should still run even
        # when the Full-KV load is skipped by thresholding. max(1, ...): an
        # entirely empty spec (e.g. foreign-pin rejection) must never report
        # success, even at load_back_threshold <= 0.
        if (kv_tokens < max(1, self.load_back_threshold) and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(node_id, ancestor_lock_params)
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False

        avail = self._component_available_size(ComponentType.FULL)
        if avail < kv_tokens:
            needed = kv_tokens - avail
            self.evict_for_alloc(EvictParams(num_tokens=needed))
            if self._component_available_size(ComponentType.FULL) < kv_tokens:
                self.dec_lock_ref(node_id, ancestor_lock_params)
                self.dec_host_lock_ref(node_id, host_anchor_params)
                return False

        # Load H→D
        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        device_indices = self.cache_controller.load(
            host_indices=kv_xfer.host_indices,
            node_id=node_id,
            extra_pools=aux_xfers or None,
        )

        self.dec_lock_ref(node_id, ancestor_lock_params)
        if device_indices is None:
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False

        # Commit the loaded KV back onto the node + apply its emitted actions.
        self._apply_cache_actions(
            self.tree_core.commit_load_back(
                node_id, device_indices, kv_xfer, comp_xfers
            )
        )

        self.ongoing_load_back[node_id] = _OngoingLoadBack(
            node_id,
            self.inc_lock_ref(node_id).to_dec_params(),
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

    def write_backup_storage(self, node_id: NodeId) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return
        spec = self.tree_core.build_storage_backup_spec(
            node_id, self.hicache_storage_pass_prefix_keys
        )
        if spec is None:
            return

        kv_xfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=spec.host_value,
            keys=spec.hash_value,
        )
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.BACKUP_STORAGE, kv_xfer, spec.comp_xfers
        )
        aux_xfers = [x for xfers in spec.comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)

        operation_id = self.cache_controller.write_storage(
            spec.host_value,
            spec.token_ids,
            spec.hash_value,
            spec.prefix_keys,
            extra_pools=aux_xfers or None,
        )
        self.ongoing_backup[operation_id] = (
            node_id,
            self.inc_host_lock_ref(node_id).to_dec_params(),
        )

    def is_backuped(self, node_id: NodeId) -> bool:
        return self.tree_core.is_backuped(node_id)

    def is_root(self, node_id: NodeId) -> bool:
        return self.tree_core.is_root(node_id)

    def get_last_hash_value(self, node_id: NodeId) -> Optional[str]:
        return self.tree_core.get_last_hash_value(node_id)

    def get_prefix_hash_values(self, node_id: NodeId) -> list[str]:
        return self.tree_core.get_prefix_hash_values(node_id)

    def query_storage_hit_length(
        self,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> int:
        """Synchronously probe L3 storage for the reusable prefix length."""
        if (
            not self.enable_storage
            or self.cache_controller is None
            or self.cache_controller.prefetch_rate_limited()
        ):
            return 0

        extra_key, cache_salt = self.tree_core.prefetch_anchor_info(last_host_node_id)
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            PrefetchOperation,
        )

        operation = PrefetchOperation(
            "__storage_hit_query__",
            prefetch_key,
            last_hash,
            prefix_keys,
        )
        _, storage_hit_count = self.cache_controller._storage_hit_query(operation)
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count -= storage_hit_count % self.page_size
        return storage_hit_count

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return

        buffer_mode = self.host_memory_mode == "buffer_only"
        # Key the span by the request's namespace, not the anchor's (a root
        # anchor has none): a span published under the wrong namespace gets
        # re-owned by the request's own insert (double free).
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        assert (anchor_extra_key is None or anchor_extra_key == extra_key) and (
            anchor_cache_salt is None or anchor_cache_salt == cache_salt
        ), (
            f"prefetch anchor namespace {(anchor_extra_key, anchor_cache_salt)} "
            f"!= request namespace {(extra_key, cache_salt)}"
        )
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        stats = self._prefetch_outcome_stats
        if prefetch_length > 0:
            stats["attempts"] += 1
        if prefetch_length < self.prefetch_threshold:
            if prefetch_length > 0:
                stats["declined_too_short"] += 1
            # A too-short/fully-matched suffix can become a full recompute if
            # the device match evicts while queued; arm the paced retry.
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if not buffer_mode and self.cache_controller.prefetch_rate_limited():
            stats["declined_rate_limited"] += 1
            self._storage_prefetch_missed_rids.add(req_id)
            return
        if req_id in self.ongoing_prefetch or (
            buffer_mode and self.buffer_pipeline.has_staged(req_id)
        ):
            # A fetch (or an unconsumed hold) already exists for this rid;
            # overwriting would leak its staging slots.
            return

        # Buffer mode holds no tree state during the fetch: buffers are
        # operation-owned, so the anchor needs no pin.
        anchor_lock_params = (
            None
            if buffer_mode
            else self.inc_host_lock_ref(last_host_node_id).to_dec_params()
        )
        comp_xfers: dict[ComponentType, list[PoolTransfer]] = {}
        alloc_failed = False
        for ct in self.tree_components:
            if ct == BASE_COMPONENT_TYPE:
                continue
            # Pre-allocate the component's prefetch host buffer so the build stays pure.
            prep = self.components[ct].prepare_prefetch(
                last_host_node_id, prefetch_tokens=len(prefetch_key)
            )
            if prep.alloc_failed:
                alloc_failed = True
                break
            if prep.host_indices is None:
                continue
            transfers = self.tree_core.build_hicache_transfers(
                ct,
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                token_ids=prefetch_key.token_ids,
                prefetch_tokens=len(prefetch_key),
                last_hash=last_hash,
                host_indices=prep.host_indices,
            )
            if transfers:
                comp_xfers[ct] = transfers
        kv_xfer = PoolTransfer(name=PoolName.KV, host_indices=None)
        sidecar_xfers = self._build_sidecar_transfers(
            CacheTransferPhase.PREFETCH, kv_xfer, comp_xfers
        )
        if alloc_failed:
            # The whole storage fetch is forfeited over one aux staging
            # alloc (e.g. a single SWA window) — count it, or write-burst
            # starvation of the aux pool reads as generic hit-rate loss.
            if (
                self.enable_storage_metrics
                and self.storage_metrics_collector is not None
            ):
                self.storage_metrics_collector.log_prefetch_aux_alloc_failed_tokens(
                    len(prefetch_key)
                )
            self.cache_controller.append_host_mem_release(
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            if anchor_lock_params is not None:
                self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            # Forfeited over transient staging pressure; retryable.
            self._storage_prefetch_missed_rids.add(req_id)
            return

        aux_xfers = [x for xfers in comp_xfers.values() for x in xfers]
        aux_xfers.extend(sidecar_xfers)
        operation = self.cache_controller.prefetch(
            req_id,
            prefetch_key,
            last_hash,
            prefix_keys,
            extra_pools=aux_xfers or None,
        )
        stats["issued"] += 1
        # Snapshots for the L3 miss accounting at the query outcome (the
        # hit/revoke drains): requested span and total prompt length.
        operation.stats_requested_tokens = prefetch_length
        operation.stats_total_tokens = prefetch_length + len(
            matched_prefix_tokens or []
        )
        self.ongoing_prefetch[req_id] = _OngoingPrefetch(
            last_host_node_id,
            prefetch_key,
            None,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        if buffer_mode:
            self.buffer_pipeline.set_prefix_ctx(
                req_id,
                matched_prefix_tokens,
                extra_key=extra_key,
                cache_salt=cache_salt,
            )
            # Pin the just-matched anchor now: deferred to IO commit it is
            # often already deleted under churn. The IO-commit call remains
            # as the second chance that decides the fetch's fate.
            self.buffer_pipeline.try_lock_anchor(req_id)
        else:
            # Cache mode reserves the requested span up front; buffer mode
            # grants occupancy later at hit-alloc time, sized to the hit.
            self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    @rank_consensus(same_results=True)
    def _can_terminate_prefetch(self, operation: PrefetchOperation) -> bool:
        if self.prefetch_stop_policy == "best_effort":
            return True
        if self.prefetch_stop_policy == "wait_complete":
            return False
        elif self.prefetch_stop_policy == "timeout":
            # Wall-clock time may differ among ranks, all-reduce is needed to ensure
            # all ranks reach the same final result. Otherwise PP/TP ranks will diverge.
            #
            # For TP, if any rank reaches the timeout, the final result is timeout.
            #
            # For PP, PP0 makes the decision and other ranks follow PP0's decision.
            should_terminate = False
            if self.pp_rank == 0:
                should_terminate = self._prefetch_timeout_check_linear_func(operation)
            should_terminate_tensor = torch.tensor(
                int(should_terminate), dtype=torch.int, device="cpu"
            )
            self._all_reduce(should_terminate_tensor, torch.distributed.ReduceOp.MAX)
            return should_terminate_tensor.item() == 1
        else:
            return True

    @rank_consensus(same_params=True, same_results=True)
    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        _, _, _, operation, _, _ = self.ongoing_prefetch[req_id]

        # Determine whether or not we should terminate this prefetch request.
        should_terminate = operation.is_terminated() or self._can_terminate_prefetch(
            operation
        )

        if not should_terminate:
            return False

        self.cache_controller.terminate_prefetch(operation)
        if operation.host_indices is None:
            self._storage_prefetch_missed_rids.add(req_id)
            self.revoke_pending_prefetch(req_id)
        else:
            self._handle_prefetch_result(operation)
        return True

    def _handle_prefetch_result(self, operation: PrefetchOperation) -> None:
        # This function **owns**:
        # - host_indices[0 : completed_tokens]
        # - sidecar pool hits if operation.pool_transfers_done is true
        #
        # That is, when this function returns the host memory referenced must be inserted
        # into the radix tree or released to pool.

        req_id = operation.request_id
        completed_tokens = operation.completed_tokens
        hash_value = operation.hash_value

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            _,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]

        # All PP/TP ranks will get the same `min_completed_tokens`, because `completed_tokens`
        # and `pool_hits` in their operations are same.  No need to sync cross-rank here.
        if not self._check_hybrid_prefetch_result(
            req_id,
            operation,
            completed_tokens,
            hash_value,
            host_indices,
            last_host_node_id,
            anchor_lock_params,
            prefetch_key,
        ):
            # Hybrid all-or-nothing check failed; result already discarded.
            return

        if self.buffer_pipeline is not None:
            # No graft: release the rank-local tail beyond the synced usable
            # length, then park the bounce for admission-time consumption.
            return self.buffer_pipeline.stage_completed_prefetch(
                req_id, completed_tokens, hash_value
            )

        fetched_key = prefetch_key[:completed_tokens]
        insert_result = self.tree_core.insert_host(
            last_host_node_id,
            fetched_key,
            host_indices[:completed_tokens],
            hash_value[: completed_tokens // self.page_size],
        )

        # Apply the host-insert walk's actions before the transfer commit.
        self._apply_cache_actions(insert_result.cache_actions)

        if insert_result.host_insert_dropped:
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            loaded_from_storage = 0
        else:
            commit_actions: list[CacheAction | ComponentAction] = []
            self.tree_core.commit_hicache_transfers(
                last_host_node_id,
                CacheTransferPhase.PREFETCH,
                comp_xfers,
                cache_actions=commit_actions,
                insert_result=insert_result,
                pool_storage_result=operation.pool_storage_result,
            )
            self._apply_cache_actions(commit_actions)
            # The commit emits via commit_actions only; the walk's were applied above.
            assert not insert_result.cache_actions

            self.cache_controller.mem_pool_host.free(
                host_indices[: insert_result.prefix_len]
            )
            loaded_from_storage = completed_tokens - insert_result.prefix_len

        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d",
            "dropped" if insert_result.host_insert_dropped else "success",
            req_id,
            completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            self.cache_controller.prefetch_tokens_occupied,
        )
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)
        return

    def _check_hybrid_prefetch_result(
        self,
        req_id: str,
        operation: PrefetchOperation,
        completed_tokens: int,
        hash_value: list[str],
        host_indices: torch.Tensor,
        last_host_node_id: NodeId,
        anchor_lock_params: DecLockRefParams,
        prefetch_key: RadixKey,
    ) -> bool:
        """Decide the length of usable prefix.

        Two strategies depending on the hybrid layout:

        * DSA-style (Full attention + KV-derived ALL_PAGES sidecar such as the
          DSA / MiniMax indexer): *clamp* to the minimum fetched prefix shared by
          the Full KV pool and every sidecar. A partial prefix is still usable
          because the sidecar is page-aligned with KV and required for every page.
        * Everything else (SWA / Mamba components, mixed DeepSeekV4 stacks):
          *all-or-nothing*. Their pools only cover a window / tail and cannot be
          truncated page by page, so any shortfall discards the whole prefetch.

        Returns true if prefetch success, or false when an all-or-nothing prefetch
        was discarded (the caller should then treat the prefetch as finished).
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        #
        # Skip KV-derived pools, which do not report hits in operation.pool_storage_result.
        # Their hit lengths are stored in completed_tokens.
        pool_transfers = [
            transfer
            for transfer in operation.pool_transfers or []
            if transfer.indices_from_pool != PoolName.KV
        ]
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        completed_tokens = operation.completed_tokens
        # Hybrid cache state is all-or-nothing: every extra pool (SWA / Mamba / ...)
        # must cover the same fetched prefix. If any pool falls short the whole
        # prefetch result is unusable, so discard it and release everything.
        expected_tokens = len(hash_value) * self.page_size
        all_succeeded = completed_tokens == expected_tokens and all(
            transfer.keys is not None and count == len(transfer.keys)
            for transfer, count in zip(pool_transfers, pool_hit_pages)
        )
        if pool_transfers and not all_succeeded:
            # Drop the KV beliefs from the first page any pool failed to serve;
            # the next insert then re-writes that span through one FULL check,
            # restoring the missing aux pages.
            keep_pages = completed_tokens // self.page_size
            for transfer, count in zip(pool_transfers, pool_hit_pages):
                if transfer.keys is None:
                    keep_pages = 0
                elif count < len(transfer.keys):
                    # Aux transfers key the chain's trailing pages.
                    keep_pages = min(
                        keep_pages, max(0, len(hash_value) - len(transfer.keys))
                    )
            self.storage_existence_cache.invalidate_beyond(
                PoolName.KV, hash_value, keep_pages=keep_pages
            )
            # The controller's prefetch IO thread already releases the untransferred
            # tail (host_indices[completed_tokens:])
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=pool_transfers if operation.pool_transfers_done else None,
            )
            if anchor_lock_params is not None:
                self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            if self.buffer_pipeline is not None:
                self.buffer_pipeline.pop_prefix_ctx(req_id)
                self.buffer_pipeline.release_anchor_lock(req_id)
            del self.ongoing_prefetch[req_id]
            self.cache_controller.prefetch_tokens_occupied -= (
                self._prefetch_occupied_span(prefetch_key, host_indices)
            )
            self.prefetch_loaded_tokens_by_reqid[req_id] = 0
            logger.warning(
                "HiCache hybrid prefetch discarded req=%s completed=%d requested=%d "
                "kv_beliefs_kept_pages=%d",
                req_id,
                completed_tokens,
                expected_tokens,
                keep_pages,
            )
            return False
        return True

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        # The request is being scheduled; a still-unserved miss marker is moot.
        self._storage_prefetch_missed_rids.discard(req_id)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def pop_storage_prefetch_miss(self, req_id: str) -> bool:
        """True once per resolved storage-prefetch miss for a live request;
        the scheduler uses it to arm the paced availability-check retry."""
        if req_id in self._storage_prefetch_missed_rids:
            self._storage_prefetch_missed_rids.discard(req_id)
            return True
        return False

    def plan_staged_splice(
        self, req_id: str, device_prefix_len: int
    ) -> tuple[int, int]:
        """(kv, swa) host-hit tokens a staged buffer-mode prefetch will splice
        given the request's live device prefix; frees unusable holds."""
        if self.buffer_pipeline is None:
            return 0, 0
        return self.buffer_pipeline.plan_staged_splice(req_id, device_prefix_len)

    def staged_prefetch_swa_tokens(self, req_id: str) -> int:
        """SWA device tokens consuming a staged buffer-mode prefetch will
        allocate; surfaced as the request's swa_host_hit_length."""
        if self.buffer_pipeline is None:
            return 0
        return self.buffer_pipeline.staged_prefetch_swa_tokens(req_id)

    @rank_consensus(same_params=True)
    def release_aborted_request(self, rid: str) -> None:
        if self.linker is not None:
            self.linker.release_request(rid)
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._storage_prefetch_missed_rids.discard(rid)
        if (
            self.buffer_pipeline is not None
            and self.buffer_pipeline.release_staged_hold(rid)
        ):
            return
        if rid not in self.ongoing_prefetch:
            return

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            self.cache_controller.terminate_prefetch(operation)
            self.revoke_pending_prefetch(rid)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(rid)
            self.buffer_pipeline.release_anchor_lock(rid)
        pool_transfers = [x for xfers in comp_xfers.values() for x in xfers]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=pool_transfers if operation.pool_transfers_done else None,
        )
        # Buffer mode granted occupancy at hit-alloc, sized to the bounce;
        # cache mode reserved the requested span at enqueue.
        self.cache_controller.prefetch_tokens_occupied -= self._prefetch_occupied_span(
            prefetch_key, host_indices
        )

    def _invalidate_absent_from_hit_query(self, operation) -> None:
        """Drop KV beliefs beyond the folded usable cut (rank-synced): the
        next insert then re-writes the node (all pools), healing stale
        positives and aux holes at the cut through one FULL check."""
        if self.host_memory_mode != "buffer_only":
            return
        chain = operation.all_hash_values
        if chain is None:
            return
        self.storage_existence_cache.invalidate_beyond(
            PoolName.KV, chain, keep_pages=operation.storage_hit_count // self.page_size
        )

    def _account_prefetch_outcome(self, operation, revoked: bool) -> None:
        """Feed the cumulative prefetch-outcome counters at the (rank-synced)
        query outcome: T = prompt tokens, L = requested, m = L3-miss."""
        requested = operation.stats_requested_tokens
        if requested <= 0:
            return
        stats = self._prefetch_outcome_stats
        hit = max(0, min(operation.storage_hit_count, requested))
        if revoked:
            if hit > 0:
                stats["revoked_insufficient"] += 1
            else:
                stats["revoked_full_miss"] += 1
        miss = requested - hit
        total = max(operation.stats_total_tokens, requested, 1)
        stats["l3_demand_requests"] += 1
        stats["l1l2_miss_tokens"] += requested
        stats["l3_miss_tokens"] += miss
        stats["l3_demand_total_tokens"] += total
        stats["l3_sum_rate_all"] += miss / total
        stats["l3_sum_rate_main_weighted"] += (miss / requested) * total

    def prefetch_outcome_stats_snapshot(self) -> dict:
        """Cumulative counters + instantaneous occupancy, in the schema
        log_prefetch_stats consumers expect."""
        cc = self.cache_controller
        cap = max(cc.prefetch_capacity_limit, 1)
        return {
            **self._prefetch_outcome_stats,
            "occupancy_ratio": cc.prefetch_tokens_occupied / cap,
        }

    def _prefetch_occupied_span(self, prefetch_key, host_indices) -> int:
        """Occupancy units held by a prefetch: cache mode reserves the
        requested span at enqueue; buffer mode grants at hit-alloc, sized
        to the allocation (0 while still querying / parked)."""
        if self.host_memory_mode == "buffer_only":
            return len(host_indices) if host_indices is not None else 0
        return len(prefetch_key)

    def revoke_pending_prefetch(self, req_id: str) -> None:
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return
        (
            last_host_node_id,
            prefetch_key,
            _host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        self._invalidate_absent_from_hit_query(operation)
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.pop_prefix_ctx(req_id)
            self.buffer_pipeline.release_anchor_lock(req_id)
        cc = self.cache_controller
        cc.append_host_mem_release(
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
        )
        if anchor_lock_params is not None:
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        # Every revoke path runs before the bounce alloc, so buffer mode
        # holds no occupancy here; post-alloc aborts go through
        # release_aborted_request instead.
        assert _host_indices is None or self.host_memory_mode != "buffer_only"
        cc.prefetch_tokens_occupied = max(
            0,
            cc.prefetch_tokens_occupied
            - self._prefetch_occupied_span(prefetch_key, _host_indices),
        )

    def _drain_storage_control_queues_impl(
        self,
        n_storage_hit: Optional[int],
        n_ack_prefetch: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        extra_release_counts: Optional[dict[PoolName, int]],
        log_metrics: bool,
    ) -> None:
        cc = self.cache_controller

        def _drain_queue(q: Queue[T], n: Optional[int]) -> Iterator[T]:
            """If n is None, consume all items from the queue.
            Otherwise, consume n items from the queue.  Blocking if there are no enough n items.

            In TP, each rank consumes the a minimal number of items of all ranks.
            In PP, each rank consumes the exact number of items of PP0.  Refer to _pp_sync for more details.

            This prevents TP/PP divergence.
            """
            if n is None:
                while not q.empty():
                    item = q.get()
                    yield item
            else:
                for _ in range(n):
                    # Block when there are not enough elements.
                    # All TP/PP ranks must consume the same number of elements.
                    item = q.get()
                    yield item

        buffer_mode = self.host_memory_mode == "buffer_only"

        def _try_alloc_storage_hit(operation) -> bool:
            """Allocate the hit-sized bounce and launch the transfer.
            Returns False when staging pressure defers the allocation
            (buffer mode parks and retries; cache mode revokes)."""
            req_id = operation.request_id
            info = self.ongoing_prefetch.get(req_id)
            if info is None:
                return True  # aborted/cleaned; nothing to retry
            if operation.is_terminated():
                self.revoke_pending_prefetch(req_id)
                return True

            if buffer_mode and cc.prefetch_rate_limited():
                # Pool is load-saturated: hold the KNOWN hit until staged
                # prefetches ahead of us are consumed. The op stays in
                # ongoing_prefetch, so wait_complete keeps gating admission.
                return False
            if buffer_mode:
                # IO commit: pin before the bounce alloc so a cancel is a
                # plain revoke and a parked op keeps its pin; a fetch whose
                # splice base is gone is not worth its storage read.
                if self.buffer_pipeline.try_lock_anchor(req_id) == "anchor_lost":
                    self._prefetch_outcome_stats["declined_anchor_lost"] += 1
                    # Span still L3-resident: arm the paced retry to re-fetch
                    # from the shorter post-loss match.
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    return True
                if self.buffer_pipeline.staged_span_covered(
                    req_id, operation.storage_hit_count
                ):
                    # Live tree already covers the span: nothing left to
                    # splice, so skip the storage read.
                    self._prefetch_outcome_stats["declined_device_covered"] += 1
                    self.revoke_pending_prefetch(req_id)
                    return True
            alloc_len = operation.storage_hit_count
            host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                self.evict_host(alloc_len)
                host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None and not buffer_mode:
                # Memory-pressure fallback: a shorter page-aligned prefix.
                # (Cache mode only — buffer mode parks for the full hit.)
                available_size = cc.mem_pool_host.available_size()
                alloc_len = min(
                    operation.storage_hit_count,
                    available_size - (available_size % self.page_size),
                )
                if alloc_len >= self.prefetch_threshold:
                    host_indices = cc.mem_pool_host.alloc(alloc_len)
            if host_indices is None:
                if buffer_mode:
                    return False
                self.revoke_pending_prefetch(req_id)
                return True

            operation.storage_hit_count = alloc_len
            operation.hash_value = operation.hash_value[: alloc_len // self.page_size]
            operation.host_indices = host_indices
            self.ongoing_prefetch[req_id] = info._replace(host_indices=host_indices)
            if buffer_mode:
                cc.prefetch_tokens_occupied += alloc_len
            cc.prefetch_buffer.put(operation)
            return True

        def _drain_and_alloc_storage_hit():
            # Parked hits first (FIFO fairness with retries; buffer only).
            if buffer_mode:
                parked = self.buffer_pipeline.pending_hit_allocs
                while parked:
                    if not _try_alloc_storage_hit(parked[0]):
                        break
                    parked.popleft()
            for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
                req_id = operation.request_id
                info = self.ongoing_prefetch.get(req_id)
                if info is None:
                    # Request already aborted/cleaned up; still flush the
                    # query's absent-hash feedback.
                    self._invalidate_absent_from_hit_query(operation)
                    continue
                if operation.is_terminated():
                    # Controller-side miss termination (retryable) or an abort
                    # race (abort cleanup discards the marker).
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                if operation.storage_hit_count < self.prefetch_threshold:
                    # Below-threshold hit: classify + feed the L3 miss
                    # accounting, then revoke (not enough benefit).
                    self._account_prefetch_outcome(operation, revoked=True)
                    self._storage_prefetch_missed_rids.add(req_id)
                    self.revoke_pending_prefetch(req_id)
                    continue
                self._invalidate_absent_from_hit_query(operation)
                self._account_prefetch_outcome(operation, revoked=False)
                if not _try_alloc_storage_hit(operation):
                    # Counted once at first parking, not per retry tick.
                    self._prefetch_outcome_stats["declined_rate_limited"] += 1
                    self.buffer_pipeline.pending_hit_allocs.append(operation)

        def _drain_ack_prefetch():
            for ack in _drain_queue(cc.ack_prefetch_queue, n_ack_prefetch):
                operation = ack.operation
                if ack.completed_tokens is not None:
                    if operation.request_id in self.ongoing_prefetch:
                        assert operation.completed_tokens <= ack.completed_tokens
                        operation.completed_tokens = ack.completed_tokens
                if ack.pool_hits is not None:
                    if operation.request_id in self.ongoing_prefetch:
                        operation.pool_storage_result.update_extra_pool_hit_pages(
                            ack.pool_hits
                        )
                        operation.pool_transfers_done = True
                if ack.completed_req:
                    if operation.request_id in self.ongoing_prefetch:
                        # check_prefetch_progress() is not called for this rid yet.
                        # Let us insert the prefetch result into the radix tree.
                        self._handle_prefetch_result(operation)
                    cc.append_host_mem_release(
                        operation.host_indices[operation.completed_tokens :],
                        (
                            operation.pool_transfers
                            if not operation.pool_transfers_done
                            else None
                        ),
                    )

        def _drain_backup():
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
                if buffer_mode:
                    # Storage write acked: free the staging.
                    self.buffer_pipeline.finish_storage_write_ack(operation.id)
                else:
                    entry = self.ongoing_backup.pop(operation.id, None)
                    if entry is not None:
                        node_id, lock_params = entry
                        self.dec_host_lock_ref(node_id, lock_params)
                if (
                    log_metrics
                    and self.enable_storage_metrics
                    and self.storage_metrics_collector is not None
                ):
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )
            return drained

        def _drain_release():
            host_indices_list = []
            released_tokens = 0
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
                released_tokens += len(host_indices)
            if host_indices_list:
                cc.mem_pool_host.free(torch.cat(host_indices_list, dim=0))
            return len(host_indices_list), released_tokens

        def _drain_extra_release():
            drained: dict[PoolName, tuple[int, int]] = {}
            if not extra_release_counts:
                return drained
            for pool_name, limit in extra_release_counts.items():
                release_queue = cc.extra_host_mem_release_queues.get(pool_name)
                if release_queue is None:
                    continue
                host_indices_list = []
                released_tokens = 0
                for host_indices in _drain_queue(release_queue, limit):
                    host_indices_list.append(host_indices)
                    released_tokens += len(host_indices)
                if host_indices_list:
                    cc.mem_pool_host.free(
                        torch.cat(host_indices_list, dim=0), pool=pool_name
                    )
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_and_alloc_storage_hit()
        _drain_ack_prefetch()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

    def drain_storage_control_queues(self) -> None:
        cc = self.cache_controller
        extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
        extra_pool_names = list(extra_release_queues)
        local_qsize_list = [
            cc.prefetch_hit_queue.qsize(),
            cc.ack_prefetch_queue.qsize(),
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
        self._all_reduce(qsizes, torch.distributed.ReduceOp.MIN)
        qsize_list = list(map(int, qsizes.tolist()))
        n_storage_hit, n_ack_prefetch, n_backup, n_release = qsize_list[:4]
        extra_release_counts = {
            pool_name: count
            for pool_name, count in zip(extra_pool_names, qsize_list[4:])
        }
        self._drain_storage_control_queues_impl(
            n_storage_hit=n_storage_hit,
            n_ack_prefetch=n_ack_prefetch,
            n_backup=n_backup,
            n_release=n_release,
            extra_release_counts=extra_release_counts,
            log_metrics=True,
        )

    def drain_storage_control_queues_local(self) -> None:
        """Drain the storage control queues without cross-rank synchronization.

        For the detach / shutdown path, where best-effort cleanup matters more than
        keeping the drained counts identical across ranks. The prefetch-hit queue is
        deliberately skipped: servicing it would allocate host pages for a prefetch
        that can no longer complete.
        """
        cc = self.cache_controller
        # The storage queues are created by the controller when the storage threads
        # start, so they are still None when a backend was never attached.
        if cc is None or cc.prefetch_hit_queue is None:
            return
        self._drain_storage_control_queues_impl(
            n_storage_hit=0,
            n_ack_prefetch=0,
            n_backup=None,
            n_release=None,
            extra_release_counts={
                name: None for name in cc.extra_host_mem_release_queues
            },
            log_metrics=False,
        )

    # ---- HiCache: Storage backend lifecycle (delegated) ----

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return (
                False,
                "HiCache is not initialized; launch with "
                "--enable-hierarchical-cache to attach a storage backend.",
            )
        return self._storage_attachment.attach(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy=hicache_write_policy,
        )

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) the HiCache storage backend at runtime."""
        if self._storage_attachment is None:
            return False, "HiCache storage backend is not initialized."
        return self._storage_attachment.detach()

    def shutdown(self) -> None:
        """Best-effort auto-detach of the storage backend on process shutdown."""
        if self._storage_attachment is not None:
            self._storage_attachment.shutdown()

    def clear_storage_backend(self) -> bool:
        if self._storage_attachment is None:
            return False
        ok = self._storage_attachment.clear()
        if ok:
            # L3 is empty now: every storage-presence belief is stale, and a
            # retained positive would skip that page's backup forever.
            self.storage_existence_cache.clear()
        return ok

    # ---- HiCache: Async Event Management ----

    def _count_ready_acks(self, ack_queue) -> int:
        ready_count = 0
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready_count += 1
        return ready_count

    def _sync_hicache_ready_counts(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[PoolName, ...]]:
        cc = self.cache_controller
        if cc is None:
            write_acks = 0
            load_acks = 0
            storage_queue_sizes = ()
            extra_pool_names = ()
        else:
            write_acks = self._count_ready_acks(cc.ack_write_queue)
            load_acks = self._count_ready_acks(cc.ack_load_queue)
            extra_release_queues = getattr(cc, "extra_host_mem_release_queues", {})
            extra_pool_names = (
                tuple(extra_release_queues) if self.enable_storage else ()
            )
            storage_queue_sizes = (
                (
                    cc.prefetch_hit_queue.qsize(),
                    cc.ack_prefetch_queue.qsize(),
                    cc.ack_backup_queue.qsize(),
                    cc.host_mem_release_queue.qsize(),
                    *(extra_release_queues[name].qsize() for name in extra_pool_names),
                )
                if self.enable_storage
                else ()
            )

        # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
        # equal iff reclaim victim order matched on every rank.
        digest = self.tree_core.write_back_duplicate_reclaim_digest
        ready_counts = torch.tensor(
            [
                write_acks,
                load_acks,
                *storage_queue_sizes,
                digest,
                -digest,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        self._all_reduce(ready_counts, torch.distributed.ReduceOp.MIN)

        count_values = list(map(int, ready_counts.tolist()))
        assert count_values[-2] == -count_values[-1], (
            "write_back duplicate-reclaim victims diverged across TP ranks"
        )
        return (
            count_values[0],
            count_values[1],
            tuple(count_values[2:-2]),
            extra_pool_names,
        )

    def writing_check(
        self, write_back: bool = False, finish_count: Optional[int] = None
    ) -> None:
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
                    self._log_write_ack_metrics(ack)
                cc.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_write_through can
            # diverge across ranks (e.g. write_backup returning 0 on a subset).
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_write_queue)
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        # Process completed acks
        while finish_count > 0:
            ack = cc.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id)
            self._log_write_ack_metrics(ack)
            finish_count -= 1

    def _log_write_ack_metrics(self, ack: HiCacheAck) -> None:
        """Record D->H backup volume and duration for a completed write ack."""
        if self.metrics_collector is None:
            return
        for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
            if num_tokens > 0:
                self.metrics_collector.increment_backup_num_tokens(
                    num_tokens=num_tokens, pool=pool
                )
        if ack.num_bytes > 0:
            self.metrics_collector.increment_backup_num_bytes(ack.num_bytes)
        if ack.timing_enabled:
            duration_ms = ack.start_event.elapsed_time(ack.finish_event)
            self.metrics_collector.observe_backup_duration(duration_ms / 1000.0)

    def loading_check(self, finish_count: Optional[int] = None) -> None:
        """Poll load-back completions."""
        cc = self.cache_controller
        if cc is None:
            return
        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_load_back can
            # diverge across ranks.
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(cc.ack_load_queue)
            # Piggybacked TP check: [digest, -digest] MIN-reduces to [min, -max],
            # equal iff reclaim victim order matched on every rank.
            digest = self.tree_core.write_back_duplicate_reclaim_digest
            sync_tensor = torch.tensor(
                [finish_count, digest, -digest], dtype=torch.int64, device="cpu"
            )
            self._all_reduce(sync_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = int(sync_tensor[0].item())
            assert sync_tensor[1].item() == -sync_tensor[2].item(), (
                "write_back duplicate-reclaim victims diverged across TP ranks"
            )

        while finish_count > 0:
            ack = cc.ack_load_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                if (
                    self.buffer_pipeline is not None
                    and self.buffer_pipeline.try_finish_load_back(ack_id)
                ):
                    continue
                node, lock_params, host_lock_params = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(node, lock_params)
                self.dec_host_lock_ref(node, host_lock_params)
                # Unpin the loaded nodes; host copies stay as reclaimable duplicates.
                self.tree_core.finish_load_back(node)

            if self.metrics_collector is not None:
                for pool, num_tokens in (ack.num_tokens_by_pool or {}).items():
                    if num_tokens > 0:
                        self.metrics_collector.increment_load_back_num_tokens(
                            num_tokens=num_tokens, pool=pool
                        )
                if ack.num_bytes > 0:
                    self.metrics_collector.increment_load_back_num_bytes(ack.num_bytes)
                if ack.timing_enabled:
                    duration_ms = ack.start_event.elapsed_time(ack.finish_event)
                    self.metrics_collector.observe_load_back_duration(
                        duration_ms / 1000.0
                    )
            finish_count -= 1

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, NodeId]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node). Buffer mode dispatches to the
        staged-prefetch consumption (BufferModePipeline.init_load_back)."""
        if self.buffer_pipeline is not None:
            return self.buffer_pipeline.init_load_back(params)
        best_match_node_id = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
        if self.linker is not None and self.linker.has_hit(req.rid):
            return self.linker.load_back(req)
        last_best_match_device_node_id = req.last_node

        if (
            self.tree_core.is_full_device_evicted(best_match_node_id)
            or params.host_hit_length > 0
            or (
                req is not None
                and (req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0)
            )
        ):
            if self.load_back(best_match_node_id, mem_quota, req=req):
                new_indices = self.tree_core.collect_full_device_indices(
                    best_match_node_id, last_best_match_device_node_id
                )
                if new_indices.numel() == 0:
                    return (
                        self.tree_core.empty_match_result.device_indices,
                        last_best_match_device_node_id,
                    )

                logger.debug(
                    "init_load_back success: loaded %d tokens for node %d",
                    len(new_indices),
                    best_match_node_id,
                )
                return new_indices, best_match_node_id

        return (
            self.tree_core.empty_match_result.device_indices,
            last_best_match_device_node_id,
        )

    def check_hicache_events(self) -> None:
        """Called per scheduler step to poll async HiCache events."""
        if self.linker is not None:
            finish_counts = torch.tensor(
                [
                    self.linker.num_completed_loads(),
                    self.linker.num_completed_offloads(),
                ],
                dtype=torch.int,
                device="cpu",
            )
            self._all_reduce_attn_groups(finish_counts, torch.distributed.ReduceOp.MIN)
            load_count, offload_count = map(int, finish_counts.tolist())
            self.linker.drain_loads(load_count)
            local_successes = self.linker.take_completed_offloads(offload_count)
            if local_successes:
                successes = torch.tensor(local_successes, dtype=torch.int, device="cpu")
                self._all_reduce_attn_groups(successes, torch.distributed.ReduceOp.MIN)
                self.linker.commit_completed_offloads(
                    [bool(success) for success in successes.tolist()]
                )
            return

        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()

        if self.pp_size != 1:
            finish_counts = torch.zeros(2, dtype=torch.int, device="cpu")
            if self.pp_rank == 0 and self.cache_controller is not None:
                finish_counts[0] = self._count_ready_acks(
                    self.cache_controller.ack_write_queue
                )
                finish_counts[1] = self._count_ready_acks(
                    self.cache_controller.ack_load_queue
                )
            self._all_reduce(finish_counts, torch.distributed.ReduceOp.MIN)
            write_finish_count, load_finish_count = map(int, finish_counts.tolist())
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)
            if self.enable_storage:
                self.drain_storage_control_queues()
        else:
            (
                write_finish_count,
                load_finish_count,
                storage_queue_sizes,
                extra_pool_names,
            ) = self._sync_hicache_ready_counts()
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)

            if self.enable_storage and storage_queue_sizes:
                n_storage_hit, n_ack_prefetch, n_backup, n_release = (
                    storage_queue_sizes[:4]
                )
                extra_release_counts = {
                    pool_name: count
                    for pool_name, count in zip(
                        extra_pool_names,
                        storage_queue_sizes[4:],
                    )
                }
                self._drain_storage_control_queues_impl(
                    n_storage_hit=n_storage_hit,
                    n_ack_prefetch=n_ack_prefetch,
                    n_backup=n_backup,
                    n_release=n_release,
                    extra_release_counts=extra_release_counts,
                    log_metrics=True,
                )
        if self.buffer_pipeline is not None:
            self.buffer_pipeline.flush_pending_writes()
        if self.enable_storage_metrics and self.storage_metrics_collector is not None:
            storage_metrics = self.cache_controller.storage_backend.get_stats()
            if storage_metrics is None:
                # Backends without native stats (e.g. file) still carry the
                # controller-side prefetch outcome counters.
                storage_metrics = StorageMetrics()
            storage_metrics.prefetch_stats = self.prefetch_outcome_stats_snapshot()
            self.storage_metrics_collector.log_storage_metrics(storage_metrics)

    def ready_to_load_host_cache(self) -> int:
        """Notify the cache controller to start the KV cache loading."""
        if self.linker is not None:
            return self.linker.start_layer_wise_loading()
        if self.cache_controller is not None:
            return self.cache_controller.start_loading()
        return 0

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after the local load-back event is complete.

        Mirrors ``HiRadixCache`` so the disagg decode restore state machine
        (``DecodeHiCacheTransferMixin``) can gate on load-back completion; the
        controller-level ``layer_done_counter`` event is shared across cache
        implementations, while the tree-side bookkeeping runs in
        ``loading_check``.
        """
        if consumer_index < 0 or self.cache_controller is None:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    # ---- Query / Inspection APIs ----
    # These APIs exist for compatibility with other RadixTree implementations.
    # TODO: simplify and consolidate in a future refactor.

    @property
    def sliding_window_size(self):
        return self._sliding_window_size

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
            and not self.tree_core.has_swa_host_pool
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def swa_retain_floor(self, req) -> int | None:
        if not self.is_mamba_enabled or self._sliding_window_size is None:
            return None
        checkpoint = req.kv.mamba_last_track_seqlen
        if checkpoint is None:
            return None
        return checkpoint - self._sliding_window_size

    def supports_swa(self) -> bool:
        return self.is_swa_enabled

    def supports_mamba(self) -> bool:
        return self.is_mamba_enabled

    # ---- Session radix cache API (delegates to composed UnifiedSessionRefTracker) ----

    def open_radix_session(self, session_id: str) -> Optional[int]:
        return self.session_refs.open_radix_session(session_id)

    def ensure_session_generation(self, session_id: str) -> int:
        return self.session_refs.ensure_session_generation(session_id)

    def release_radix_session(self, session_id: str) -> int:
        return self.session_refs.release_radix_session(session_id)

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
        return self.tree_core.evictable_size()

    def protected_size(self) -> int:
        return self.tree_core.protected_size()

    def full_evictable_size(self) -> int:
        return self.tree_core.full_evictable_size()

    def full_protected_size(self) -> int:
        return self.tree_core.full_protected_size()

    def swa_evictable_size(self) -> int:
        return self.tree_core.swa_evictable_size()

    def mamba_evictable_size(self) -> int:
        return self.tree_core.mamba_evictable_size()

    def swa_protected_size(self) -> int:
        return self.tree_core.swa_protected_size()

    def mamba_protected_size(self) -> int:
        return self.tree_core.mamba_protected_size()

    def total_size(self) -> tuple[int, int]:
        return self.tree_core.total_size()

    def all_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_values_flatten()

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self.tree_core.all_mamba_values_flatten()

    def available_and_evictable_str(self) -> str:
        # TODO(zhangmj): need more detailed log info for session reference.
        if self.supports_swa():
            full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        else:
            full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable = self.tree_core.component_evictable_size(BASE_COMPONENT_TYPE)
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
                f"Available {ct}: {available_size + self.tree_core.component_evictable_size(ct)} "
                f"(available_size={available_size} + component_evictable_size_={self.tree_core.component_evictable_size(ct)})"
            )
        return "\n".join(lines) + "\n"

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

        # Pass ongoing ops as lightweight (id, node_id) pairs so the tree core
        # can resolve + validate them without reaching into Controller state.
        if self.buffer_pipeline is not None:
            ongoing_write_through = [
                (nid, entry.intent.node_id)
                for nid, entry in self.buffer_pipeline.ongoing_write_through.items()
            ]
        else:
            ongoing_write_through = [
                (nid, wt.node_id) for nid, wt in self.ongoing_write_through.items()
            ]
        ongoing_load_back = [
            (nid, lb.node_id) for nid, lb in self.ongoing_load_back.items()
        ]
        self.tree_core.sanity_check(ongoing_write_through, ongoing_load_back)

    def pretty_print(self) -> None:
        self.tree_core.pretty_print()

    # ---- TreeCore state delegation ----
    # The facade re-exposes tree-owned config (page_size, enable_storage, ...) so its
    # own coordination methods and external callers read them off the cache.

    # ``page_size`` keeps a setter: StreamingSession forwards assignment onto its
    # inner cache (the PrefixCacheTrait surface).
    @property
    def page_size(self):
        return self.tree_core.page_size

    @page_size.setter
    def page_size(self, value) -> None:
        self.tree_core.page_size = value

    @property
    def enable_storage(self):
        return self.tree_core.enable_storage

    @enable_storage.setter
    def enable_storage(self, value) -> None:
        self.tree_core.enable_storage = value

    @property
    def write_through_threshold(self):
        return self.tree_core.write_through_threshold

    @write_through_threshold.setter
    def write_through_threshold(self, value) -> None:
        self.tree_core.write_through_threshold = value

    @property
    def is_write_back(self):
        return self.tree_core.is_write_back

    @is_write_back.setter
    def is_write_back(self, value) -> None:
        self.tree_core.is_write_back = value

    @property
    def device(self):
        return self.tree_core.device

    @property
    def root_node(self):
        return self.tree_core.root_node

    def take_events(self):
        # Drain the KV event queue from the TreeCore.
        return self.tree_core.take_events()

    def resolve_node_handle(self, node_handle):
        """Look up the node object from its NodeId.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        if isinstance(node_handle, int):
            return self.tree_core.node_by_id(node_handle)
        # Internal callers (and the session sentinel / None) pass a non-int through.
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        """The root's NodeId -- URC match results carry NodeIds."""
        return self.tree_core.root_node_handle(extra_key)

    def dfs_weight_order(self, node_handles: Sequence[NodeId]) -> list[int]:
        return self.tree_core.dfs_weight_order(node_handles)
