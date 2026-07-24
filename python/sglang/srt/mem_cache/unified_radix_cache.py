from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from queue import Empty, Queue
from typing import TYPE_CHECKING, Iterator, NamedTuple, Optional, TypeVar

import torch

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
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
    CacheAction,
    ComponentAction,
    FreeComponentDeviceSlot,
    FreeDeviceKV,
    ReplaceWriteThroughOnNodeSplit,
)

# UnifiedTreeNode / UnifiedLRUList live on the tree core; re-exported here
# because other modules and tests import them from this module.
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (  # noqa: F401
    NodeId,
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    CacheTransferPhase,
    ComponentType,
    FullComponent,
    MambaComponent,
    PrepareLoadBackResult,
    SWAComponent,
    TreeComponent,
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
        # TODO(Jialin): make the TreeCore configurable so an alternative
        # implementation (e.g. a Rust TreeCore) can be swapped in.
        self.tree_core = UnifiedTreeCore(params, self.components)

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
        self.tree_core.reset()

        # Reset Controller.
        self.session.slots.clear()
        self.ongoing_write_through: dict[int, _OngoingWriteThrough] = {}
        self.ongoing_load_back: dict[int, _OngoingLoadBack] = {}
        self.enable_storage = False
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.ongoing_prefetch: dict[str, _OngoingPrefetch] = {}
        self.ongoing_backup: dict[int, tuple[NodeId, DecLockRefParams]] = {}

        if self.cache_controller is not None:
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()
            self.enable_storage = self.cache_controller.enable_storage

        self.tree_core._record_all_cleared_event()

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
        # Tag HiCache enablement on the TreeCore.
        if self.cache_controller is not None:
            self.tree_core.set_hicache_enabled()

        # State initialization
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.is_write_back = (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
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

    def release_host_resources(self) -> None:
        if self.host_pool_group is not None:
            self.host_pool_group.destroy()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        if self.disable:
            return self.tree_core.empty_match_result
        result = self.tree_core.match_prefix(params)
        for component in self._components_tuple:
            result = component.finalize_match_result_in_cache(params, result)
        self._apply_cache_actions(result.cache_actions)
        return result

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)
        result = self.tree_core.insert(params)
        self._apply_cache_actions(result.cache_actions)
        return result

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}
        device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)

        request_by_type = {
            ComponentType.FULL: params.num_tokens,
            ComponentType.SWA: params.swa_num_tokens,
            ComponentType.MAMBA: params.mamba_num,
        }
        try:
            self._evict_components(request_by_type, tracker, device_frees, host_frees)
        finally:
            # Drain even on a mid-walk raise: tombstoned slots must reach the
            # allocator or they leak.
            self._drain_frees(device_frees, host_frees)

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

    def _evict_components(
        self,
        request_by_type: dict[ComponentType, int],
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        for ct in self.tree_components:
            request_cnt = request_by_type[ct]
            # Skip eviction walk if request is already met
            if tracker[ct] >= request_cnt:
                continue
            self.tree_core.evict_device_start(ct, request_cnt)
            try:
                while (
                    node_id := self.tree_core.evict_device_next_node(
                        ct, tracker, device_frees, host_frees
                    )
                ) is not None:
                    backup_kv = self.tree_core.evict_device_leaf(
                        node_id, tracker, device_frees, host_frees, self.is_write_back
                    )
                    if backup_kv is not None:
                        # Deferred demote: run the D->H backup, demote only on success.
                        written = self._execute_and_commit_kv_backup(
                            backup_kv, write_back=True
                        )
                        if written > 0:
                            self.writing_check(write_back=True)
                            self.tree_core.demote(
                                node_id,
                                tracker,
                                device_frees=device_frees,
                                host_frees=host_frees,
                            )
            finally:
                self.tree_core.evict_device_end(ct)

    def inc_lock_ref(self, node_id: NodeId) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node_id)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        return self.tree_core.inc_lock_ref(node_id)

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

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int] = None,
    ) -> None:
        if self.disable:
            return
        device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        try:
            self.tree_core.dec_swa_lock_only(
                node_id, swa_uuid_for_lock, device_frees, host_frees
            )
        finally:
            self._drain_frees(device_frees, host_frees)

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
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            for comp in self._components_tuple:
                comp.cleanup_after_caching_req(req, is_finished=True)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        result = None
        insert_params = None

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
                token_ids, req.extra_key, is_bigram=self.tree_core.is_eagle
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
            is_bigram=self.tree_core.is_eagle,
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

    def _apply_cache_actions(
        self, actions: list[CacheAction | ComponentAction]
    ) -> None:
        for action in actions:
            self._apply_cache_action(action)

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
        elif isinstance(action, FreeDeviceKV):
            for indices in action.indices:
                self.token_to_kv_pool_allocator.free(indices)
        elif isinstance(action, BackupKV):
            self._execute_and_commit_kv_backup(action)
        else:
            raise AssertionError(f"unhandled CacheAction: {type(action).__name__}")

    def _drain_device_frees(
        self, device_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component device slots.
        for ct, indices in device_frees.items():
            self._apply_cache_action(
                FreeComponentDeviceSlot(indices, component_type=ct)
            )

    def _drain_host_frees(
        self, host_frees: dict[ComponentType, list[torch.Tensor]]
    ) -> None:
        # Free per component host-pool slots collected during eviction walks.
        for ct, host_values in host_frees.items():
            self.components[ct].free_host_values(host_values)

    def _drain_frees(
        self,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        # Both drains must run even if one raises.
        try:
            self._drain_device_frees(device_frees)
        finally:
            self._drain_host_frees(host_frees)

    def evict_host(
        self, num_tokens: int, component_type: ComponentType = BASE_COMPONENT_TYPE
    ) -> int:
        """Evict host resources for a specific component to free host pool space."""
        tracker: dict[ComponentType, int] = {ct: 0 for ct in self.tree_components}
        device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
        try:
            self.tree_core.drive_host_eviction(
                component_type, num_tokens, tracker, device_frees, host_frees
            )
        finally:
            self._drain_frees(device_frees, host_frees)
        return tracker[component_type]

    # ---- HiCache: Backup / LoadBack ----

    def _execute_and_commit_kv_backup(
        self, action: BackupKV, write_back: bool = False
    ) -> int:
        """Run a backup action top-down, stopping at the first failed backup; a
        failure is a deterministic host-space shortfall, so no intra-drain retry."""
        written = 0
        for node_id in action.node_ids:
            # Overlapping chain actions: skip already-backed nodes.
            if self.tree_core.is_backuped(node_id):
                continue
            device_value, comp_xfers = self.tree_core.build_backup_spec(node_id)
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
        # when the Full-KV load is skipped by thresholding.
        if (kv_tokens < self.load_back_threshold and not comp_xfers) or (
            mem_quota is not None and kv_tokens > mem_quota + result.delta
        ):
            self.dec_lock_ref(node_id, ancestor_lock_params)
            self.dec_host_lock_ref(node_id, host_anchor_params)
            return False

        if self.supports_swa():
            avail = self.token_to_kv_pool_allocator.full_available_size()
        else:
            avail = self.token_to_kv_pool_allocator.available_size()
        if avail < kv_tokens:
            needed = kv_tokens - avail
            result = self.evict(EvictParams(num_tokens=needed))
            if result.num_tokens_evicted < needed:
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

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
    ) -> None:
        if not self.enable_storage or self.cache_controller is None:
            return

        extra_key = self.tree_core.prefetch_anchor_info(last_host_node_id)
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
        ).page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        if (
            prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        anchor_lock_params = self.inc_host_lock_ref(last_host_node_id).to_dec_params()
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
            self.cache_controller.append_host_mem_release(
                extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
            )
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
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
        self.ongoing_prefetch[req_id] = _OngoingPrefetch(
            last_host_node_id,
            prefetch_key,
            None,
            operation,
            anchor_lock_params,
            comp_xfers,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation) -> bool:
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

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

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            return True

        (
            last_host_node_id,
            prefetch_key,
            host_indices,
            operation,
            anchor_lock_params,
            comp_xfers,
        ) = self.ongoing_prefetch[req_id]
        if not self.can_terminate_prefetch(operation):
            return False
        if operation.host_indices is None:
            self.cache_controller.terminate_prefetch(operation)
            self._revoke_pending_prefetch(req_id)
            return True

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )

        min_completed_tokens = self._sync_and_check_hybrid_prefetch_result(
            req_id,
            operation,
            completed_tokens,
            hash_value,
            host_indices,
            last_host_node_id,
            anchor_lock_params,
            prefetch_key,
        )
        if min_completed_tokens is None:
            # Hybrid all-or-nothing check failed; result already discarded.
            return True

        fetched_key = prefetch_key[:min_completed_tokens]
        insert_result = self.tree_core.insert_host(
            last_host_node_id,
            fetched_key,
            host_indices[:min_completed_tokens],
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.tree_core.commit_hicache_transfers(
            last_host_node_id,
            CacheTransferPhase.PREFETCH,
            comp_xfers,
            cache_actions=insert_result.cache_actions,
            insert_result=insert_result,
            pool_storage_result=operation.pool_storage_result,
        )

        self._apply_cache_actions(insert_result.cache_actions)

        self.cache_controller.mem_pool_host.free(
            host_indices[: insert_result.prefix_len]
        )
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        loaded_from_storage = min_completed_tokens - insert_result.prefix_len
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage
        logger.info(
            "HiCache prefetch success req=%s completed_local=%d completed_synced=%d matched=%d loaded=%d tail_release=%d occupied=%d",
            req_id,
            completed_tokens,
            min_completed_tokens,
            insert_result.prefix_len,
            loaded_from_storage,
            completed_tokens - min_completed_tokens,
            self.cache_controller.prefetch_tokens_occupied,
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
        host_indices: torch.Tensor,
        last_host_node_id: NodeId,
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
            # The controller's prefetch IO thread already releases the untransferred
            # tail (host_indices[completed_tokens:])
            self.cache_controller.append_host_mem_release(
                host_indices=host_indices[:completed_tokens],
                extra_pools=pool_transfers,
            )
            self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
            del self.ongoing_prefetch[req_id]
            self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
            self.prefetch_loaded_tokens_by_reqid[req_id] = 0
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
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
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
            self._revoke_pending_prefetch(rid)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            host_indices=host_indices[:completed_tokens],
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers],
        )
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

    def _revoke_pending_prefetch(self, req_id: str) -> None:
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return
        (
            last_host_node_id,
            prefetch_key,
            _host_indices,
            _operation,
            anchor_lock_params,
            comp_xfers,
        ) = info
        cc = self.cache_controller
        cc.append_host_mem_release(
            extra_pools=[x for xfers in comp_xfers.values() for x in xfers]
        )
        self.dec_host_lock_ref(last_host_node_id, anchor_lock_params)
        cc.prefetch_tokens_occupied = max(
            0, cc.prefetch_tokens_occupied - len(prefetch_key)
        )

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

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                self._revoke_pending_prefetch(req_id)

        def _drain_and_alloc_storage_hit():
            for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
                req_id = operation.request_id
                info = self.ongoing_prefetch.get(req_id)
                if info is None:
                    # request already aborted/cleaned up, skip
                    continue
                if operation.is_terminated():
                    # request was aborted while the storage query was in flight
                    self._revoke_pending_prefetch(req_id)
                    continue

                alloc_len = operation.storage_hit_count
                host_indices = cc.mem_pool_host.alloc(alloc_len)
                if host_indices is None:
                    self.evict_host(alloc_len)
                    host_indices = cc.mem_pool_host.alloc(alloc_len)
                if host_indices is None:
                    # Memory-pressure fallback: a shorter page-aligned prefix.
                    available_size = cc.mem_pool_host.available_size()
                    alloc_len = min(
                        operation.storage_hit_count,
                        available_size - (available_size % self.page_size),
                    )
                    if alloc_len >= self.prefetch_threshold:
                        host_indices = cc.mem_pool_host.alloc(alloc_len)
                if host_indices is None:
                    self._revoke_pending_prefetch(req_id)
                    continue

                operation.storage_hit_count = alloc_len
                operation.hash_value = operation.hash_value[
                    : alloc_len // self.page_size
                ]
                operation.host_indices = host_indices
                self.ongoing_prefetch[req_id] = info._replace(host_indices=host_indices)
                cc.prefetch_buffer.put(operation)

        def _drain_backup():
            drained = 0
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                drained += 1
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
                    entry = cc.mem_pool_host.entry_map.get(pool_name)
                    if entry is not None:
                        entry.host_pool.free(torch.cat(host_indices_list, dim=0))
                drained[pool_name] = (len(host_indices_list), released_tokens)
            return drained

        _drain_revoke()
        _drain_and_alloc_storage_hit()
        _drain_backup()
        _drain_release()
        _drain_extra_release()

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
        # diverge across ranks (e.g. a backup returning 0 on a subset).
        finish_count = 0
        if self.pp_rank == 0:
            for ack in cc.ack_write_queue:
                if not ack.finish_event.query():
                    break
                finish_count += 1

        finish_count_tensor = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
        finish_count = finish_count_tensor.item()

        # Process completed acks
        while finish_count > 0:
            ack = cc.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id)
            finish_count -= 1

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

    # ---- HiCache: Scheduler Entry Points ----

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> tuple[torch.Tensor, NodeId]:
        """Prepare KV cache loading from host to device.
        Returns (device_indices, last_node) tuple."""
        best_match_node_id = params.best_match_node
        mem_quota = params.mem_quota
        req = params.req
        assert req is not None
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
            and swa._swa_kv_pool_host is None
        )
        return swa.sliding_window_size if unified_compress_only_hicache else 0

    def supports_swa(self) -> bool:
        return self.is_swa_enabled

    def supports_mamba(self) -> bool:
        return self.is_mamba_enabled

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

    def root_node_handle(self) -> NodeId:
        """The root's NodeId -- URC match results carry NodeIds."""
        return self.tree_core.root_node.id
