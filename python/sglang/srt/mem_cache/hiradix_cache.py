from __future__ import annotations

import atexit
import heapq
import itertools
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.base_prefix_cache import (
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
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    attach_hybrid_dsa_pool_to_hiradix_cache,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MiniMaxSparseKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host.common import get_allocator_type
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_INT63_MASK = 0x7FFFFFFFFFFFFFFF
# Backup acks synced per drain round when mirror release is on.  For MLA
# models only tp_rank 0 writes L3 (cache_controller.backup_skip) and every
# other rank acks with completed_tokens=0, so durable state must come from
# the writer's completed count, broadcast through the drain collective.
_MIRROR_ACK_SYNC_SLOTS = 8
# Sentinel a non-writer contributes so the MIN reduce recovers the smallest
# completed count among actual writers.
_MIRROR_ACK_NO_WRITER = 1 << 40


def _fnv1a64(data: bytes, h: int = _FNV64_OFFSET) -> int:
    for b in data:
        h ^= b
        h = (h * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass
class MirrorReleasePlan:
    """A release plan prepared one drain round ahead of execution.

    Mutation only happens after every rank reports the identical
    (count, tokens, digest) triple AND revalidates the plan — see
    drain_storage_control_queues().
    """

    nodes: List[TreeNode] = field(default_factory=list)
    tokens: int = 0
    digest: int = 0


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()

        allocator_type = get_allocator_type(server_args)

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = get_mha_host_pool_cls(self.kv_cache)(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
            )
        elif isinstance(self.kv_cache, DSATokenToKVPool):
            # Filled by attach_hybrid_dsa_pool_to_hiradix_cache after storage extra_config is parsed.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            # Filled by attach_hybrid_minimax_sparse_pool_to_hiradix_cache.
            self.token_to_kv_pool_host = None
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            from sglang.srt.runtime_context import get_parallel

            _parallel = get_parallel()
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
                dcp_size=_parallel.attn_dcp_size,
                dcp_rank=_parallel.attn_dcp_rank,
            )
        else:
            raise ValueError("HiRadixCache only supports MHA, MLA, DSA, and MSA models")

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.pp_group = params.pp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels

        # --- L2-as-channel: redundant host mirror release ---------------------
        # Under write_through every inserted node keeps a host copy that
        # evict_host() can never reclaim while the node stays device-resident,
        # so the host pool fills with mirrors of device data and blocks the
        # only staging path into L3.  When enabled, mirrors whose pages are all
        # durably backed to L3 become reclaimable through a TP-consistent
        # prepare/commit protocol in drain_storage_control_queues().
        self.mirror_release_enabled = self.enable_storage and os.getenv(
            "SGLANG_HICACHE_MIRROR_RELEASE", "0"
        ).lower() in ("1", "true")
        self.mirror_release_free_frac = float(
            os.getenv("SGLANG_HICACHE_MIRROR_RELEASE_FREE_FRAC", "0.2")
        )
        # After this many drain rounds a durable node may be re-staged so a
        # copy evicted by the L3 backend itself can be refreshed (0 = never).
        self.mirror_refresh_rounds = int(
            os.getenv("SGLANG_HICACHE_MIRROR_REFRESH_ROUNDS", "100000")
        )
        self.storage_backed_capacity = int(
            os.getenv("SGLANG_HICACHE_DURABLE_CAPACITY", "262144")
        )
        # page hash -> logical clock at (re-)mark; bounded LRU, overflow is a
        # conservative false negative (mirror stays, node may re-stage).
        self.storage_backed_hashes: OrderedDict[str, int] = OrderedDict()
        self.storage_backed_generation = 0
        self.hicache_logical_clock = 0
        # device-resident nodes whose host copy is redundant (device + L3 both
        # hold the data); membership is state-derived only — transient guards
        # (lock_ref / host_ref_counter) are checked at plan time, so busy nodes
        # stay members instead of being lost.
        self.redundant_host_nodes: set = set()
        self._mirror_release_plan: Optional[MirrorReleasePlan] = None
        self._mirror_release_disabled_reason: Optional[str] = None
        self._mirror_release_mismatch_streak = 0
        self._mirror_released_tokens_total = 0
        self._mirror_release_plans_executed = 0
        self._mirror_release_plans_dropped = 0

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if isinstance(self.kv_cache, DSATokenToKVPool):
            attach_hybrid_dsa_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
            )
        elif isinstance(self.kv_cache, MiniMaxSparseKVPool):
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                attach_hybrid_minimax_sparse_pool_to_hiradix_cache,
            )

            attach_hybrid_minimax_sparse_pool_to_hiradix_cache(
                self,
                params,
                server_args,
                extra_config=extra_config,
                prefetch_threshold=prefetch_threshold,
                enable_storage_metrics=self.enable_storage_metrics,
                load_cache_event=self.load_cache_event,
            )
        else:
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                pp_group=self.pp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.work_list: List[torch.distributed.Work] = []
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10
        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

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

        The following diagram illustrates the behavior of _pp_sync.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts | _pp_sync(data=?) starts | _pp_sync(data=?) starts
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=1) ends   |
        3     |                         |                         | _pp_sync(data=1) ends

        _pp_sync requires no synchronization point among ranks. The following case may also happen.

        time  | pp0                     | pp1                     | pp2
        ------|-------------------------|-------------------------|-----------------------------
        0     | _pp_sync(data=1) starts |                         |
        1     | _pp_sync(data=1) ends   |                         |
        2     |                         | _pp_sync(data=?) starts |
        3     |                         | _pp_sync(data=1) ends   |
        4     |                         |                         | _pp_sync(data=?) starts
        5     |                         |                         | _pp_sync(data=1) ends
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
            # Make a copy of data, so that the caller is safe to modify `data` after this call.
            # This is cheap, as _pp_sync is not to be used for transmitting large data.
            copy_of_data = data.clone()
            send_work = torch.distributed.isend(
                copy_of_data,
                group_dst=self.pp_rank + 1,
                group=self.pp_group,
                tag=P2PTag.HIRADIX_PP_SYNC,
            )
            self.work_list.append(send_work)

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_config: PrefetchTimeoutConfig,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        self.enable_storage = enable_storage
        # Recompute on runtime attach/detach; a fail-close (see
        # _mirror_release_fail_close) is tracked separately and stays sticky.
        self.mirror_release_enabled = enable_storage and os.getenv(
            "SGLANG_HICACHE_MIRROR_RELEASE", "0"
        ).lower() in ("1", "true")
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_config = prefetch_timeout_config
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
            existing_collector = getattr(self, "storage_metrics_collector", None)
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
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_config,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            self.cache_controller.attach_storage_backend(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
                **self._get_hybrid_storage_attach_kwargs(),
            )
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_config=prefetch_timeout_config,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        # No storage backend, no durable pages: stop treating any mirror as
        # reclaimable and forget the durable markings of the old backend.
        self.mirror_release_enabled = False
        self._reset_mirror_release_state()
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, prefetch_key, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if _operation.host_indices is not None:
                        cc.mem_pool_host.free(_operation.host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(prefetch_key)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_storage_hit=0,
            n_backup=None,
            n_release=None,
            log_metrics=False,
            acked_completed=None,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_storage_hit: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
        acked_completed: Optional[List[int]] = None,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
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
            # The L3 hit count is now known, so reserve exactly that much host
            # memory (this is the whole point: no over-allocation up front).
            # NOTE: alloc/evict here is rank-local but deterministic across TP
            # ranks without extra synchronization: host pool mutations only
            # happen on the scheduler thread at lockstep points (releases are
            # page-granular and drained by the TP-min count), so every rank
            # reaches the same success / fallback / revoke decision.
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
                    logger.debug(
                        f"Revoking prefetch for request {req_id} due to host memory allocation failure."
                    )
                    continue

                operation.storage_hit_count = alloc_len
                operation.hash_value = operation.hash_value[
                    : alloc_len // self.page_size
                ]
                operation.host_indices = host_indices
                cc.prefetch_buffer.put(operation)

        def _drain_backup():
            for ack_index, operation in enumerate(
                _drain_queue(cc.ack_backup_queue, n_backup)
            ):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if self.mirror_release_enabled:
                    # An ack is not success: _page_backup() stops advancing
                    # completed_tokens on the first failed batch but the
                    # operation is acked regardless, so only the completed
                    # page prefix is durable.  For MLA models only tp_rank 0
                    # writes L3 (backup_skip) and everyone else acks with 0,
                    # so the authoritative count is the collective-reduced
                    # per-op value, not the local one.  operation.hash_value
                    # is the list captured at write_storage() time, so
                    # marking by hash also covers nodes split mid-backup.
                    if acked_completed is not None and ack_index < len(
                        acked_completed
                    ):
                        completed = acked_completed[ack_index]
                    else:
                        completed = operation.completed_tokens
                    completed_pages = completed // self.page_size
                    for page_hash in (operation.hash_value or [])[:completed_pages]:
                        self._mark_hash_durable(page_hash)
                    if entry is not None:
                        self._update_redundant_host_status(entry)
                        # A node split during the backup leaves the prefix
                        # half as an ancestor the ack's entry pointer misses;
                        # walking up while ancestors are fully durable
                        # registers those halves (and any other now-covered
                        # ancestors) as release candidates.
                        parent = entry.parent
                        while (
                            parent is not None
                            and parent is not self.root_node
                            and self._node_fully_durable(parent)
                        ):
                            self._update_redundant_host_status(parent)
                            parent = parent.parent
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_and_alloc_storage_hit()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_config, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        defaults = PrefetchTimeoutConfig()
        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop(
            "prefetch_timeout_base", defaults.base
        )  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", defaults.per_ki_token
        )  # seconds per 1024 tokens
        prefetch_timeout_max = extra_config.pop(
            "prefetch_timeout_max", defaults.max
        )  # seconds, upper bound for the linear timeout
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(prefetch_timeout_max, (int, float)):
            raise ValueError(
                f"prefetch_timeout_max must be number, got {type(prefetch_timeout_max).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        prefetch_timeout_config = PrefetchTimeoutConfig(
            base=float(prefetch_timeout_base),
            per_ki_token=float(prefetch_timeout_per_ki_token),
            max=float(prefetch_timeout_max),
        )

        return (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_config,
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.evictable_host_leaves.clear()
        self._reset_mirror_release_state()
        super().reset()

    def release_host_resources(self) -> None:
        if self.token_to_kv_pool_host is not None:
            self.token_to_kv_pool_host.destroy()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def _get_extra_pools(self) -> dict:
        if not isinstance(self.cache_controller, HybridCacheController):
            return {}
        if isinstance(self.kv_cache, DSATokenToKVPool) or (
            isinstance(self.kv_cache, MiniMaxSparseKVPool)
            and self.kv_cache.index_k_pool is not None
        ):
            pool = PoolTransfer(
                name=PoolName.INDEXER,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                indices_from_pool=PoolName.KV,
            )
            return {"extra_pools": [pool]}
        else:
            return {}

    def _get_hybrid_storage_attach_kwargs(self) -> dict:
        """Extra kwargs for attach_storage_backend when controller is HybridCacheController."""
        if isinstance(self.cache_controller, HybridCacheController):
            return {"host_pools": self.cache_controller.mem_pool_host.entries}
        return {}

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    # everything previously marked durable is gone from L3
                    self._reset_mirror_release_state()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False) -> int:
        # Backup invariant (for write-through mode): backed-up nodes must form a
        # contiguous prefix from root — no gaps.  Skip if parent isn't backed
        # up yet.  A parent whose host mirror was released but whose pages are
        # all durable in L3 still satisfies the invariant: the prefix hash
        # chain it anchors exists in storage even without a host copy.
        if not write_back and (
            node.parent != self.root_node
            and not node.parent.backuped
            and not (
                self.mirror_release_enabled
                and self._node_fully_durable(node.parent)
            )
        ):
            return 0

        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
            **self._get_extra_pools(),
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
                **self._get_extra_pools(),
            )
        if host_indices is not None:
            node.host_value = host_indices.clone()
            assert len(node.host_value) > 0
            self._track_write_through_node(node, len(node.key))
            self._update_redundant_host_status(node)
            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def _track_write_through_node(self, node: TreeNode, backup_len: int) -> None:
        node.write_through_pending_id = node.id
        self.ongoing_write_through[node.id] = (node, backup_len, [node])

    def _replace_pending_write_through_node(
        self, old_node: TreeNode, new_nodes: List[TreeNode]
    ) -> None:
        ack_id = old_node.write_through_pending_id
        if ack_id is None:
            return

        pending = self.ongoing_write_through.get(ack_id)
        if pending is None:
            return

        lock_node, backup_len, publish_nodes = pending
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
        self.ongoing_write_through[ack_id] = (lock_node, backup_len, updated_nodes)

    def _finish_write_through_ack(self, ack_id: int, *, release_lock: bool) -> None:
        lock_node, backup_len, publish_nodes = self.ongoing_write_through.pop(ack_id)
        for node in publish_nodes:
            if node.write_through_pending_id == ack_id:
                node.write_through_pending_id = None
            # DMA confirmed -- block is now on host.
            self._record_store_event(node, medium=StorageMedium.CPU)
        if self.enable_storage:
            self.write_backup_storage(lock_node, backup_len)
        if release_lock:
            self.dec_lock_ref(lock_node)

    def write_backup_storage(self, node: TreeNode, backup_len: Optional[int] = None):
        # Recover pre-split data via walk-and-concat if node was split.
        # prefix_keys anchored at chain top to avoid double-counting.
        if backup_len is None or len(node.key) == backup_len:
            top, key, hash_value, host_value = (
                node,
                node.key,
                node.hash_value,
                node.host_value,
            )
        else:
            top, key, hash_value, host_value = self._concat_split_chain(
                node, backup_len
            )

        prefix_keys = (
            top.get_prefix_hash_values(top.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )

        operation_id = self.cache_controller.write_storage(
            host_value, key, hash_value, prefix_keys, **self._get_extra_pools()
        )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _concat_split_chain(self, node: TreeNode, backup_len: int):
        """Recover enqueue-time key/hash/host by walking the split chain."""
        chain, accumulated = [], 0
        current = node
        while current is not self.root_node and accumulated < backup_len:
            chain.append(current)
            accumulated += len(current.key)
            current = current.parent
        assert accumulated == backup_len, (
            f"backup chain length mismatch for node {node.id}: "
            f"expected {backup_len}, got {accumulated}"
        )
        chain.reverse()  # parent-first
        top = chain[0]
        if top.key.is_bigram:
            # Bigram segments share boundary tokens; drop overlap after first.
            token_ids = list(chain[0].key.token_ids)
            for n in chain[1:]:
                token_ids.extend(n.key.token_ids[1:])
        else:
            token_ids = []
            for n in chain:
                token_ids.extend(n.key.token_ids)
        key = RadixKey(token_ids, top.key.extra_key, top.key.is_bigram)

        if all(n.hash_value is not None for n in chain):
            hash_value = []
            for n in chain:
                hash_value.extend(n.hash_value)
        else:
            hash_value = None
        host_value = torch.cat([n.host_value for n in chain])
        return top, key, hash_value, host_value

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                if self._skip_restage_durable(node):
                    return
                # write to host if the node is not backuped
                self.write_backup(node)

    def _skip_restage_durable(self, node: TreeNode) -> bool:
        """Anti-churn: a released mirror flips backuped to False, and without
        this guard every later hit would re-stage and re-write the node to L3
        in an endless release/rewrite loop.  Age-gated so a copy the L3
        backend evicted on its own can still be refreshed eventually."""
        if not (self.mirror_release_enabled and self._node_fully_durable(node)):
            return False
        return (
            self.mirror_refresh_rounds <= 0
            or self._node_durable_age(node) < self.mirror_refresh_rounds
        )

    def _count_ready_acks(self, ack_queue) -> int:
        ready_count = 0
        for ack in ack_queue:
            if not ack.finish_event.query():
                break
            ready_count += 1
        return ready_count

    def _sync_hicache_ready_counts(self) -> tuple[int, int, tuple[int, ...]]:
        cache_controller = self.cache_controller
        storage_queue_sizes = (
            (
                cache_controller.prefetch_revoke_queue.qsize(),
                cache_controller.prefetch_hit_queue.qsize(),
                cache_controller.ack_backup_queue.qsize(),
                cache_controller.host_mem_release_queue.qsize(),
            )
            if self.enable_storage
            else ()
        )

        ready_counts = torch.tensor(
            [
                self._count_ready_acks(cache_controller.ack_write_queue),
                self._count_ready_acks(cache_controller.ack_load_queue),
                *storage_queue_sizes,
            ],
            dtype=torch.int,
            device="cpu",
        )
        self._all_reduce(ready_counts, torch.distributed.ReduceOp.MIN)

        count_values = list(map(int, ready_counts.tolist()))
        return count_values[0], count_values[1], tuple(count_values[2:])

    def writing_check(self, write_back=False, finish_count: Optional[int] = None):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for ack in self.cache_controller.ack_write_queue:
                    ack.finish_event.synchronize()
                    for ack_id in ack.node_ids:
                        self._finish_write_through_ack(ack_id, release_lock=False)
                    self._log_write_ack_metrics(ack)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        if finish_count is None:
            # Every rank must enter the all_reduce below; ongoing_write_through can
            # diverge across ranks (e.g. write_backup returning 0 on a subset under
            # host memory pressure), so a conditional skip desyncs the NCCL op
            # sequence and deadlocks under TP > 1. (Matches UnifiedRadixCache.)
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(
                    self.cache_controller.ack_write_queue
                )
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} write back operations")
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                self._finish_write_through_ack(ack_id, release_lock=True)
            self._log_write_ack_metrics(ack)
            finish_count -= 1

    def _log_write_ack_metrics(self, ack) -> None:
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

    def loading_check(self, finish_count: Optional[int] = None):
        if finish_count is None:
            finish_count = 0
            if self.pp_rank == 0:
                finish_count = self._count_ready_acks(
                    self.cache_controller.ack_load_queue
                )
            finish_count_tensor = torch.tensor(
                finish_count, dtype=torch.int, device="cpu"
            )
            self._all_reduce(finish_count_tensor, torch.distributed.ReduceOp.MIN)
            finish_count = finish_count_tensor.item()

        if finish_count > 0:
            logger.debug(f"Process {finish_count} load operations")
        while finish_count > 0:
            ack = self.cache_controller.ack_load_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

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

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        """Return True after the local load-back event is complete."""
        if consumer_index < 0:
            return True

        finish_event = self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False

        self.loading_check()
        return True

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.backuped:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    # --- L2-as-channel: durable tracking + mirror release -------------------

    def _mark_hash_durable(self, page_hash: str):
        d = self.storage_backed_hashes
        if page_hash in d:
            d.move_to_end(page_hash)
        d[page_hash] = self.hicache_logical_clock
        if len(d) > self.storage_backed_capacity:
            d.popitem(last=False)

    def _node_fully_durable(self, node: TreeNode) -> bool:
        hashes = node.hash_value
        if not hashes:
            return False
        d = self.storage_backed_hashes
        return all(h in d for h in hashes)

    def _node_durable_age(self, node: TreeNode) -> int:
        """Rounds since the oldest page of this node was last marked durable."""
        d = self.storage_backed_hashes
        return self.hicache_logical_clock - min(d[h] for h in node.hash_value)

    def _revoke_node_durable(self, node: TreeNode):
        """Forget durable markings when a node leaves the tree entirely.

        A durable mark is a record of one past backup ack, not a retention
        lease — the L3 backend may self-evict the pages at any time.  While
        the node is tree-resident that staleness is bounded (device or host
        still holds the data, and the refresh age re-stages eventually), but
        once the node is deleted a stale mark would keep claiming the pages
        exist and anti-churn would block the re-inserted prefix from ever
        re-staging.  Revoking on deletion costs at most one redundant L3
        rewrite after a full local eviction.
        """
        if not self.mirror_release_enabled:
            return
        for page_hash in node.hash_value or []:
            self.storage_backed_hashes.pop(page_hash, None)
        self.redundant_host_nodes.discard(node)

    def _update_redundant_host_status(self, node: TreeNode):
        if not self.mirror_release_enabled:
            return
        if (
            node is not self.root_node
            and node.value is not None
            and node.host_value is not None
            and self._node_fully_durable(node)
        ):
            self.redundant_host_nodes.add(node)
        else:
            self.redundant_host_nodes.discard(node)

    def _mirror_release_active(self) -> bool:
        return (
            self.mirror_release_enabled
            and self._mirror_release_disabled_reason is None
        )

    def _mirror_release_local_deficit(self) -> int:
        host_pool = self.cache_controller.mem_pool_host
        target = int(host_pool.size * self.mirror_release_free_frac)
        return max(0, target - int(host_pool.available_size()))

    def _mirror_node_releasable(self, node: TreeNode) -> bool:
        # lock_ref == 0 rules out all three in-flight windows: device->host DMA
        # (write_backup holds a lock until the ack), host->device load-back
        # (load_back locks the chain until loading_check), and request use;
        # host_ref_counter == 0 rules out an in-flight L3 backup or prefetch.
        return (
            node is not self.root_node
            and node.value is not None
            and node.host_value is not None
            and node.lock_ref == 0
            and node.host_ref_counter == 0
            and self._node_fully_durable(node)
        )

    def _prepare_mirror_release_plan(
        self, budget_tokens: int
    ) -> Optional[MirrorReleasePlan]:
        if budget_tokens <= 0 or not self.redundant_host_nodes:
            return None
        candidates = []
        for node in list(self.redundant_host_nodes):
            if self._mirror_node_releasable(node):
                candidates.append(node)
            elif (
                node.value is None
                or node.host_value is None
                or not self._node_fully_durable(node)
            ):
                # Permanently ineligible in this state; the state-transition
                # hooks re-register it if it becomes redundant again.  Nodes
                # that are merely busy (lock/ref held) stay members.
                self.redundant_host_nodes.discard(node)
        if not candidates:
            return None
        candidates.sort(
            key=lambda n: (self.eviction_strategy.get_priority(n), n.id)
        )
        plan = MirrorReleasePlan()
        digest = _FNV64_OFFSET
        for node in candidates:
            plan.nodes.append(node)
            plan.tokens += len(node.host_value)
            for page_hash in node.hash_value:
                digest = _fnv1a64(page_hash.encode(), digest)
            digest = _fnv1a64(len(node.host_value).to_bytes(8, "little"), digest)
            if plan.tokens >= budget_tokens:
                break
        plan.digest = digest & _INT63_MASK
        return plan

    def _release_mirror_of(self, node: TreeNode):
        host_indices = node.host_value
        node.host_value = None
        # Only the CPU medium is removed; the GPU copy stays device-resident
        # and downstream indexers keep scoring it as device-local.
        self._record_remove_event(node, medium=StorageMedium.CPU)
        released = self.cache_controller.evict_host(host_indices)
        self._mirror_released_tokens_total += released
        self.redundant_host_nodes.discard(node)
        self._update_host_leaf_status(node.parent)
        return released

    def _execute_mirror_release_plan(self, plan: MirrorReleasePlan):
        for node in plan.nodes:
            self._release_mirror_of(node)
        self._mirror_release_plans_executed += 1

    def _mirror_release_fail_close(self, reason: str):
        self._mirror_release_disabled_reason = reason
        self._mirror_release_plan = None
        logger.error(
            "HiCache mirror release permanently disabled (fail-close): %s. "
            "Reverting to pre-feature behavior; host mirrors are no longer "
            "reclaimed on this process.",
            reason,
        )

    def _advance_mirror_release(
        self,
        global_deficit: int,
        pc_min: int,
        pc_max: int,
        pt_min: int,
        pt_max: int,
        pd_min: int,
        pd_max: int,
    ):
        """Commit-or-drop last round's plan, then prepare the next one.

        All branch decisions below derive from all-reduced values, so every
        rank takes the same path; the only rank-local input is the plan
        revalidation, which is itself MIN-synced before any mutation.
        """
        plan = self._mirror_release_plan
        self._mirror_release_plan = None
        consensus = pc_min == pc_max and pt_min == pt_max and pd_min == pd_max
        if pc_max > 0 or pt_max > 0 or pd_max > 0:
            if consensus and pc_min > 0:
                # Every rank prepared the identical plan; make sure it is
                # still executable everywhere before mutating anything.
                local_valid = plan is not None and all(
                    self._mirror_node_releasable(n) for n in plan.nodes
                )
                valid = torch.tensor(
                    [1 if local_valid else 0], dtype=torch.int64
                )
                self._all_reduce_attn_groups(valid, torch.distributed.ReduceOp.MIN)
                if int(valid.item()) == 1:
                    self._execute_mirror_release_plan(plan)
                else:
                    # Benign state change (e.g. a planned node got locked).
                    # Under lockstep ranks this invalidation is identical
                    # everywhere, so dropping the whole plan stays consistent.
                    self._mirror_release_plans_dropped += 1
                self._mirror_release_mismatch_streak = 0
            elif not consensus:
                self._mirror_release_plans_dropped += 1
                self._mirror_release_mismatch_streak += 1
                logger.warning(
                    "mirror-release plan mismatch: my_plan=%s candidates=%d "
                    "durable_hashes=%d deficit=%d clock=%d "
                    "(count %d/%d tokens %d/%d digest %x/%x)",
                    f"{len(plan.nodes)}n/{plan.tokens}t/{plan.digest:x}"
                    if plan
                    else "none",
                    len(self.redundant_host_nodes),
                    len(self.storage_backed_hashes),
                    global_deficit,
                    self.hicache_logical_clock,
                    pc_min,
                    pc_max,
                    pt_min,
                    pt_max,
                    pd_min,
                    pd_max,
                )
                if self._mirror_release_mismatch_streak >= 4:
                    self._mirror_release_fail_close(
                        f"release-plan consensus mismatch x{self._mirror_release_mismatch_streak} "
                        f"(count {pc_min}/{pc_max}, tokens {pt_min}/{pt_max}, "
                        f"digest {pd_min:x}/{pd_max:x}) — durable state has "
                        "diverged across ranks"
                    )
                    return
        else:
            self._mirror_release_mismatch_streak = 0
        if self._mirror_release_active():
            self._mirror_release_plan = self._prepare_mirror_release_plan(
                global_deficit
            )

    def _reset_mirror_release_state(self):
        self.storage_backed_generation += 1
        self.storage_backed_hashes.clear()
        self.redundant_host_nodes.clear()
        self._mirror_release_plan = None
        self._mirror_release_mismatch_streak = 0

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        if self.cache_controller.write_policy == "write_back":
            num_evicted = self._evict_write_back(num_tokens)
        else:
            num_evicted = self._evict_write_through(num_tokens)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _make_eviction_heap(self):
        heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in self.evictable_leaves
        ]
        heapq.heapify(heap)
        return heap

    def _promote_parent(self, node: TreeNode, heap) -> None:
        # Once all of a node's children are evicted, it becomes a device leaf.
        p = node.parent
        if p is not self.root_node and all(c.evicted for c in p.children.values()):
            heapq.heappush(heap, (self.eviction_strategy.get_priority(p), p))

    def _evict_write_through(self, num_tokens: int) -> int:
        """write_through / write_through_selective: drop non-backuped leaves,
        demote already-backuped ones. Nothing is staged to host during eviction,
        so this is a plain on-the-fly pass.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            # A node demoted earlier in this same pass can be popped again via
            # a stale heap entry; with mirror release on, released-parent
            # pruning also mutates the tree mid-pass.  An already-evicted node
            # has no device memory left to reclaim.
            if x.evicted:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif x.children:
                # A released-mirror parent whose children were demoted
                # to host-only: it has no host copy to demote to, and
                # _evict_regular would delete a node that still anchors
                # children.  Eviction MUST make progress here — both
                # skipping (device pin -> prefill OOM) and re-staging
                # (whole heap turns into no-op pops under pressure)
                # starved the allocator in load tests.  Prune the
                # host-only subtree (its data lives in L3 or is
                # recomputable) so the parent becomes a true leaf and
                # frees its device tokens right now; fall back to
                # re-staging only while the subtree is busy.
                freed = self._evict_released_parent(x)
                if freed == 0:
                    self.write_backup(x)
                num_evicted += freed
            else:
                num_evicted += self._evict_regular(x)
            self._promote_parent(x, heap)
        return num_evicted

    def _evict_write_back(self, num_tokens: int) -> int:
        """eviction for write_back mode: demote already-backuped leaves, stage non-backuped ones to host if possible, otherwise drop them.
        note this path will be deprecated in the future.
        """
        heap = self._make_eviction_heap()
        num_evicted = 0
        staged: List[Tuple[TreeNode, torch.Tensor]] = []

        def flush_staged() -> None:
            if not staged:
                return
            self.writing_check(write_back=True)
            for node, device_indices in staged:
                self.cache_controller.evict_device(device_indices)
                node.release_host()
            staged.clear()

        while num_evicted < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x.lock_ref > 0:
                continue
            if x.backuped:
                num_evicted += self._evict_backuped(x)
            elif self.write_backup(x, write_back=True) > 0:
                x.protect_host()
                staged.append((x, x.value))
                num_evicted += self._detach_backuped(x)
            else:
                flush_staged()
                num_evicted += self._drop_subtree_no_host(x)
            self._promote_parent(x, heap)
        flush_staged()
        return num_evicted

    def _detach_backuped(self, node: TreeNode) -> int:
        # detach nodes from tree while keeping device slots, for write-back eviction
        self._record_remove_event(node, medium=StorageMedium.GPU)
        num_evicted = len(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # demoted to host-only: its host copy is the real copy now, not a mirror
        self._update_redundant_host_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_backuped(self, node: TreeNode):
        device_indices = node.value
        num_evicted = self._detach_backuped(node)
        self.cache_controller.evict_device(device_indices)
        return num_evicted

    def _evict_released_parent(self, node: TreeNode) -> int:
        """Evict a released-mirror parent by pruning its host-only subtree.

        Every descendant must be device-evicted, host-backed and unpinned;
        their host copies are freed and the nodes unlinked (the durable data
        stays in L3, anything not yet durable is recomputable), after which
        the parent is a true leaf and goes through _evict_regular.  Returns
        the device tokens freed, or 0 if the subtree is busy.
        """
        descendants = []
        stack = list(node.children.values())
        while stack:
            d = stack.pop()
            if (
                d.value is not None
                or d.host_value is None
                or d.lock_ref > 0
                or d.host_ref_counter > 0
            ):
                return 0
            descendants.append(d)
            stack.extend(d.children.values())

        for d in descendants:
            self._record_remove_event(d, medium=StorageMedium.CPU)
            self.cache_controller.evict_host(d.host_value)
            d.host_value = None
            self.evictable_host_leaves.discard(d)
            self._revoke_node_durable(d)
        node.children = {}
        self._update_host_leaf_status(node)
        return self._evict_regular(node)

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host -- emit BlockRemoved
        assert len(node.children) == 0, f"non-leaf, {node.id=}"

        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        self._revoke_node_durable(node)
        return num_evicted

    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        nodes = []
        stack = [root]
        while stack:
            n = stack.pop()
            nodes.append(n)
            stack.extend(n.children.values())

        if any(n.host_ref_counter > 0 for n in nodes):
            return 0

        logger.warning(
            "write_back: KV cache on device are dropped without backup due to host memory pressure, subtree root %d, num_nodes %d",
            root.id,
            len(nodes),
        )

        freed_device = 0
        for n in nodes:
            if n.host_value is not None:
                self._record_remove_event(n, medium=StorageMedium.CPU)
                self.cache_controller.evict_host(n.host_value)
                n.host_value = None
            if n.value is not None:
                self._record_remove_event(n, medium=StorageMedium.GPU)
                self.cache_controller.mem_pool_device_allocator.free(n.value)
                freed_device += len(n.value)
                self.evictable_size_ -= len(n.value)
                n.value = None
            self.ongoing_write_through.pop(n.id, None)
            self.evictable_leaves.discard(n)
            self.evictable_host_leaves.discard(n)

        key = root.key.child_key(self.page_size)
        root.parent.children.pop(key, None)
        self._update_leaf_status(root.parent)
        self._update_host_leaf_status(root.parent)
        if freed_device > 0 and self.metrics_collector is not None:
            self.metrics_collector.increment_dropped_tokens(
                num_tokens=freed_device,
                reason="host_pressure",
                pool=PoolName.KV.value,
            )
        return freed_device

    def evict_host(self, num_tokens: int):
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --
            # emit remove(CPU) so the router drops the host-tier entry.
            self._record_remove_event(x, medium=StorageMedium.CPU)
            num_evicted += self.cache_controller.evict_host(x.host_value)

            key = x.key.child_key(self.page_size)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._revoke_node_durable(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        result = self.inc_lock_ref(ancester_node)
        delta = result.delta

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        # Protect the nodes being loaded from host eviction.
        for n in nodes_to_load:
            n.protect_host()

        device_indices = self.cache_controller.load(
            host_indices=host_indices,
            node_id=last_hit_node.id,
            **self._get_extra_pools(),
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            for n in nodes_to_load:
                n.release_host()
            logger.warning(
                "load_back: FAILED to load %d tokens for node %d "
                "even after eviction (evictable_size=%d)",
                len(host_indices),
                last_hit_node.id,
                self.evictable_size_,
            )
            return None

        for n in nodes_to_load:
            n.release_host()
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)].clone()
            offset += len(node.host_value)
            # Block promoted from host to GPU -- emit store(GPU) so downstream
            # indexers see it as device-local again.
            self._record_store_event(node, medium=StorageMedium.GPU)
            # device + host again: if all pages are durable the host copy is a
            # mirror once more (the lock taken below defers actual release
            # until the load completes).
            self._update_redundant_host_status(node)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ):
        last_node = params.best_match_node
        mem_quota = params.mem_quota
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            self._empty_match_result.device_indices,
            last_node,
        )

    def query_storage_hit_length(
        self,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> int:
        if not self.enable_storage or self.cache_controller.prefetch_rate_limited():
            return 0

        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        ).page_aligned(self.page_size)
        if len(prefetch_key) < self.prefetch_threshold:
            return 0

        prefetch_op_cls = (
            HybridPrefetchOperation
            if isinstance(self.cache_controller, HybridCacheController)
            else PrefetchOperation
        )
        extra_kwargs = {}
        if prefetch_op_cls is HybridPrefetchOperation:
            extra_kwargs["pool_transfers"] = self._get_extra_pools().get("extra_pools")
        operation = prefetch_op_cls(
            "__storage_hit_query__",
            prefetch_key,
            last_hash,
            prefix_keys,
            **extra_kwargs,
        )
        hash_values, storage_hit_count = self.cache_controller._storage_hit_query(
            operation
        )
        storage_hit_count_tensor = torch.tensor(storage_hit_count, dtype=torch.int)
        self._all_reduce_attn_groups(
            storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
        )
        storage_hit_count = storage_hit_count_tensor.item()
        storage_hit_count = storage_hit_count - (storage_hit_count % self.page_size)
        return storage_hit_count

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def check_hicache_events(self):
        # Reap the previous round's PP-sync sends before issuing new ones.
        self._drain_async_work()

        if self.pp_size != 1 or self.mirror_release_enabled:
            # Mirror release rides the storage-drain collective (deficit,
            # release-plan consensus triple, synced backup acks), which the
            # fused ready-counts path below does not carry.  The flag is
            # env-driven and identical on every rank, so all ranks take the
            # same branch.
            self.writing_check()
            self.loading_check()
            if self.enable_storage:
                self.drain_storage_control_queues()
        else:
            (
                write_finish_count,
                load_finish_count,
                storage_queue_sizes,
            ) = self._sync_hicache_ready_counts()
            self.writing_check(finish_count=write_finish_count)
            self.loading_check(finish_count=load_finish_count)

            if self.enable_storage and storage_queue_sizes:
                n_revoke, n_storage_hit, n_backup, n_release = storage_queue_sizes[:4]
                self._drain_storage_control_queues_impl(
                    n_revoke=n_revoke,
                    n_storage_hit=n_storage_hit,
                    n_backup=n_backup,
                    n_release=n_release,
                    log_metrics=True,
                )
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.

        The same collective also carries the mirror-release protocol fields:
        the pool deficit (negated, so the MIN recovers the cross-rank MAX) and
        last round's release-plan consensus triple (count/tokens/digest, each
        paired with its negation so MIN yields both min and max).  A plan only
        executes after every rank reports the identical triple — mutation
        never precedes consensus.
        """
        cc = self.cache_controller

        self.hicache_logical_clock += 1
        mirror_active = self._mirror_release_active()
        plan = self._mirror_release_plan if mirror_active else None
        plan_count = len(plan.nodes) if plan else 0
        plan_tokens = plan.tokens if plan else 0
        plan_digest = plan.digest if plan else 0
        local_deficit = self._mirror_release_local_deficit() if mirror_active else 0

        ack_slots = [_MIRROR_ACK_NO_WRITER] * _MIRROR_ACK_SYNC_SLOTS
        if self.mirror_release_enabled:
            # Peek (not pop) the acks this round may consume.  The queue is
            # FIFO with a single consumer, so the first min(qsize) entries
            # are the same logical operations in the same order on every
            # rank; only the writer ranks contribute their completed counts.
            # The backup thread puts concurrently, so both the length and
            # the head snapshot must be taken under the queue's own mutex —
            # bare iteration over Queue.queue can raise "deque mutated
            # during iteration" and kill the scheduler loop.
            with cc.ack_backup_queue.mutex:
                queue = cc.ack_backup_queue.queue
                n_backup_local = len(queue)
                ack_peek = list(itertools.islice(queue, _MIRROR_ACK_SYNC_SLOTS))
            if not getattr(cc, "backup_skip", False):
                for i, operation in enumerate(ack_peek):
                    ack_slots[i] = operation.completed_tokens
        else:
            n_backup_local = cc.ack_backup_queue.qsize()

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.prefetch_hit_queue.qsize(),
                n_backup_local,
                cc.host_mem_release_queue.qsize(),
                -local_deficit,
                plan_count,
                -plan_count,
                plan_tokens,
                -plan_tokens,
                plan_digest,
                -plan_digest,
                *ack_slots,
            ],
            # int64: the plan digest and the NO_WRITER ack sentinel exceed
            # int32 range; queue counts are unaffected by the wider dtype.
            dtype=torch.int64,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        vals = qsizes.tolist()
        n_revoke, n_storage_hit, n_backup, n_release = (int(v) for v in vals[:4])
        acked_completed = None
        if self.mirror_release_enabled:
            # Consume at most as many acks as we synced completed counts for.
            n_backup = min(n_backup, _MIRROR_ACK_SYNC_SLOTS)
            acked_completed = [
                0 if v >= _MIRROR_ACK_NO_WRITER else max(0, int(v))
                for v in vals[11 : 11 + _MIRROR_ACK_SYNC_SLOTS]
            ]
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_storage_hit=n_storage_hit,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
            acked_completed=acked_completed,
        )
        if mirror_active:
            self._advance_mirror_release(
                global_deficit=-int(vals[4]),
                pc_min=int(vals[5]),
                pc_max=-int(vals[6]),
                pt_min=int(vals[7]),
                pt_max=-int(vals[8]),
                pd_min=int(vals[9]),
                pd_max=-int(vals[10]),
            )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        cfg = self.prefetch_timeout_config
        num_tokens = len(operation.hash_value) * self.page_size
        timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)
        return time.monotonic() - operation.start_time > timeout

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
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
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def _revoke_pending_prefetch(self, req_id: str):
        info = self.ongoing_prefetch.pop(req_id, None)
        if info is None:
            return
        last_host_node, prefetch_key, _ = info
        last_host_node.release_host()
        cc = self.cache_controller
        cc.prefetch_tokens_occupied = max(
            0, cc.prefetch_tokens_occupied - len(prefetch_key)
        )

    def check_prefetch_progress(self, req_id: str) -> bool:
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        last_host_node, prefetch_key, operation = self.ongoing_prefetch[req_id]

        if not self.can_terminate_prefetch(operation):
            return False

        if operation.host_indices is None:
            # Stopping before host memory was committed (best_effort, timeout, or
            # still mid-query): signal the worker to stop, then release the request.
            self.cache_controller.terminate_prefetch(operation)
            self._revoke_pending_prefetch(req_id)
            return True

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = self._sync_and_clamp_prefetch_result(
            operation, completed_tokens
        )

        fetched_key = prefetch_key[:min_completed_tokens]
        written_indices = operation.host_indices[:min_completed_tokens]
        matched_length = self._insert_helper_host(
            last_host_node,
            fetched_key,
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
        )

        self.cache_controller.mem_pool_host.free(
            operation.host_indices[:matched_length]
        )
        self.cache_controller.append_host_mem_release(
            operation.host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        self.prefetch_loaded_tokens_by_reqid[req_id] = loaded_from_storage

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def _sync_and_clamp_prefetch_result(
        self,
        operation: PrefetchOperation,
        completed_tokens: int,
    ) -> int:
        """Sync prefetch results across ATTN groups and decide the usable prefix.

        HiRadixCache only wires DSA-style stacks (Full attention + a KV-derived
        ALL_PAGES sidecar such as the DSA / MiniMax indexer); For the DSA case we *clamp*
        to the minimum fetched prefix shared by the Full KV pool and every
        sidecar rather than discarding everything. With no sidecar (FULL-only)
        this is just the synced Full KV completion.
        """
        # Sync completed tokens and per-pool hit pages across ATTN groups, taking
        # the minimum so every rank agrees on the same usable prefix length.
        pool_transfers = getattr(operation, "pool_transfers", None) or []
        hit_pages = (
            operation.pool_storage_result.extra_pool_hit_pages if pool_transfers else {}
        )
        pool_hit_pages = [hit_pages.get(t.name, 0) for t in pool_transfers]
        packed = torch.tensor([completed_tokens, *pool_hit_pages], dtype=torch.int)
        self._all_reduce_attn_groups(packed, torch.distributed.ReduceOp.MIN)
        min_completed_tokens = int(packed[0].item())
        pool_hit_pages = list(map(int, packed[1:].tolist()))

        # Clamp to the shared minimum prefix of the Full KV completion and each
        # KV-derived ALL_PAGES sidecar (e.g. the DSA indexer). FULL-only has no
        # sidecar, so the usable prefix is just the Full KV completion.
        usable_pages = min_completed_tokens // self.page_size
        if pool_transfers:
            usable_pages = min(usable_pages, *pool_hit_pages)
        return usable_pages * self.page_size

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, operation = self.ongoing_prefetch[req_id]
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def match_prefix(self, params: MatchPrefixParams):
        if self.disable:
            return self._empty_match_result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        while not last_host_node.backuped:
            last_host_node = last_host_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            # TODO(ispobock): use best_match_node as start node for load_back
            best_match_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        prefetch_key = RadixKey(
            new_input_tokens,
            extra_key=last_host_node.key.extra_key,
            is_bigram=self.is_eagle,
        )
        # align the number of fetching tokens to the page size
        prefetch_key = prefetch_key.page_aligned(self.page_size)
        prefetch_length = len(prefetch_key)
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            return

        last_host_node.protect_host()
        # NOTE: host_indices is no longer pre-allocated here. It is allocated
        # lazily in _drain_and_alloc_storage_hit() once the L3 storage hit count is known,
        # so we only reserve host memory for pages that actually hit.
        operation = self.cache_controller.prefetch(
            req_id,
            prefetch_key,
            last_hash,
            prefix_keys,
            **self._get_extra_pools(),
        )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            prefetch_key,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(prefetch_key)

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = node.key.match(key, page_size=self.page_size)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            # Publish the newly materialized host suffix immediately so downstream
            # cache indexers can resolve descendants that extend this L2-only prefix.
            self._record_store_event(new_node, medium=StorageMedium.CPU)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = key.child_key(self.page_size)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode(priority=child.priority)
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        if child.backuped:
            self._replace_pending_write_through_node(child, [new_node, child])
        # both halves must re-qualify for mirror release from their own hash
        # slices; registering only the ack'd original would leak the prefix
        # half's mirror forever under shared-prefix workloads.
        self._update_redundant_host_status(new_node)
        self._update_redundant_host_status(child)

        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        node = self.root_node
        child_key = key.child_key(self.page_size)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                    # recomputation restored the device copy: a durable host
                    # copy is a redundant mirror again and must re-enter the
                    # release candidates, or it stays pinned forever.
                    self._update_redundant_host_status(node)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                    # see the full-match recomputation branch above
                    self._update_redundant_host_status(new_node)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage or kv events are enabled
            if self.enable_storage or self.enable_kv_cache_events:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)

            # Emit BlockStored so the router indexes this block.
            self._record_store_event(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, prefetch_key, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            self.cache_controller.terminate_prefetch(operation)
            self._revoke_pending_prefetch(rid)
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(
            operation.host_indices[:completed_tokens]
        )
        self.cache_controller.prefetch_tokens_occupied -= len(prefetch_key)
