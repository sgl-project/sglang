from __future__ import annotations

import atexit
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.config import (
    shared_hicache_config,
    shared_hicache_config_value,
    shared_hicache_timeout_secs,
)
from sglang.srt.mem_cache.shared_hicache.control import (
    is_indeterminate_direct_transfer_reason,
    request_source_transfer,
)
from sglang.srt.mem_cache.shared_hicache.metrics import observe_reuse
from sglang.srt.mem_cache.shared_hicache.pending import (
    SharedHiCachePendingFetch,
    format_optional_ms,
    pending_ready_wait_ms,
    pending_should_stop_waiting,
    pending_wait_ms,
    transfer_bytes_for_pages,
)
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_PLAN_VERSION,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.service import (
    SharedHiCacheSourceService,
    endpoint_format_fields,
    format_control_endpoint,
)
from sglang.srt.mem_cache.shared_hicache.source import (
    ResolvedHostPage,
    handle_source_transfer,
)
from sglang.srt.mem_cache.shared_hicache.target import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.transfer import (
    SharedHiCacheTransferBackend,
    make_shared_hicache_transfer_backend,
    scheduler_parallel_metadata,
    shared_hicache_parallel_rejection,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

SHARED_HICACHE_MAX_CONTROL_BODY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SharedHiCacheResult:
    staged_tokens: int = 0
    prefix_len: int = 0
    pending: bool = False


def _shared_hicache_enabled(server_args: "ServerArgs") -> bool:
    config = shared_hicache_config(server_args)
    return bool(getattr(server_args, "enable_shared_hicache", False) or config)


class SharedHiCacheManager:
    def __init__(
        self,
        *,
        server_args: "ServerArgs",
        tree_cache,
        worker_id: Optional[int],
        attn_dp_rank: int,
        parallel_metadata: Optional[Mapping[str, int]] = None,
        direct_transfer: Optional[SharedHiCacheTransferBackend] = None,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self._set_parallel_metadata(
            parallel_metadata, attn_dp_rank=attn_dp_rank
        )
        self.timeout_secs = shared_hicache_timeout_secs(server_args)
        self.prefetch_stop_policy = getattr(
            server_args, "hicache_storage_prefetch_policy", "timeout"
        )
        self.prefetch_timeout_config = getattr(
            tree_cache, "prefetch_timeout_config", None
        )
        self.direct_transfer = direct_transfer
        self.metrics_collector = metrics_collector
        if not self._direct_transfer_enabled():
            logger.warning(
                "SharedHiCache is enabled but no direct transfer backend is available; "
                "SharedHiCache plans will be treated as cache misses."
            )
        endpoint_spec = shared_hicache_config_value(
            server_args,
            "control_endpoint",
            None,
        )
        self.endpoint = self._format_local_control_endpoint(
            endpoint_spec, self.attn_dp_rank
        )
        self.source_service: Optional[SharedHiCacheSourceService] = None
        self._shutdown = False
        worker_limit = max(
            1,
            int(envs.SGLANG_SHARED_HICACHE_FETCH_WORKERS.get()),
        )
        self._fetch_semaphore = threading.BoundedSemaphore(worker_limit)
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix=f"shared_hicache-fetch-adp{self.attn_dp_rank}",
        )
        self._pending_fetches: dict[str, SharedHiCachePendingFetch] = {}
        self._detached_fetches: set[Future] = set()
        self.target_cache = SharedHiCacheTarget(
            tree_cache=tree_cache,
            metrics_collector=metrics_collector,
        )
        self._finished_plan_keys: set[tuple[str, str]] = set()
        self._finished_plan_prefix_lens: dict[tuple[str, str], int] = {}
        self.max_control_body_bytes = SHARED_HICACHE_MAX_CONTROL_BODY_BYTES
        self._direct_transfer_shutdown_done = False
        self._direct_transfer_shutdown_deferred = False
        self._direct_transfer_shutdown_lock = threading.Lock()

        if self.endpoint is not None:
            self.source_service = SharedHiCacheSourceService(
                endpoint=self.endpoint,
                worker_id=self.worker_id,
                attn_dp_rank=self.attn_dp_rank,
                worker_limit=worker_limit,
                max_body_bytes=self._max_control_body_bytes,
                direct_transfer_enabled=self._direct_transfer_enabled,
                handle_source_transfer=self._handle_source_transfer,
            )
            self.source_service.start()
        atexit.register(self.shutdown)

    def _set_parallel_metadata(
        self,
        parallel_metadata: Optional[Mapping[str, int]],
        *,
        attn_dp_rank: int,
    ) -> None:
        metadata = {
            key: int(value) for key, value in (parallel_metadata or {}).items()
        }
        self.attn_dp_rank = int(metadata.get("attn_dp_rank", attn_dp_rank))
        self.attn_dp_size = int(metadata.get("attn_dp_size", 1))
        self.attn_tp_rank = int(metadata.get("attn_tp_rank", 0))
        self.attn_tp_size = int(metadata.get("attn_tp_size", 1))
        self.tp_rank = int(metadata.get("tp_rank", self.attn_tp_rank))
        self.tp_size = int(metadata.get("tp_size", self.attn_tp_size))
        self.pp_rank = int(metadata.get("pp_rank", 0))
        self.pp_size = int(metadata.get("pp_size", 1))
        self.attn_cp_rank = int(metadata.get("attn_cp_rank", 0))
        self.attn_cp_size = int(metadata.get("attn_cp_size", 1))
        self.moe_ep_rank = int(metadata.get("moe_ep_rank", 0))
        self.moe_ep_size = int(metadata.get("moe_ep_size", 1))

    def _endpoint_format_values(self) -> dict[str, int]:
        return {
            "attn_dp_rank": self.attn_dp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_tp_size": self.attn_tp_size,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_cp_size": self.attn_cp_size,
            "moe_ep_rank": self.moe_ep_rank,
            "moe_ep_size": self.moe_ep_size,
        }

    def _format_local_control_endpoint(
        self, endpoint_spec: object, attn_dp_rank: int
    ) -> Optional[str]:
        endpoint = format_control_endpoint(
            endpoint_spec,
            attn_dp_rank,
            self._endpoint_format_values(),
        )
        if endpoint is None:
            return None
        fields = endpoint_format_fields(endpoint_spec)
        if self.attn_tp_size > 1 and not fields.intersection(
            {"attn_tp_rank", "tp_rank"}
        ):
            logger.warning(
                "SharedHiCache source resolver endpoint must include {attn_tp_rank} "
                "or {tp_rank} when attn_tp_size=%d; not starting source resolver "
                "for this rank",
                self.attn_tp_size,
            )
            return None
        if self.attn_dp_size > 1 and "attn_dp_rank" not in fields:
            logger.warning(
                "SharedHiCache source resolver endpoint must include {attn_dp_rank} "
                "when attn_dp_size=%d; not starting source resolver "
                "for this rank",
                self.attn_dp_size,
            )
            return None
        return endpoint

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["SharedHiCacheManager"]:
        server_args = scheduler.server_args
        if not _shared_hicache_enabled(server_args):
            return None
        if not scheduler.enable_hierarchical_cache:
            logger.warning(
                "SharedHiCache disabled because hierarchical cache is not enabled"
            )
            return None
        required_tree_methods = (
            "lookup_hicache_host_blocks",
            "insert_shared_hicache_device_blocks",
        )
        missing_tree_methods = [
            name
            for name in required_tree_methods
            if not callable(getattr(scheduler.tree_cache, name, None))
        ]
        if missing_tree_methods:
            logger.warning(
                "SharedHiCache disabled because the active tree cache lacks HiCache "
                "shared-cache primitives: %s",
                ", ".join(missing_tree_methods),
            )
            return None
        worker_id = getattr(server_args, "shared_hicache_worker_id", None)
        if worker_id is None:
            worker_id = shared_hicache_config_value(server_args, "worker_id", None)
        if worker_id is None:
            logger.warning(
                "SharedHiCache disabled because worker_id is not set; "
                "set --shared-hicache-worker-id"
            )
            return None
        direct_transfer = make_shared_hicache_transfer_backend(scheduler)
        parallel_metadata = scheduler_parallel_metadata(scheduler)
        return cls(
            server_args=server_args,
            tree_cache=scheduler.tree_cache,
            worker_id=worker_id,
            attn_dp_rank=parallel_metadata["attn_dp_rank"],
            parallel_metadata=parallel_metadata,
            direct_transfer=direct_transfer,
            metrics_collector=(
                scheduler.metrics_collector
                if getattr(scheduler, "enable_metrics", False)
                else None
            ),
        )

    def _current_backend_label(self) -> str:
        if self._direct_transfer_enabled():
            direct_transfer = getattr(self, "direct_transfer", None)
            return str(getattr(direct_transfer, "name", "direct"))
        return "none"

    def _direct_transfer_enabled(self) -> bool:
        direct_transfer = getattr(self, "direct_transfer", None)
        return direct_transfer is not None and bool(
            getattr(direct_transfer, "enabled", False)
        )

    def _max_control_body_bytes(self) -> int:
        return int(
            getattr(
                self,
                "max_control_body_bytes",
                SHARED_HICACHE_MAX_CONTROL_BODY_BYTES,
            )
        )

    def _try_acquire_fetch_worker(self) -> bool:
        semaphore = getattr(self, "_fetch_semaphore", None)
        if semaphore is None:
            return True
        return semaphore.acquire(blocking=False)

    def _release_fetch_worker(self) -> None:
        semaphore = getattr(self, "_fetch_semaphore", None)
        if semaphore is None:
            return
        try:
            semaphore.release()
        except ValueError:
            logger.debug(
                "SharedHiCache fetch worker semaphore release ignored", exc_info=True
            )

    def _on_pending_fetch_done(
        self, pending: SharedHiCachePendingFetch, future: Future
    ) -> None:
        try:
            pending.done_at = time.perf_counter()
        finally:
            self._release_fetch_worker()

    def _shutdown_direct_transfer_backend(self) -> None:
        direct_transfer = getattr(self, "direct_transfer", None)
        shutdown = getattr(direct_transfer, "shutdown", None)
        if shutdown is None:
            return
        lock = getattr(self, "_direct_transfer_shutdown_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._direct_transfer_shutdown_lock = lock
        with lock:
            if getattr(self, "_direct_transfer_shutdown_done", False):
                return
            shutdown()
            self._direct_transfer_shutdown_done = True

    def _defer_direct_transfer_shutdown(self) -> None:
        direct_transfer = getattr(self, "direct_transfer", None)
        if getattr(direct_transfer, "shutdown", None) is None:
            return
        if getattr(self, "_direct_transfer_shutdown_deferred", False):
            return
        self._direct_transfer_shutdown_deferred = True

        def _wait_for_pending_and_shutdown():
            while self.has_pending():
                time.sleep(0.01)
            try:
                self.target_cache.release_quarantined_device_indices()
                self._shutdown_direct_transfer_backend()
            except Exception:
                logger.warning(
                    "SharedHiCache deferred direct transfer backend shutdown failed",
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_wait_for_pending_and_shutdown,
            name="shared_hicache-direct-transfer-shutdown",
            daemon=True,
        )
        self._direct_transfer_shutdown_thread = thread
        thread.start()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        source_service = getattr(self, "source_service", None)
        self.source_service = None
        if source_service is not None:
            source_service.shutdown()

        for pending in self._pending_fetches.values():
            self._release_pending_fetch(pending)
        self._pending_fetches.clear()
        self._fetch_executor.shutdown(wait=False, cancel_futures=True)

        timeout_secs = float(getattr(self, "timeout_secs", 0.0))
        deadline = time.monotonic() + min(max(timeout_secs, 0.0), 5.0)
        while self.has_pending() and time.monotonic() < deadline:
            time.sleep(0.01)

        if self.has_pending():
            logger.warning(
                "Deferring direct transfer backend shutdown while SharedHiCache work is still pending"
            )
            self._defer_direct_transfer_shutdown()
            return

        self.target_cache.release_quarantined_device_indices()
        self._shutdown_direct_transfer_backend()

    def _candidate_endpoints_for_plan(self, plan: SharedHiCachePlan) -> list[str]:
        endpoints: list[str] = []

        def add(endpoint: Optional[str]) -> None:
            if endpoint and endpoint not in endpoints:
                endpoints.append(endpoint)

        if plan.source_endpoint:
            values = self._endpoint_format_values()
            values.update(
                {
                    "attn_dp_rank": int(plan.source_attn_dp_rank),
                    "source_attn_tp_rank": int(self.attn_tp_rank),
                    "source_attn_tp_size": int(plan.source_attn_tp_size),
                    "source_attn_dp_rank": int(plan.source_attn_dp_rank),
                    "target_attn_dp_rank": int(self.attn_dp_rank),
                    "target_attn_tp_rank": int(self.attn_tp_rank),
                    "target_attn_tp_size": int(self.attn_tp_size),
                }
            )
            add(
                format_control_endpoint(
                    plan.source_endpoint, plan.source_attn_dp_rank, values
                )
            )
        return endpoints

    def _request_source_transfer(
        self,
        *,
        transfer_backend: SharedHiCacheTransferBackend,
        endpoints: list[str],
        plan: SharedHiCachePlan,
        start_block: int,
        max_blocks: int,
        target_page_indices: list[int],
    ) -> tuple[list[ResolvedHostPage], str]:
        return request_source_transfer(
            transfer_backend=transfer_backend,
            endpoints=endpoints,
            plan=plan,
            start_block=start_block,
            max_blocks=max_blocks,
            target_page_indices=target_page_indices,
            timeout_secs=self.timeout_secs,
        )

    def _handle_source_transfer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return handle_source_transfer(
            payload=payload,
            transfer_backend=self.direct_transfer,
            tree_cache=getattr(self, "tree_cache", None),
            worker_id=getattr(self, "worker_id", None),
            attn_dp_rank=self.attn_dp_rank,
            attn_tp_rank=getattr(self, "attn_tp_rank", 0),
            attn_tp_size=getattr(self, "attn_tp_size", 1),
            pp_size=getattr(self, "pp_size", 1),
            attn_cp_size=getattr(self, "attn_cp_size", 1),
        )

    def _max_cacheable_blocks(self, req: "Req") -> int:
        max_prefix_len = max(len(req.fill_ids) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if req.positional_embed_overrides is not None:
            max_prefix_len = 0
        return max_prefix_len // self.tree_cache.page_size

    def _validate_plan(self, plan: SharedHiCachePlan) -> Optional[str]:
        if self.worker_id is None:
            return "missing_worker_id"
        if plan.target_worker_id != self.worker_id:
            return "wrong_target_worker"
        if plan.target_attn_dp_rank != self.attn_dp_rank:
            return "wrong_target_attn_dp_rank"
        rank_rejection = self._validate_target_rank(plan)
        if rank_rejection is not None:
            return rank_rejection
        if (
            plan.source_worker_id == plan.target_worker_id
            and plan.source_attn_dp_rank == plan.target_attn_dp_rank
        ):
            return "source_is_target"
        if plan.plan_version != SHARED_HICACHE_PLAN_VERSION:
            return "unsupported_plan_version"
        if plan.is_expired():
            return "plan_expired"
        if not plan.is_shared_hicache():
            return "unsupported_source_medium"
        if plan.block_size_tokens != self.tree_cache.page_size:
            return "incompatible_block_size"
        return None

    def _validate_target_rank(self, plan: SharedHiCachePlan) -> Optional[str]:
        topology_rejection = shared_hicache_parallel_rejection(
            pp_size=getattr(self, "pp_size", 1),
            attn_cp_size=getattr(self, "attn_cp_size", 1),
        )
        if topology_rejection is not None:
            return f"unsupported_target_topology:{topology_rejection}"

        if plan.target_attn_tp_size != self.attn_tp_size:
            return (
                "wrong_target_attn_tp_size:"
                f"plan={plan.target_attn_tp_size}:local={self.attn_tp_size}"
            )
        if plan.source_attn_tp_size != self.attn_tp_size:
            return (
                "incompatible_source_attn_tp_size:"
                f"source={plan.source_attn_tp_size}:target={self.attn_tp_size}"
            )
        return None

    def _plan_key(self, req: "Req", plan: SharedHiCachePlan) -> tuple[str, str]:
        return str(req.rid), plan.plan_id

    def has_pending(self) -> bool:
        detached_fetches = getattr(self, "_detached_fetches", set())
        for future in list(detached_fetches):
            if future.done():
                detached_fetches.discard(future)
        source_service = getattr(self, "source_service", None)
        active_source_count = (
            source_service.active_count() if source_service is not None else 0
        )
        return (
            bool(getattr(self, "_pending_fetches", {}))
            or bool(detached_fetches)
            or active_source_count > 0
        )

    def _lock_request_prefix(self, req: "Req") -> Optional[TreeNode]:
        last_node = getattr(req, "last_node", None)
        if last_node is None or last_node is self.tree_cache.root_node:
            return None
        self.tree_cache.inc_lock_ref(last_node)
        return last_node

    def _unlock_pending_prefix(self, pending: SharedHiCachePendingFetch) -> None:
        locked_node = getattr(pending, "locked_node", None)
        if locked_node is None:
            return
        pending.locked_node = None
        self.tree_cache.dec_lock_ref(locked_node)

    def _release_pending_fetch(self, pending: SharedHiCachePendingFetch) -> None:
        cancelled = pending.future.cancel()
        self._unlock_pending_prefix(pending)
        if pending.device_indices is None:
            return
        backend = getattr(pending, "backend", self._current_backend_label())
        if cancelled:
            self.target_cache.free_device_indices(pending.device_indices)
        elif pending.future.done():
            self.target_cache.release_device_indices_after_fetch_done(
                pending.future, pending.device_indices, backend=backend
            )
        else:
            detached_fetches = getattr(self, "_detached_fetches", None)
            if detached_fetches is None:
                detached_fetches = self._detached_fetches = set()
            detached_fetches.add(pending.future)

            def _free_detached_fetch(_future, device_indices=pending.device_indices):
                try:
                    self.target_cache.release_device_indices_after_fetch_done(
                        _future, device_indices, backend=backend
                    )
                finally:
                    detached_fetches.discard(_future)

            pending.future.add_done_callback(_free_detached_fetch)

    def _submit_direct_transfer(
        self,
        plan: SharedHiCachePlan,
        *,
        start_block: int,
        max_blocks: int,
        token_count: int,
    ) -> tuple[Optional[Future], Optional[torch.Tensor]]:
        direct_transfer = getattr(self, "direct_transfer", None)
        if not self._direct_transfer_enabled():
            return None, None
        endpoints = self._candidate_endpoints_for_plan(plan)
        if not endpoints:
            return None, None
        if not self._try_acquire_fetch_worker():
            return None, None

        device_indices = self.target_cache.alloc_device_indices(token_count)
        if device_indices is None:
            self._release_fetch_worker()
            return None, None

        target_page_indices = self.target_cache.device_indices_to_page_indices(
            device_indices
        )
        if target_page_indices is None:
            logger.warning(
                "Shared HiCache direct transfer got non page-aligned target device allocation"
            )
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            return None, None
        try:
            future = self._fetch_executor.submit(
                self._request_source_transfer,
                transfer_backend=direct_transfer,
                endpoints=endpoints,
                plan=plan,
                start_block=start_block,
                max_blocks=max_blocks,
                target_page_indices=target_page_indices,
            )
        except Exception:
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            raise
        return future, device_indices

    def has_reuse_plan(self, req: "Req") -> bool:
        plan = getattr(req, "shared_hicache_plan", None)
        if not isinstance(plan, SharedHiCachePlan):
            return False
        if self._validate_plan(plan) is not None:
            return False
        return self._direct_transfer_enabled()

    def release_request(self, rid: str) -> None:
        rid = str(rid)
        pending = self._pending_fetches.pop(rid, None)

        if pending is not None:
            self._release_pending_fetch(pending)

        self._finished_plan_keys = {
            key for key in self._finished_plan_keys if key[0] != rid
        }
        self._finished_plan_prefix_lens = {
            key: prefix_len
            for key, prefix_len in self._finished_plan_prefix_lens.items()
            if key[0] != rid
        }

    def prepare_reuse(self, req: "Req") -> SharedHiCacheResult:
        plan = getattr(req, "shared_hicache_plan", None)
        if plan is None:
            return SharedHiCacheResult()
        if not isinstance(plan, SharedHiCachePlan):
            logger.debug(
                "Ignoring invalid shared HiCache plan for rid=%s: expected SharedHiCachePlan got %s",
                req.rid,
                type(plan).__name__,
            )
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason="invalid_plan",
            )
            return SharedHiCacheResult()

        rejection = self._validate_plan(plan)
        if rejection is not None:
            logger.debug(
                "Ignoring shared HiCache plan rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                rejection,
            )
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason=rejection,
            )
            return SharedHiCacheResult()

        plan_key = self._plan_key(req, plan)
        if plan_key in self._finished_plan_keys:
            return SharedHiCacheResult(
                prefix_len=self._finished_plan_prefix_lens.get(plan_key, 0)
            )

        page_size = self.tree_cache.page_size
        matched_tokens = len(req.prefix_indices) + req.host_hit_length
        if matched_tokens % page_size != 0:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=unaligned_matched_tokens matched_tokens=%d page_size=%d",
                req.rid,
                plan.plan_id,
                matched_tokens,
                page_size,
            )
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason="unaligned_matched_tokens",
            )
            return SharedHiCacheResult()
        computed_blocks = matched_tokens // page_size
        if computed_blocks < plan.start_block_index:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=before_plan_start computed_blocks=%d start_block_index=%d",
                req.rid,
                plan.plan_id,
                computed_blocks,
                plan.start_block_index,
            )
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason="before_plan_start",
            )
            return SharedHiCacheResult()

        max_plan_blocks = max(
            self._max_cacheable_blocks(req) - plan.start_block_index, 0
        )
        planned_blocks = min(plan.planned_prefix_blocks, max_plan_blocks)
        plan_offset = computed_blocks - plan.start_block_index
        if planned_blocks <= plan_offset:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=no_remaining_planned_blocks planned_blocks=%d plan_offset=%d max_plan_blocks=%d",
                req.rid,
                plan.plan_id,
                planned_blocks,
                plan_offset,
                max_plan_blocks,
            )
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason="no_remaining_planned_blocks",
            )
            return SharedHiCacheResult()

        pending = self._pending_fetches.get(str(req.rid))
        if pending is not None:
            if pending.plan.plan_id != plan.plan_id:
                self._pending_fetches.pop(str(req.rid), None)
                self._release_pending_fetch(pending)
            elif not pending.future.done():
                stop_waiting, reason = pending_should_stop_waiting(
                    pending,
                    policy=str(getattr(self, "prefetch_stop_policy", "timeout")),
                    page_size=self.tree_cache.page_size,
                    timeout_secs=float(getattr(self, "timeout_secs", 0.0)),
                    # Shared remote transfers use the SharedHiCache timeout.
                    # HiCache storage's canary timeout is tuned for local storage
                    # prefetch and can race slower TP ranks.
                    prefetch_timeout_config=None,
                )
                if stop_waiting:
                    self._pending_fetches.pop(str(req.rid), None)
                    self._release_pending_fetch(pending)
                    self._finished_plan_keys.add(self._plan_key(req, pending.plan))
                    observe_reuse(
                        self.metrics_collector,
                        backend=pending.backend,
                        outcome="miss",
                        reason=reason,
                        wait_ms=pending_wait_ms(pending),
                    )
                    return SharedHiCacheResult()
                return SharedHiCacheResult(pending=True)
            else:
                return self._finish_pending_fetch(req, pending)

        logger.debug(
            "Submitting shared HiCache fetch rid=%s plan_id=%s source=%s:%s start_block=%d max_blocks=%d matched_tokens=%d",
            req.rid,
            plan.plan_id,
            plan.source_worker_id,
            plan.source_attn_dp_rank,
            plan_offset,
            planned_blocks - plan_offset,
            matched_tokens,
        )
        expected_hashes = plan.planned_hashes[plan_offset:planned_blocks]
        max_blocks = planned_blocks - plan_offset
        token_count = max_blocks * page_size
        future = None
        device_indices = None
        direct_transfer_enabled = self._direct_transfer_enabled()
        if not direct_transfer_enabled:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=direct_transfer_unavailable",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(plan_key)
            observe_reuse(
                self.metrics_collector,
                backend="none",
                outcome="miss",
                reason="direct_transfer_unavailable",
            )
            return SharedHiCacheResult()
        if direct_transfer_enabled and req.host_hit_length > 0:
            self._finished_plan_keys.add(plan_key)
            observe_reuse(
                self.metrics_collector,
                backend=self._current_backend_label(),
                outcome="skip",
                reason="local_host_hit",
            )
            return SharedHiCacheResult()
        locked_node = None
        if req.host_hit_length == 0:
            if direct_transfer_enabled:
                locked_node = self._lock_request_prefix(req)
            try:
                future, device_indices = self._submit_direct_transfer(
                    plan,
                    start_block=plan_offset,
                    max_blocks=max_blocks,
                    token_count=token_count,
                )
            except Exception:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                raise
            if direct_transfer_enabled and future is None:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                self._finished_plan_keys.add(plan_key)
                observe_reuse(
                    self.metrics_collector,
                    backend=self._current_backend_label(),
                    outcome="miss",
                    reason="direct_submit_unavailable",
                )
                return SharedHiCacheResult()
        backend = "none"
        bytes_per_page = 0
        if device_indices is not None and direct_transfer_enabled:
            direct_transfer = getattr(self, "direct_transfer", None)
            backend = str(getattr(direct_transfer, "name", "direct"))
            try:
                bytes_per_page = sum(
                    int(length)
                    for length in getattr(direct_transfer, "target_kv_item_lens", [])
                )
            except Exception:
                bytes_per_page = 0
        pending = SharedHiCachePendingFetch(
            plan=plan,
            plan_offset=plan_offset,
            target_start_block=plan.start_block_index + plan_offset,
            expected_hashes=expected_hashes,
            future=future,
            device_indices=device_indices,
            locked_node=locked_node,
            backend=backend,
            bytes_per_page=bytes_per_page,
            submitted_at=time.perf_counter(),
        )
        self._pending_fetches[str(req.rid)] = pending
        future.add_done_callback(
            lambda done_future, pending=pending: self._on_pending_fetch_done(
                pending, done_future
            )
        )
        return SharedHiCacheResult(pending=True)

    def _finish_pending_fetch(
        self, req: "Req", pending: SharedHiCachePendingFetch
    ) -> SharedHiCacheResult:
        self._pending_fetches.pop(str(req.rid), None)
        plan = pending.plan
        if pending.done_at <= 0:
            pending.done_at = time.perf_counter()

        try:
            pages, reason = pending.future.result()
        except Exception:
            logger.exception(
                "Shared HiCache fetch failed rid=%s plan_id=%s", req.rid, plan.plan_id
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self.target_cache.free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            observe_reuse(
                self.metrics_collector,
                backend=pending.backend,
                outcome="error",
                reason="fetch_exception",
                wait_ms=pending_wait_ms(pending),
            )
            return SharedHiCacheResult()

        if not pages:
            logger.debug(
                "Shared HiCache source returned no pages rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                reason,
            )
            self._unlock_pending_prefix(pending)
            indeterminate_transfer = is_indeterminate_direct_transfer_reason(reason)
            if pending.device_indices is not None:
                if indeterminate_transfer:
                    self.target_cache.quarantine_device_indices(
                        pending.device_indices,
                        reason,
                        backend=getattr(
                            pending, "backend", self._current_backend_label()
                        ),
                    )
                else:
                    self.target_cache.free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            observe_reuse(
                self.metrics_collector,
                backend=pending.backend,
                outcome="error" if indeterminate_transfer else "miss",
                reason=reason,
                wait_ms=pending_wait_ms(pending),
            )
            return SharedHiCacheResult()

        if len(pages) > len(pending.expected_hashes):
            logger.warning(
                "Shared HiCache source returned too many pages rid=%s plan_id=%s pages=%d expected=%d",
                req.rid,
                plan.plan_id,
                len(pages),
                len(pending.expected_hashes),
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self.target_cache.free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            observe_reuse(
                self.metrics_collector,
                backend=pending.backend,
                outcome="error",
                reason="too_many_pages",
                wait_ms=pending_wait_ms(pending),
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            return SharedHiCacheResult()

        expected_hashes = pending.expected_hashes[: len(pages)]
        if tuple(page.block_hash for page in pages) != expected_hashes:
            logger.warning(
                "Shared HiCache source returned non-contiguous pages rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self.target_cache.free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            observe_reuse(
                self.metrics_collector,
                backend=pending.backend,
                outcome="error",
                reason="non_contiguous_pages",
                wait_ms=pending_wait_ms(pending),
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            return SharedHiCacheResult()

        insert_start = time.perf_counter()
        try:
            if pending.device_indices is None:
                logger.warning(
                    "Shared HiCache direct transfer completed without target device indices"
                )
                self._finished_plan_keys.add(self._plan_key(req, plan))
                observe_reuse(
                    self.metrics_collector,
                    backend=pending.backend,
                    outcome="error",
                    reason="missing_target_device_indices",
                    wait_ms=pending_wait_ms(pending),
                )
                return SharedHiCacheResult()
            staged_tokens = self.target_cache.insert_device_pages(
                req,
                pages,
                device_indices=pending.device_indices,
                start_block=pending.target_start_block,
            )
        except Exception:
            insert_ms = (time.perf_counter() - insert_start) * 1000
            logger.exception(
                "Shared HiCache insert failed rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(self._plan_key(req, plan))
            observe_reuse(
                self.metrics_collector,
                backend=pending.backend,
                outcome="error",
                reason="insert_exception",
                wait_ms=pending_wait_ms(pending),
                insert_ms=insert_ms,
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            return SharedHiCacheResult()
        finally:
            self._unlock_pending_prefix(pending)
        insert_ms = (time.perf_counter() - insert_start) * 1000
        fetched_tokens = len(pages) * self.tree_cache.page_size
        prefix_len = (
            pending.target_start_block * self.tree_cache.page_size + fetched_tokens
        )
        if staged_tokens > 0:
            req.shared_hicache_hit_length = (
                getattr(req, "shared_hicache_hit_length", 0) + staged_tokens
            )
            wait_ms = pending_wait_ms(pending)
            ready_wait_ms = pending_ready_wait_ms(pending)
            logger.info(
                "Shared HiCache staged %d tokens rid=%s plan_id=%s source=%s:%s fetched_tokens=%d prefix_len=%d wait_ms=%s future_ready_wait_ms=%s insert_ms=%.3f direct=%s",
                staged_tokens,
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                plan.source_attn_dp_rank,
                fetched_tokens,
                prefix_len,
                format_optional_ms(wait_ms),
                format_optional_ms(ready_wait_ms),
                insert_ms,
                pending.device_indices is not None,
            )
        self._finished_plan_keys.add(self._plan_key(req, plan))
        if staged_tokens > 0:
            self._finished_plan_prefix_lens[self._plan_key(req, plan)] = prefix_len
        outcome = "hit" if staged_tokens > 0 else "miss"
        observe_reuse(
            self.metrics_collector,
            backend=pending.backend,
            outcome=outcome,
            reason=reason if staged_tokens > 0 else "insert_returned_zero",
            tokens=staged_tokens,
            wait_ms=pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes_for_pages(pending, pages),
        )
        return SharedHiCacheResult(staged_tokens=staged_tokens, prefix_len=prefix_len)
