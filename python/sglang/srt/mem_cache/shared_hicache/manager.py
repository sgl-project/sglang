from __future__ import annotations

import atexit
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.config import (
    SharedHiCacheConfig,
    shared_hicache_timeout_secs,
)
from sglang.srt.mem_cache.shared_hicache.control import (
    SHARED_HICACHE_TRANSFER_DONE,
    SHARED_HICACHE_TRANSFER_REQUEST,
    SharedHiCacheTargetTransferTracker,
    SharedHiCacheTransferHandle,
    is_indeterminate_direct_transfer_reason,
)
from sglang.srt.mem_cache.shared_hicache.pending import (
    SharedHiCachePendingFetch,
    format_optional_ms,
    pending_ready_wait_ms,
    pending_should_stop_waiting,
    pending_wait_ms,
    transfer_bytes_for_pages,
)
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.service import (
    SharedHiCacheSourceService,
)
from sglang.srt.mem_cache.shared_hicache.source_queue import (
    SharedHiCacheSourceTransferQueue,
)
from sglang.srt.mem_cache.shared_hicache.target import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.topology import (
    SharedHiCacheTopology,
    validate_shared_hicache_plan,
)
from sglang.srt.mem_cache.shared_hicache.transfer import (
    SharedHiCacheTransferBackend,
    make_shared_hicache_transfer_backend,
    scheduler_parallel_metadata,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedHiCacheResult:
    staged_tokens: int = 0
    prefix_len: int = 0
    pending: bool = False


def _shared_hicache_enabled(server_args: "ServerArgs") -> bool:
    return bool(
        getattr(server_args, "enable_shared_hicache", False)
        or getattr(server_args, "shared_hicache_config", None)
    )


class SharedHiCacheManager:
    def __init__(
        self,
        *,
        server_args: "ServerArgs",
        tree_cache,
        worker_id: Optional[int],
        parallel_metadata: Optional[Mapping[str, int]] = None,
        direct_transfer: Optional[SharedHiCacheTransferBackend] = None,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self._set_parallel_metadata(parallel_metadata)
        self.timeout_secs = shared_hicache_timeout_secs(server_args)
        self.prefetch_stop_policy = getattr(
            server_args, "hicache_storage_prefetch_policy", "timeout"
        )
        self.direct_transfer = direct_transfer
        self.metrics_collector = metrics_collector
        if not self._direct_transfer_enabled():
            logger.warning(
                "SharedHiCache is enabled but no direct transfer backend is available; "
                "SharedHiCache plans will be treated as cache misses."
            )
        config = getattr(server_args, "shared_hicache_config", None)
        endpoint_spec = (
            config.control_endpoint if isinstance(config, SharedHiCacheConfig) else None
        )
        self.endpoint = self._format_local_control_endpoint(endpoint_spec)
        self.source_service: Optional[SharedHiCacheSourceService] = None
        self._shutdown = False
        fetch_worker_limit = max(
            1,
            int(envs.SGLANG_SHARED_HICACHE_FETCH_WORKERS.get()),
        )
        self._target_transfer_capacity = threading.BoundedSemaphore(fetch_worker_limit)
        source_worker_limit = fetch_worker_limit
        self._pending_fetches: dict[str, SharedHiCachePendingFetch] = {}
        self.source_transfer_queue: Optional[SharedHiCacheSourceTransferQueue] = None
        self.target_cache = SharedHiCacheTarget(
            tree_cache=tree_cache,
            metrics_collector=metrics_collector,
        )
        self.target_transfer_tracker = SharedHiCacheTargetTransferTracker(
            transfer_backend=direct_transfer,
        )
        self._finished_plan_keys: set[tuple[str, str]] = set()
        self._finished_plan_prefix_lens: dict[tuple[str, str], int] = {}
        self._direct_transfer_shutdown_done = False
        self._direct_transfer_shutdown_deferred = False
        self._direct_transfer_shutdown_lock = threading.Lock()

        if self.endpoint is not None:
            self.source_transfer_queue = SharedHiCacheSourceTransferQueue(
                tree_cache=tree_cache,
                worker_id=worker_id,
                transfer_backend=direct_transfer,
                worker_limit=source_worker_limit,
                send_transfer_done=self._send_transfer_done,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                attn_tp_size=self.attn_tp_size,
                attn_cp_size=self.attn_cp_size,
                attn_dp_size=self.attn_dp_size,
            )
            self.source_service = SharedHiCacheSourceService(
                endpoint=self.endpoint,
                worker_id=self.worker_id,
                handle_control_message=self._handle_control_message,
            )
            self.source_service.start()
        atexit.register(self.shutdown)

    def _set_parallel_metadata(
        self,
        parallel_metadata: Optional[Mapping[str, int]],
    ) -> None:
        self.topology = SharedHiCacheTopology.from_mapping(parallel_metadata)
        for key, value in self.topology.to_dict().items():
            setattr(self, key, value)

    def _endpoint_format_values(self) -> dict[str, int]:
        return self.topology.endpoint_format_values()

    def _format_local_control_endpoint(self, endpoint_spec: object) -> Optional[str]:
        return self.topology.format_local_control_endpoint(endpoint_spec)

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
            config = getattr(server_args, "shared_hicache_config", None)
            if isinstance(config, SharedHiCacheConfig):
                worker_id = config.worker_id
        if worker_id is None:
            logger.warning(
                "SharedHiCache disabled because worker_id is not set; "
                "set --shared-hicache-worker-id"
            )
            return None
        direct_transfer = make_shared_hicache_transfer_backend(scheduler)
        parallel_metadata = scheduler_parallel_metadata(scheduler)
        metrics_reporter = getattr(scheduler, "metrics_reporter", None)
        metrics_collector = (
            scheduler.metrics_collector
            if getattr(metrics_reporter, "enable_metrics", False)
            else None
        )
        return cls(
            server_args=server_args,
            tree_cache=scheduler.tree_cache,
            worker_id=worker_id,
            parallel_metadata=parallel_metadata,
            direct_transfer=direct_transfer,
            metrics_collector=metrics_collector,
        )

    def _current_backend_label(self) -> str:
        if self._direct_transfer_enabled():
            direct_transfer = getattr(self, "direct_transfer", None)
            return str(getattr(direct_transfer, "name", "direct"))
        return "none"

    def _observe_reuse(
        self,
        *,
        backend: str,
        outcome: str,
        reason: str,
        tokens: int = 0,
        wait_ms: Optional[float] = None,
        insert_ms: Optional[float] = None,
        transfer_bytes: Optional[int] = None,
    ) -> None:
        if self.metrics_collector is None:
            return
        self.metrics_collector.observe_shared_hicache(
            backend=backend,
            outcome=outcome,
            reason=reason,
            tokens=max(0, int(tokens)),
            wait_ms=wait_ms,
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes,
        )

    def _direct_transfer_enabled(self) -> bool:
        direct_transfer = getattr(self, "direct_transfer", None)
        return direct_transfer is not None and bool(
            getattr(direct_transfer, "enabled", False)
        )

    def _try_acquire_fetch_worker(self) -> bool:
        semaphore = getattr(self, "_target_transfer_capacity", None)
        if semaphore is None:
            return True
        return semaphore.acquire(blocking=False)

    def _release_fetch_worker(self) -> None:
        semaphore = getattr(self, "_target_transfer_capacity", None)
        if semaphore is None:
            return
        try:
            semaphore.release()
        except ValueError:
            logger.debug(
                "SharedHiCache fetch worker semaphore release ignored", exc_info=True
            )

    def _finish_target_transfer_capacity(self) -> None:
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

        timeout_secs = float(getattr(self, "timeout_secs", 0.0))
        deadline = time.monotonic() + min(max(timeout_secs, 0.0), 5.0)
        while self.has_pending() and time.monotonic() < deadline:
            time.sleep(0.01)

        if self.has_pending():
            logger.warning(
                "Deferring direct transfer backend shutdown while SharedHiCache work is still pending"
            )
            source_transfer_queue = getattr(self, "source_transfer_queue", None)
            if source_transfer_queue is not None:
                source_transfer_queue.shutdown(wait=False, cancel_futures=True)
            self._defer_direct_transfer_shutdown()
            return

        source_transfer_queue = getattr(self, "source_transfer_queue", None)
        if source_transfer_queue is not None:
            source_transfer_queue.shutdown(wait=False, cancel_futures=True)
        self.target_cache.release_quarantined_device_indices()
        self._shutdown_direct_transfer_backend()

    def _candidate_endpoints_for_plan(self, plan: SharedHiCachePlan) -> list[str]:
        return self.topology.candidate_endpoints_for_plan(plan)

    def _send_control_message(self, endpoint: str, payload: Mapping[str, Any]) -> None:
        source_service = getattr(self, "source_service", None)
        if source_service is None:
            raise RuntimeError("SharedHiCache ZMQ control service is not running")
        source_service.send(endpoint, payload)

    def _handle_control_message(self, payload: Mapping[str, Any]) -> None:
        kind = str(payload.get("kind") or "")
        if kind == SHARED_HICACHE_TRANSFER_REQUEST:
            source_transfer_queue = getattr(self, "source_transfer_queue", None)
            if source_transfer_queue is None:
                transfer_id = str(payload.get("transfer_id") or "")
                response = {
                    "ok": False,
                    "reason": "source_transfer_queue_unavailable",
                    "transfer_id": transfer_id,
                    "transferred_blocks": 0,
                }
            else:
                response = source_transfer_queue.handle(payload)
            if not response.get("accepted"):
                target_endpoint = str(payload.get("target_control_endpoint") or "")
                if target_endpoint:
                    self._send_transfer_done(target_endpoint, response)
            return
        if kind == SHARED_HICACHE_TRANSFER_DONE:
            self._handle_target_transfer_done(payload)
            return
        logger.warning("Ignoring unknown SharedHiCache control message kind=%s", kind)

    def _send_transfer_done(self, endpoint: str, payload: Mapping[str, Any]) -> None:
        message = dict(payload)
        message["kind"] = SHARED_HICACHE_TRANSFER_DONE
        message["pending"] = False
        try:
            self._send_control_message(endpoint, message)
        except Exception:
            logger.warning(
                "SharedHiCache failed to send transfer completion endpoint=%s transfer_id=%s",
                endpoint,
                message.get("transfer_id"),
                exc_info=True,
            )

    def _handle_target_transfer_done(self, payload: Mapping[str, Any]) -> None:
        self.target_transfer_tracker.handle_done(payload)

    def _pop_target_transfer_completion(
        self, transfer_id: str
    ) -> Optional[Mapping[str, Any]]:
        return self.target_transfer_tracker.pop_completion(transfer_id)

    def _start_target_transfer(self, transfer_id: str) -> None:
        self.target_transfer_tracker.start(transfer_id)

    def _finish_target_transfer(self, transfer_id: str) -> None:
        self.target_transfer_tracker.finish(transfer_id)

    def _max_cacheable_blocks(self, req: "Req") -> int:
        max_prefix_len = max(len(req.fill_ids) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if req.positional_embed_overrides is not None:
            max_prefix_len = 0
        return max_prefix_len // self.tree_cache.page_size

    def _validate_plan(self, plan: SharedHiCachePlan) -> Optional[str]:
        return validate_shared_hicache_plan(
            plan,
            worker_id=self.worker_id,
            page_size=self.tree_cache.page_size,
            topology=self.topology,
        )

    def _plan_key(self, req: "Req", plan: SharedHiCachePlan) -> tuple[str, str]:
        return str(req.rid), plan.plan_id

    def has_pending(self) -> bool:
        source_transfer_queue = getattr(self, "source_transfer_queue", None)
        source_transfer_count = (
            source_transfer_queue.active_count()
            if source_transfer_queue is not None
            else 0
        )
        source_service = getattr(self, "source_service", None)
        active_source_count = (
            source_service.active_count() if source_service is not None else 0
        )
        return (
            bool(getattr(self, "_pending_fetches", {}))
            or source_transfer_count > 0
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
        self._unlock_pending_prefix(pending)
        if pending.device_indices is None:
            transfer_id = getattr(pending.transfer, "transfer_id", None)
            if transfer_id:
                self._finish_target_transfer(transfer_id)
            self._finish_target_transfer_capacity()
            return
        backend = getattr(pending, "backend", self._current_backend_label())
        transfer = pending.transfer
        if transfer.done():
            _, reason = transfer.result()
            transfer_id = getattr(transfer, "transfer_id", None)
            if transfer_id:
                self._finish_target_transfer(transfer_id)
            if is_indeterminate_direct_transfer_reason(reason):
                self.target_cache.quarantine_device_indices(
                    pending.device_indices, reason, backend=backend
                )
            else:
                self.target_cache.free_device_indices(pending.device_indices)
        else:
            transfer_id = getattr(transfer, "transfer_id", None)
            if transfer_id:
                self._finish_target_transfer(transfer_id)
            self.target_cache.quarantine_device_indices(
                pending.device_indices,
                SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
                backend=backend,
            )
        self._finish_target_transfer_capacity()

    def _submit_direct_transfer(
        self,
        plan: SharedHiCachePlan,
        *,
        start_block: int,
        max_blocks: int,
        token_count: int,
    ) -> tuple[
        Optional[SharedHiCacheTransferHandle],
        Optional[torch.Tensor],
        Optional[str],
    ]:
        direct_transfer = getattr(self, "direct_transfer", None)
        if not self._direct_transfer_enabled():
            return None, None, "direct_transfer_unavailable"
        endpoints = self._candidate_endpoints_for_plan(plan)
        if not endpoints:
            return None, None, "source_endpoint_unavailable"
        if not self._try_acquire_fetch_worker():
            return None, None, "fetch_worker_unavailable"

        device_indices = self.target_cache.alloc_device_indices(token_count)
        if device_indices is None:
            self._release_fetch_worker()
            return None, None, "target_staging_alloc_failed"

        target_page_indices = self.target_cache.device_indices_to_page_indices(
            device_indices
        )
        if target_page_indices is None:
            logger.warning(
                "Shared HiCache direct transfer got non page-aligned target device allocation"
            )
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            return None, None, "target_page_alignment_failed"
        transfer_id = uuid.uuid4().hex
        handle = SharedHiCacheTransferHandle(
            transfer_backend=direct_transfer,
            transfer_id=transfer_id,
            plan=plan,
            start_block=start_block,
            max_blocks=max_blocks,
            timeout_secs=self.timeout_secs,
            pop_source_completion=self._pop_target_transfer_completion,
        )
        target_descriptor = direct_transfer.target_descriptor()
        target_page_indices_payload = [
            int(index)
            for index in (
                target_page_indices.detach().cpu().tolist()
                if hasattr(target_page_indices, "detach")
                else list(target_page_indices)
            )
        ]
        self._start_target_transfer(transfer_id)
        try:
            self._send_control_message(
                endpoints[0],
                {
                    "kind": SHARED_HICACHE_TRANSFER_REQUEST,
                    "transfer_id": transfer_id,
                    "target_control_endpoint": self.endpoint,
                    "plan": plan.to_dict(),
                    "start_block": start_block,
                    "max_blocks": max_blocks,
                    "target_session_id": direct_transfer.target_session_id,
                    "transfer_backend": direct_transfer.name,
                    "target_metadata": target_descriptor,
                    "target_kv_ptrs": direct_transfer.target_kv_ptrs,
                    "target_kv_item_lens": direct_transfer.target_kv_item_lens,
                    "target_page_indices": target_page_indices_payload,
                },
            )
        except Exception:
            self._finish_target_transfer(transfer_id)
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            logger.warning(
                "Shared HiCache direct transfer control send failed plan_id=%s source_worker=%s endpoint=%s",
                plan.plan_id,
                plan.source_worker_id,
                endpoints[0],
                exc_info=True,
            )
            return None, None, "control_send_failed"
        return handle, device_indices, None

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
            self._observe_reuse(
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
            self._observe_reuse(
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
            self._observe_reuse(
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
            self._observe_reuse(
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
            self._observe_reuse(
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
            elif pending.transfer.poll() not in (KVPoll.Success, KVPoll.Failed):
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
                    self._observe_reuse(
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
            "Submitting shared HiCache fetch rid=%s plan_id=%s source_worker=%s start_block=%d max_blocks=%d matched_tokens=%d",
            req.rid,
            plan.plan_id,
            plan.source_worker_id,
            plan_offset,
            planned_blocks - plan_offset,
            matched_tokens,
        )
        expected_hashes = plan.planned_hashes[plan_offset:planned_blocks]
        max_blocks = planned_blocks - plan_offset
        token_count = max_blocks * page_size
        transfer = None
        device_indices = None
        direct_submit_reason = None
        direct_transfer_enabled = self._direct_transfer_enabled()
        if not direct_transfer_enabled:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=direct_transfer_unavailable",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
                backend="none",
                outcome="miss",
                reason="direct_transfer_unavailable",
            )
            return SharedHiCacheResult()
        if direct_transfer_enabled and req.host_hit_length > 0:
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
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
                transfer, device_indices, direct_submit_reason = (
                    self._submit_direct_transfer(
                        plan,
                        start_block=plan_offset,
                        max_blocks=max_blocks,
                        token_count=token_count,
                    )
                )
            except Exception:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                raise
            if direct_transfer_enabled and transfer is None:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                self._finished_plan_keys.add(plan_key)
                direct_submit_reason = (
                    direct_submit_reason or "direct_submit_unavailable"
                )
                logger.info(
                    "Shared HiCache direct submit unavailable rid=%s plan_id=%s reason=%s source_worker=%s start_block=%d max_blocks=%d token_count=%d host_hit_length=%d",
                    req.rid,
                    plan.plan_id,
                    direct_submit_reason,
                    plan.source_worker_id,
                    plan_offset,
                    max_blocks,
                    token_count,
                    req.host_hit_length,
                )
                self._observe_reuse(
                    backend=self._current_backend_label(),
                    outcome="miss",
                    reason=direct_submit_reason,
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
            transfer=transfer,
            device_indices=device_indices,
            locked_node=locked_node,
            backend=backend,
            bytes_per_page=bytes_per_page,
            submitted_at=time.perf_counter(),
        )
        self._pending_fetches[str(req.rid)] = pending
        return SharedHiCacheResult(pending=True)

    def _finish_pending_fetch(
        self, req: "Req", pending: SharedHiCachePendingFetch
    ) -> SharedHiCacheResult:
        self._pending_fetches.pop(str(req.rid), None)
        plan = pending.plan
        if pending.done_at <= 0:
            pending.done_at = (
                float(getattr(pending.transfer, "done_at", 0.0)) or time.perf_counter()
            )

        try:
            pages, reason = pending.transfer.result()
        except Exception:
            transfer_id = getattr(pending.transfer, "transfer_id", None)
            if transfer_id:
                self._finish_target_transfer(transfer_id)
            logger.exception(
                "Shared HiCache fetch failed rid=%s plan_id=%s", req.rid, plan.plan_id
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self.target_cache.free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="fetch_exception",
                wait_ms=pending_wait_ms(pending),
            )
            self._finish_target_transfer_capacity()
            return SharedHiCacheResult()
        transfer_id = getattr(pending.transfer, "transfer_id", None)
        if transfer_id:
            self._finish_target_transfer(transfer_id)

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
            self._observe_reuse(
                backend=pending.backend,
                outcome="error" if indeterminate_transfer else "miss",
                reason=reason,
                wait_ms=pending_wait_ms(pending),
            )
            self._finish_target_transfer_capacity()
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
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="too_many_pages",
                wait_ms=pending_wait_ms(pending),
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            self._finish_target_transfer_capacity()
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
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="non_contiguous_pages",
                wait_ms=pending_wait_ms(pending),
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            self._finish_target_transfer_capacity()
            return SharedHiCacheResult()

        insert_start = time.perf_counter()
        try:
            if pending.device_indices is None:
                logger.warning(
                    "Shared HiCache direct transfer completed without target device indices"
                )
                self._finished_plan_keys.add(self._plan_key(req, plan))
                self._observe_reuse(
                    backend=pending.backend,
                    outcome="error",
                    reason="missing_target_device_indices",
                    wait_ms=pending_wait_ms(pending),
                )
                self._finish_target_transfer_capacity()
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
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="insert_exception",
                wait_ms=pending_wait_ms(pending),
                insert_ms=insert_ms,
                transfer_bytes=transfer_bytes_for_pages(pending, pages),
            )
            self._finish_target_transfer_capacity()
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
                "Shared HiCache staged %d tokens rid=%s plan_id=%s source_worker=%s fetched_tokens=%d prefix_len=%d wait_ms=%s ready_wait_ms=%s insert_ms=%.3f direct=%s",
                staged_tokens,
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
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
        self._observe_reuse(
            backend=pending.backend,
            outcome=outcome,
            reason=reason if staged_tokens > 0 else "insert_returned_zero",
            tokens=staged_tokens,
            wait_ms=pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes_for_pages(pending, pages),
        )
        self._finish_target_transfer_capacity()
        return SharedHiCacheResult(staged_tokens=staged_tokens, prefix_len=prefix_len)
