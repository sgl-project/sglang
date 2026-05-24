from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Protocol
from urllib.parse import urlparse

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.g2plus_transfer import (
    G2plusTransferBackend,
    g2plus_config,
    g2plus_config_value,
    g2plus_timeout_secs,
    make_g2plus_transfer_backend,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.router_kv_control import (
    is_indeterminate_direct_transfer_reason,
    request_source_transfer,
    start_source_transfer_server,
)
from sglang.srt.mem_cache.router_kv_plan import (
    REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY as _NO_PLAN_REASON_KEY,
    REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY as _PLAN_EXTRA_ARGS_KEY,
    REMOTE_KV_REUSE_PLAN_VERSION as _PLAN_VERSION,
    RemoteKvReusePlan,
    normalize_endpoint,
)
from sglang.srt.mem_cache.router_kv_source import (
    ResolvedHostPage,
    handle_source_transfer,
    resolve_host_pages as _resolve_host_pages,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY = _PLAN_EXTRA_ARGS_KEY
REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY = _NO_PLAN_REASON_KEY
REMOTE_KV_REUSE_PLAN_VERSION = _PLAN_VERSION
REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES = 16 * 1024 * 1024
resolve_host_pages = _resolve_host_pages


def _normalize_metric_label(value: Any, default: str = "unknown") -> str:
    value = str(value or default).split(":", 1)[0].strip().lower()
    chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in value]
    value = "".join(chars).strip("_")
    return (value or default)[:80]


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{field_name} must be an integer, got empty string")
        try:
            return int(value, 10)
        except ValueError as err:
            raise ValueError(
                f"{field_name} must be an integer, got {value!r}"
            ) from err
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _normalize_endpoint(endpoint: str) -> str:
    return normalize_endpoint(endpoint)


@dataclass(frozen=True)
class RouterKVReuseResult:
    staged_tokens: int = 0
    pending: bool = False


@dataclass
class _RemoteG2PendingFetch:
    plan: RemoteKvReusePlan
    plan_offset: int
    target_start_block: int
    expected_hashes: tuple[int, ...]
    future: Future
    device_indices: Optional[torch.Tensor] = None
    locked_node: Optional[TreeNode] = None
    backend: str = "unknown"
    bytes_per_page: int = 0
    submitted_at: float = 0.0


def _format_optional_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _is_indeterminate_direct_transfer_reason(reason: str) -> bool:
    return is_indeterminate_direct_transfer_reason(reason)


def _endpoint_to_bind(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(_normalize_endpoint(endpoint))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported remote G2 endpoint scheme: {parsed.scheme}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"remote G2 endpoint must include host and port: {endpoint}")
    return parsed.hostname, parsed.port


def _select_dp_endpoint(endpoint_spec: object, dp_rank: int) -> Optional[str]:
    if not endpoint_spec:
        return None
    if isinstance(endpoint_spec, Mapping):
        endpoint = endpoint_spec.get(str(dp_rank))
        if endpoint is None:
            return None
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(
                "g2plus_config.control.endpoint values must be non-empty strings; "
                f"got {endpoint!r} for dp_rank={dp_rank}"
            )
        return _normalize_endpoint(endpoint)
    if not isinstance(endpoint_spec, str):
        raise ValueError(
            "g2plus_config.control.endpoint must be a string or JSON object"
        )
    spec = endpoint_spec.strip()
    if not spec:
        return None
    if spec.startswith("{"):
        endpoints = json.loads(spec)
        if not isinstance(endpoints, Mapping):
            raise ValueError("g2plus_config.control.endpoint must be a JSON object")
        endpoint = endpoints.get(str(dp_rank))
        if endpoint is None:
            return None
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(
                "g2plus_config.control.endpoint values must be non-empty strings; "
                f"got {endpoint!r} for dp_rank={dp_rank}"
            )
        return _normalize_endpoint(endpoint)
    if "{dp_rank}" in spec:
        spec = spec.format(dp_rank=dp_rank)
    return _normalize_endpoint(spec)


def _router_kv_reuse_enabled(server_args: "ServerArgs") -> bool:
    config = g2plus_config(server_args)
    return bool(
        getattr(server_args, "enable_router_kv_reuse", False)
        or config
        or getattr(server_args, "enable_g2plus", False)
    )


class RouterKVReuseHandler(Protocol):
    def has_reuse_plan(self, req: "Req") -> bool: ...

    def has_pending(self) -> bool: ...

    def prefetch_remote_prefix(self, req: "Req") -> RouterKVReuseResult: ...

    def check_remote_prefix(self, req: "Req") -> RouterKVReuseResult: ...

    def maybe_stage_remote_prefix(self, req: "Req") -> int: ...

    def release_request(self, rid: str) -> None: ...


class RemoteG2ReuseHandler:
    def __init__(
        self,
        *,
        server_args: "ServerArgs",
        tree_cache,
        worker_id: Optional[int],
        dp_rank: int,
        direct_transfer: Optional[G2plusTransferBackend] = None,
        direct_transfer_diagnostics: Optional[list[str]] = None,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self.dp_rank = dp_rank
        self.timeout_secs = g2plus_timeout_secs(server_args)
        self.prefetch_stop_policy = getattr(
            server_args, "hicache_storage_prefetch_policy", "timeout"
        )
        self.prefetch_timeout_config = getattr(
            tree_cache, "prefetch_timeout_config", None
        )
        self.direct_transfer = direct_transfer
        self.metrics_collector = metrics_collector
        self.control_backend = str(
            g2plus_config_value(server_args, "control_backend", "router")
        ).lower()
        if not self._direct_transfer_enabled():
            diagnostic_suffix = ""
            if direct_transfer_diagnostics:
                diagnostic_suffix = (
                    " Diagnostics: " + "; ".join(direct_transfer_diagnostics)
                )
            logger.warning(
                "Router KV reuse is enabled but no direct G2plus transfer backend "
                "is available; remote KV reuse plans will be treated as cache misses.%s",
                diagnostic_suffix,
            )
        endpoint_spec = g2plus_config_value(
            server_args,
            "control_endpoint",
            None,
        )
        self.endpoint = _select_dp_endpoint(endpoint_spec, dp_rank)
        self._source_server: Optional[Any] = None
        self._source_thread: Optional[threading.Thread] = None
        self._shutdown = False
        worker_limit = max(
            1,
            int(envs.SGLANG_G2PLUS_FETCH_WORKERS.get()),
        )
        self._source_activity_lock = threading.Lock()
        self._active_source_resolver_ops = 0
        self._source_resolver_semaphore = threading.BoundedSemaphore(worker_limit)
        self._fetch_semaphore = threading.BoundedSemaphore(worker_limit)
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix=f"g2plus-fetch-dp{dp_rank}",
        )
        self._pending_fetches: dict[str, _RemoteG2PendingFetch] = {}
        self._detached_fetches: set[Future] = set()
        self._quarantined_device_indices: list[torch.Tensor] = []
        self._quarantined_tokens_by_backend: dict[str, int] = {}
        self._finished_plan_keys: set[tuple[str, str]] = set()
        self.max_control_body_bytes = REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES
        self._direct_transfer_shutdown_done = False
        self._direct_transfer_shutdown_deferred = False
        self._direct_transfer_shutdown_lock = threading.Lock()

        if self.endpoint is not None:
            self._start_source_resolver()
        atexit.register(self.shutdown)

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["RemoteG2ReuseHandler"]:
        server_args = scheduler.server_args
        if not _router_kv_reuse_enabled(server_args):
            return None
        if not scheduler.enable_hierarchical_cache:
            logger.warning(
                "Router KV reuse disabled because hierarchical cache is not enabled"
            )
            return None
        if not hasattr(scheduler.tree_cache, "_insert_helper_host"):
            logger.warning(
                "Router KV reuse disabled because the active tree cache does not support host inserts"
            )
            return None
        worker_id = g2plus_config_value(
            server_args, "worker_id", getattr(server_args, "g2plus_worker_id", None)
        )
        if worker_id is None:
            logger.warning(
                "Router KV reuse disabled because G2plus worker identity is not set; "
                "set worker_id in --g2plus-config when enabling G2plus remote KV reuse"
            )
            return None
        transfer_diagnostics: list[str] = []
        direct_transfer = make_g2plus_transfer_backend(
            scheduler, diagnostics=transfer_diagnostics
        )
        return cls(
            server_args=server_args,
            tree_cache=scheduler.tree_cache,
            worker_id=worker_id,
            dp_rank=scheduler.dp_rank or 0,
            direct_transfer=direct_transfer,
            direct_transfer_diagnostics=transfer_diagnostics,
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
        metrics_collector = getattr(self, "metrics_collector", None)
        observe = getattr(metrics_collector, "observe_router_kv_reuse", None)
        if observe is None:
            return
        try:
            observe(
                backend=_normalize_metric_label(backend),
                outcome=_normalize_metric_label(outcome),
                reason=_normalize_metric_label(reason),
                tokens=max(0, int(tokens)),
                wait_ms=wait_ms,
                insert_ms=insert_ms,
                transfer_bytes=transfer_bytes,
            )
        except Exception:
            logger.debug("Failed to record router KV reuse metrics", exc_info=True)

    def _pending_wait_ms(self, pending: _RemoteG2PendingFetch) -> Optional[float]:
        submitted_at = getattr(pending, "submitted_at", 0.0)
        if submitted_at <= 0:
            return None
        return (time.perf_counter() - submitted_at) * 1000

    def _pending_timeout_secs(self, pending: _RemoteG2PendingFetch) -> float:
        cfg = getattr(self, "prefetch_timeout_config", None)
        if cfg is None:
            return float(getattr(self, "timeout_secs", 0.0))
        num_tokens = len(pending.expected_hashes) * self.tree_cache.page_size
        return float(min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024))

    def _pending_should_stop_waiting(
        self, pending: _RemoteG2PendingFetch
    ) -> tuple[bool, str]:
        policy = str(getattr(self, "prefetch_stop_policy", "timeout"))
        if policy == "best_effort":
            return True, "best_effort_incomplete"
        if policy == "wait_complete":
            return False, ""
        if policy == "timeout":
            timeout_secs = self._pending_timeout_secs(pending)
            elapsed = time.perf_counter() - pending.submitted_at
            if timeout_secs >= 0 and elapsed > timeout_secs:
                return True, "prefetch_timeout"
            return False, ""
        return True, "unknown_prefetch_policy"

    def _pending_ready_wait_ms(
        self, pending: _RemoteG2PendingFetch
    ) -> Optional[float]:
        done_at = getattr(pending.future, "_g2plus_done_at", 0.0)
        if done_at <= 0:
            return None
        return max(0.0, (time.perf_counter() - done_at) * 1000)

    def _transfer_bytes_for_pages(
        self, pending: _RemoteG2PendingFetch, pages: list[ResolvedHostPage]
    ) -> int:
        bytes_per_page = int(getattr(pending, "bytes_per_page", 0) or 0)
        if bytes_per_page > 0:
            return len(pages) * bytes_per_page
        return sum(len(page.data) for page in pages)

    def _max_control_body_bytes(self) -> int:
        return int(
            getattr(
                self,
                "max_control_body_bytes",
                REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES,
            )
        )

    def _try_enter_source_resolver(self) -> bool:
        if not self._source_resolver_semaphore.acquire(blocking=False):
            return False
        if not hasattr(self, "_source_activity_lock"):
            self._source_activity_lock = threading.Lock()
            self._active_source_resolver_ops = 0
        with self._source_activity_lock:
            self._active_source_resolver_ops += 1
        return True

    def _exit_source_resolver(self) -> None:
        if not hasattr(self, "_source_activity_lock"):
            self._source_activity_lock = threading.Lock()
            self._active_source_resolver_ops = 0
        with self._source_activity_lock:
            self._active_source_resolver_ops -= 1
        self._source_resolver_semaphore.release()

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
            logger.debug("G2plus fetch worker semaphore release ignored", exc_info=True)

    def _on_fetch_worker_done(self, future: Future) -> None:
        try:
            setattr(future, "_g2plus_done_at", time.perf_counter())
        finally:
            self._release_fetch_worker()

    def _start_source_resolver(self) -> None:
        host, port = _endpoint_to_bind(self.endpoint)
        self._source_server, self._source_thread = start_source_transfer_server(
            host=host,
            port=port,
            endpoint=self.endpoint,
            worker_id=self.worker_id,
            dp_rank=self.dp_rank,
            max_body_bytes=self._max_control_body_bytes,
            try_enter=self._try_enter_source_resolver,
            exit_resolver=self._exit_source_resolver,
            direct_transfer_enabled=self._direct_transfer_enabled,
            handle_source_transfer=self._handle_source_transfer,
        )

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
                self._release_quarantined_device_indices()
                self._shutdown_direct_transfer_backend()
            except Exception:
                logger.warning(
                    "G2plus deferred direct transfer backend shutdown failed",
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_wait_for_pending_and_shutdown,
            name="g2plus-direct-transfer-shutdown",
            daemon=True,
        )
        self._direct_transfer_shutdown_thread = thread
        thread.start()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        server = self._source_server
        self._source_server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                logger.debug("Remote G2 source resolver shutdown failed", exc_info=True)

        source_thread = self._source_thread
        self._source_thread = None
        if (
            source_thread is not None
            and source_thread is not threading.current_thread()
        ):
            try:
                source_thread.join(timeout=1)
            except Exception:
                logger.debug("Remote G2 source resolver join failed", exc_info=True)

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
                "Deferring direct transfer backend shutdown while G2plus work is still pending"
            )
            self._defer_direct_transfer_shutdown()
            return

        self._release_quarantined_device_indices()
        self._shutdown_direct_transfer_backend()

    def _candidate_endpoints_for_plan(self, plan: RemoteKvReusePlan) -> list[str]:
        endpoints: list[str] = []

        def add(endpoint: Optional[str]) -> None:
            if endpoint and endpoint not in endpoints:
                endpoints.append(endpoint)

        add(plan.source_endpoint)
        return endpoints

    def _request_source_transfer(
        self,
        *,
        transfer_backend: G2plusTransferBackend,
        endpoints: list[str],
        plan: RemoteKvReusePlan,
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
            dp_rank=getattr(self, "dp_rank", 0),
        )

    def _max_cacheable_blocks(self, req: "Req") -> int:
        max_prefix_len = max(len(req.fill_ids) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if req.positional_embed_overrides is not None:
            max_prefix_len = 0
        return max_prefix_len // self.tree_cache.page_size

    def _validate_plan(self, plan: RemoteKvReusePlan) -> Optional[str]:
        if self.worker_id is None:
            return "missing_worker_id"
        if plan.target_worker_id != self.worker_id:
            return "wrong_target_worker"
        if plan.target_dp_rank != self.dp_rank:
            return "wrong_target_dp_rank"
        if (
            plan.source_worker_id == plan.target_worker_id
            and plan.source_dp_rank == plan.target_dp_rank
        ):
            return "source_is_target"
        if plan.plan_version != REMOTE_KV_REUSE_PLAN_VERSION:
            return "unsupported_plan_version"
        if plan.is_expired():
            return "plan_expired"
        if not plan.is_remote_g2():
            return "unsupported_source_tier"
        if plan.block_size_tokens != self.tree_cache.page_size:
            return "incompatible_block_size"
        return None

    def _plan_key(self, req: "Req", plan: RemoteKvReusePlan) -> tuple[str, str]:
        return str(req.rid), plan.plan_id

    def _alloc_device_indices(self, token_count: int) -> Optional[torch.Tensor]:
        allocator = self.tree_cache.cache_controller.mem_pool_device_allocator
        device_indices = allocator.alloc(token_count)
        if device_indices is None:
            self.tree_cache.evict(EvictParams(num_tokens=token_count))
            device_indices = allocator.alloc(token_count)
        if device_indices is None:
            logger.warning("Remote G2 failed to allocate %d device tokens", token_count)
        return device_indices

    def _free_device_indices(self, device_indices: Optional[torch.Tensor]) -> None:
        if device_indices is None:
            return
        self.tree_cache.cache_controller.mem_pool_device_allocator.free(device_indices)

    def _observe_quarantine(
        self, *, backend: str, reason: str, tokens: int, current_tokens: int
    ) -> None:
        metrics_collector = getattr(self, "metrics_collector", None)
        observe = getattr(
            metrics_collector, "observe_router_kv_reuse_quarantine", None
        )
        if observe is None:
            return
        try:
            observe(
                backend=_normalize_metric_label(backend),
                reason=_normalize_metric_label(reason),
                tokens=max(0, int(tokens)),
                current_tokens=max(0, int(current_tokens)),
            )
        except Exception:
            logger.debug(
                "Failed to record router KV reuse quarantine metrics", exc_info=True
            )

    def _quarantine_device_indices(
        self, device_indices: torch.Tensor, reason: str, *, backend: str
    ) -> None:
        quarantined = getattr(self, "_quarantined_device_indices", None)
        if quarantined is None:
            quarantined = self._quarantined_device_indices = []
        quarantined.append(device_indices)
        backend_label = _normalize_metric_label(backend)
        token_count = int(device_indices.numel())
        tokens_by_backend = getattr(self, "_quarantined_tokens_by_backend", None)
        if tokens_by_backend is None:
            tokens_by_backend = self._quarantined_tokens_by_backend = {}
        current_tokens = int(tokens_by_backend.get(backend_label, 0)) + token_count
        tokens_by_backend[backend_label] = current_tokens
        self._observe_quarantine(
            backend=backend_label,
            reason=reason,
            tokens=token_count,
            current_tokens=current_tokens,
        )
        logger.error(
            "Quarantining %d G2plus target KV indices after indeterminate direct transfer: %s",
            token_count,
            reason,
        )

    def _release_quarantined_device_indices(self) -> None:
        quarantined = getattr(self, "_quarantined_device_indices", None)
        if not quarantined:
            return
        self._quarantined_device_indices = []
        tokens_by_backend = getattr(self, "_quarantined_tokens_by_backend", {})
        self._quarantined_tokens_by_backend = {}
        for device_indices in quarantined:
            self._free_device_indices(device_indices)
        for backend in tokens_by_backend:
            self._observe_quarantine(
                backend=backend,
                reason="released",
                tokens=0,
                current_tokens=0,
            )

    def _device_indices_to_page_indices(
        self, device_indices: torch.Tensor
    ) -> Optional[list[int]]:
        page_size = self.tree_cache.page_size
        indices = device_indices.detach().cpu().numpy()
        if len(indices) == 0 or len(indices) % page_size != 0:
            return None

        page_rows = indices.reshape(-1, page_size)
        starts = page_rows[:, 0]
        offsets = np.arange(page_size, dtype=page_rows.dtype)
        if np.any(starts % page_size != 0) or np.any(
            page_rows != starts[:, None] + offsets[None, :]
        ):
            return None

        page_indices = starts // page_size
        if np.any(page_indices < 0) or np.any(page_indices > np.iinfo(np.int32).max):
            return None
        return page_indices.astype(np.int32, copy=False).tolist()

    def _active_source_resolver_count(self) -> int:
        lock = getattr(self, "_source_activity_lock", None)
        if lock is None:
            return int(getattr(self, "_active_source_resolver_ops", 0))
        with lock:
            return int(getattr(self, "_active_source_resolver_ops", 0))

    def has_pending(self) -> bool:
        detached_fetches = getattr(self, "_detached_fetches", set())
        for future in list(detached_fetches):
            if future.done():
                detached_fetches.discard(future)
        return (
            bool(getattr(self, "_pending_fetches", {}))
            or bool(detached_fetches)
            or self._active_source_resolver_count() > 0
        )

    def _lock_request_prefix(self, req: "Req") -> Optional[TreeNode]:
        last_node = getattr(req, "last_node", None)
        if last_node is None or last_node is self.tree_cache.root_node:
            return None
        self.tree_cache.inc_lock_ref(last_node)
        return last_node

    def _unlock_pending_prefix(self, pending: _RemoteG2PendingFetch) -> None:
        locked_node = getattr(pending, "locked_node", None)
        if locked_node is None:
            return
        pending.locked_node = None
        self.tree_cache.dec_lock_ref(locked_node)

    def _release_pending_fetch(self, pending: _RemoteG2PendingFetch) -> None:
        cancelled = pending.future.cancel()
        self._unlock_pending_prefix(pending)
        if pending.device_indices is None:
            return
        backend = getattr(pending, "backend", self._current_backend_label())
        if cancelled:
            self._free_device_indices(pending.device_indices)
        elif pending.future.done():
            self._release_device_indices_after_fetch_done(
                pending.future, pending.device_indices, backend=backend
            )
        else:
            detached_fetches = getattr(self, "_detached_fetches", None)
            if detached_fetches is None:
                detached_fetches = self._detached_fetches = set()
            detached_fetches.add(pending.future)

            def _free_detached_fetch(_future, device_indices=pending.device_indices):
                try:
                    self._release_device_indices_after_fetch_done(
                        _future, device_indices, backend=backend
                    )
                finally:
                    detached_fetches.discard(_future)

            pending.future.add_done_callback(_free_detached_fetch)

    def _release_device_indices_after_fetch_done(
        self, future: Future, device_indices: torch.Tensor, *, backend: str
    ) -> None:
        try:
            pages, reason = future.result()
        except Exception:
            self._free_device_indices(device_indices)
            return
        if not pages and _is_indeterminate_direct_transfer_reason(reason):
            self._quarantine_device_indices(device_indices, reason, backend=backend)
            return
        self._free_device_indices(device_indices)

    def _submit_direct_transfer(
        self,
        plan: RemoteKvReusePlan,
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

        device_indices = self._alloc_device_indices(token_count)
        if device_indices is None:
            self._release_fetch_worker()
            return None, None

        target_page_indices = self._device_indices_to_page_indices(device_indices)
        if target_page_indices is None:
            logger.warning(
                "Remote G2 direct transfer got non page-aligned target device allocation"
            )
            self._free_device_indices(device_indices)
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
            self._free_device_indices(device_indices)
            self._release_fetch_worker()
            raise
        future.add_done_callback(self._on_fetch_worker_done)
        return future, device_indices

    def maybe_stage_remote_prefix(self, req: "Req") -> int:
        return self.check_remote_prefix(req).staged_tokens

    def has_reuse_plan(self, req: "Req") -> bool:
        plan_data = getattr(req, "remote_kv_reuse_plan", None)
        if plan_data is None:
            return False
        try:
            plan = RemoteKvReusePlan.from_dict(plan_data)
        except ValueError:
            return False
        if self._validate_plan(plan) is not None:
            return False
        return self._direct_transfer_enabled()

    def prefetch_remote_prefix(self, req: "Req") -> RouterKVReuseResult:
        return RouterKVReuseResult()

    def release_request(self, rid: str) -> None:
        rid = str(rid)
        pending = self._pending_fetches.pop(rid, None)

        if pending is not None:
            self._release_pending_fetch(pending)

        self._finished_plan_keys = {
            key for key in self._finished_plan_keys if key[0] != rid
        }

    def check_remote_prefix(self, req: "Req") -> RouterKVReuseResult:
        plan_data = getattr(req, "remote_kv_reuse_plan", None)
        if plan_data is None:
            return RouterKVReuseResult()

        try:
            plan = RemoteKvReusePlan.from_dict(plan_data)
        except ValueError as err:
            logger.debug("Ignoring invalid remote G2 plan for rid=%s: %s", req.rid, err)
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="invalid_plan",
            )
            return RouterKVReuseResult()

        rejection = self._validate_plan(plan)
        if rejection is not None:
            logger.debug(
                "Ignoring remote G2 plan rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                rejection,
            )
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason=rejection,
            )
            return RouterKVReuseResult()

        plan_key = self._plan_key(req, plan)
        if plan_key in self._finished_plan_keys:
            return RouterKVReuseResult()

        page_size = self.tree_cache.page_size
        matched_tokens = len(req.prefix_indices) + req.host_hit_length
        if matched_tokens % page_size != 0:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=unaligned_matched_tokens matched_tokens=%d page_size=%d",
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
            return RouterKVReuseResult()
        computed_blocks = matched_tokens // page_size
        if computed_blocks < plan.start_block_index:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=before_plan_start computed_blocks=%d start_block_index=%d",
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
            return RouterKVReuseResult()

        max_plan_blocks = max(
            self._max_cacheable_blocks(req) - plan.start_block_index, 0
        )
        planned_blocks = min(plan.planned_prefix_blocks, max_plan_blocks)
        plan_offset = computed_blocks - plan.start_block_index
        if planned_blocks <= plan_offset:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=no_remaining_planned_blocks planned_blocks=%d plan_offset=%d max_plan_blocks=%d",
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
            return RouterKVReuseResult()

        pending = self._pending_fetches.get(str(req.rid))
        if pending is not None:
            if pending.plan.plan_id != plan.plan_id:
                self._pending_fetches.pop(str(req.rid), None)
                self._release_pending_fetch(pending)
            elif not pending.future.done():
                stop_waiting, reason = self._pending_should_stop_waiting(pending)
                if stop_waiting:
                    self._pending_fetches.pop(str(req.rid), None)
                    self._release_pending_fetch(pending)
                    self._finished_plan_keys.add(self._plan_key(req, pending.plan))
                    self._observe_reuse(
                        backend=pending.backend,
                        outcome="miss",
                        reason=reason,
                        wait_ms=self._pending_wait_ms(pending),
                    )
                    return RouterKVReuseResult()
                return RouterKVReuseResult(pending=True)
            else:
                return self._finish_pending_fetch(req, pending)

        logger.debug(
            "Submitting remote G2 fetch rid=%s plan_id=%s source=%s:%s start_block=%d max_blocks=%d matched_tokens=%d",
            req.rid,
            plan.plan_id,
            plan.source_worker_id,
            plan.source_dp_rank,
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
                "Skipping remote G2 plan rid=%s plan_id=%s reason=direct_transfer_unavailable",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
                backend="none",
                outcome="miss",
                reason="direct_transfer_unavailable",
            )
            return RouterKVReuseResult()
        if direct_transfer_enabled and req.host_hit_length > 0:
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="local_host_hit",
            )
            return RouterKVReuseResult()
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
                self._observe_reuse(
                    backend=self._current_backend_label(),
                    outcome="miss",
                    reason="direct_submit_unavailable",
                )
                return RouterKVReuseResult()
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
        self._pending_fetches[str(req.rid)] = _RemoteG2PendingFetch(
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
        return RouterKVReuseResult(pending=True)

    def _finish_pending_fetch(
        self, req: "Req", pending: _RemoteG2PendingFetch
    ) -> RouterKVReuseResult:
        self._pending_fetches.pop(str(req.rid), None)
        plan = pending.plan

        try:
            pages, reason = pending.future.result()
        except Exception:
            logger.exception(
                "Remote G2 fetch failed rid=%s plan_id=%s", req.rid, plan.plan_id
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="fetch_exception",
                wait_ms=self._pending_wait_ms(pending),
            )
            return RouterKVReuseResult()

        if not pages:
            logger.debug(
                "Remote G2 source returned no pages rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                reason,
            )
            self._unlock_pending_prefix(pending)
            indeterminate_transfer = _is_indeterminate_direct_transfer_reason(reason)
            if pending.device_indices is not None:
                if indeterminate_transfer:
                    self._quarantine_device_indices(
                        pending.device_indices,
                        reason,
                        backend=getattr(
                            pending, "backend", self._current_backend_label()
                        ),
                    )
                else:
                    self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error" if indeterminate_transfer else "miss",
                reason=reason,
                wait_ms=self._pending_wait_ms(pending),
            )
            return RouterKVReuseResult()

        if len(pages) > len(pending.expected_hashes):
            logger.warning(
                "Remote G2 source returned too many pages rid=%s plan_id=%s pages=%d expected=%d",
                req.rid,
                plan.plan_id,
                len(pages),
                len(pending.expected_hashes),
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="too_many_pages",
                wait_ms=self._pending_wait_ms(pending),
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()

        expected_hashes = pending.expected_hashes[: len(pages)]
        if tuple(page.block_hash for page in pages) != expected_hashes:
            logger.warning(
                "Remote G2 source returned non-contiguous pages rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="non_contiguous_pages",
                wait_ms=self._pending_wait_ms(pending),
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()

        insert_start = time.perf_counter()
        try:
            if pending.device_indices is None:
                logger.warning(
                    "Remote G2 direct transfer completed without target device indices"
                )
                self._finished_plan_keys.add(self._plan_key(req, plan))
                self._observe_reuse(
                    backend=pending.backend,
                    outcome="error",
                    reason="missing_target_device_indices",
                    wait_ms=self._pending_wait_ms(pending),
                )
                return RouterKVReuseResult()
            staged_tokens = self._insert_device_pages(
                req,
                pages,
                device_indices=pending.device_indices,
                start_block=pending.target_start_block,
            )
        except Exception:
            insert_ms = (time.perf_counter() - insert_start) * 1000
            logger.exception(
                "Remote G2 insert failed rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="insert_exception",
                wait_ms=self._pending_wait_ms(pending),
                insert_ms=insert_ms,
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()
        finally:
            self._unlock_pending_prefix(pending)
        insert_ms = (time.perf_counter() - insert_start) * 1000
        if staged_tokens > 0:
            req.remote_g2_hit_length = (
                getattr(req, "remote_g2_hit_length", 0) + staged_tokens
            )
            wait_ms = self._pending_wait_ms(pending)
            ready_wait_ms = self._pending_ready_wait_ms(pending)
            logger.debug(
                "Remote G2 staged %d tokens rid=%s plan_id=%s source=%s:%s wait_ms=%s future_ready_wait_ms=%s insert_ms=%.3f direct=%s",
                staged_tokens,
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                plan.source_dp_rank,
                _format_optional_ms(wait_ms),
                _format_optional_ms(ready_wait_ms),
                insert_ms,
                pending.device_indices is not None,
            )
        self._finished_plan_keys.add(self._plan_key(req, plan))
        outcome = "hit" if staged_tokens > 0 else "miss"
        self._observe_reuse(
            backend=pending.backend,
            outcome=outcome,
            reason=reason if staged_tokens > 0 else "insert_returned_zero",
            tokens=staged_tokens,
            wait_ms=self._pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
        )
        return RouterKVReuseResult(staged_tokens=staged_tokens)

    def _insert_device_pages(
        self,
        req: "Req",
        pages: list[ResolvedHostPage],
        *,
        device_indices: torch.Tensor,
        start_block: int,
    ) -> int:
        page_size = self.tree_cache.page_size
        token_count = len(pages) * page_size
        token_start = start_block * page_size
        token_end = token_start + token_count
        allocated_tokens = len(device_indices)

        if token_end > len(req.fill_ids):
            token_count = ((len(req.fill_ids) - token_start) // page_size) * page_size
            pages = pages[: token_count // page_size]
            token_end = token_start + token_count

        if token_count <= 0:
            self._free_device_indices(device_indices)
            return 0

        if token_count < allocated_tokens:
            self._free_device_indices(device_indices[token_count:])
            device_indices = device_indices[:token_count]

        try:
            prefix_indices = getattr(
                req, "prefix_indices", torch.empty((0,), dtype=torch.int64)
            )
            if token_start != len(prefix_indices):
                logger.debug(
                    "Remote G2 direct insert cannot attach suffix rid=%s token_start=%d prefix_indices=%d",
                    getattr(req, "rid", None),
                    token_start,
                    len(prefix_indices),
                )
                self._free_device_indices(device_indices)
                return 0

            key = RadixKey(
                req.fill_ids[:token_end],
                extra_key=req.extra_key,
                is_bigram=self.tree_cache.is_eagle,
            )
            if token_start > 0:
                prefix_indices = prefix_indices.to(
                    dtype=torch.int64, device=device_indices.device, copy=False
                )
                insert_value = torch.cat([prefix_indices, device_indices])
            else:
                insert_value = device_indices

            result = self.tree_cache.insert(InsertParams(key=key, value=insert_value))
            matched_length = result.prefix_len
            matched_new_tokens = min(max(0, matched_length - token_start), token_count)
            if matched_new_tokens > 0:
                self._free_device_indices(device_indices[:matched_new_tokens])
            staged_tokens = token_count - matched_new_tokens
            if staged_tokens <= 0:
                return 0
            return staged_tokens
        except Exception:
            self._free_device_indices(device_indices)
            raise


class RouterKVReuseManager:
    """Scheduler-facing router reuse control plane.

    The external wire key stays `remote_kv_reuse_plan`. Internally, the
    scheduler only calls this generic manager so future router-directed reuse
    plans can install a different handler without changing request plumbing.
    """

    def __init__(self, handlers: Iterable[RouterKVReuseHandler]):
        self.handlers = list(handlers)

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["RouterKVReuseManager"]:
        if not _router_kv_reuse_enabled(scheduler.server_args):
            return None

        handlers: list[RouterKVReuseHandler] = []
        remote_g2_handler = RemoteG2ReuseHandler.from_scheduler(scheduler)
        if remote_g2_handler is not None:
            handlers.append(remote_g2_handler)

        if not handlers:
            return None
        return cls(handlers)

    def maybe_stage_reuse_plan(self, req: "Req") -> int:
        return self.check_reuse_plan_progress(req).staged_tokens

    def has_reuse_plan(self, req: "Req") -> bool:
        for handler in self.handlers:
            has_reuse_plan = getattr(handler, "has_reuse_plan", None)
            if has_reuse_plan is not None and has_reuse_plan(req):
                return True
        return False

    def has_pending(self) -> bool:
        for handler in self.handlers:
            has_pending = getattr(handler, "has_pending", None)
            if has_pending is None:
                continue
            try:
                if has_pending():
                    return True
            except Exception:
                logger.exception("Router KV reuse handler pending check failed")
                return True
        return False

    def prefetch_reuse_plan(self, req: "Req") -> RouterKVReuseResult:
        for handler in self.handlers:
            prefetch = getattr(handler, "prefetch_remote_prefix", None)
            if prefetch is not None:
                result = prefetch(req)
            else:
                check = getattr(handler, "check_remote_prefix", None)
                if check is None:
                    continue
                result = check(req)
            if result.pending or result.staged_tokens > 0:
                return result
        return RouterKVReuseResult()

    def check_reuse_plan_progress(self, req: "Req") -> RouterKVReuseResult:
        for handler in self.handlers:
            if hasattr(handler, "check_remote_prefix"):
                result = handler.check_remote_prefix(req)
            else:
                result = RouterKVReuseResult(
                    staged_tokens=handler.maybe_stage_remote_prefix(req)
                )
            if result.pending or result.staged_tokens > 0:
                return result
        return RouterKVReuseResult()

    def maybe_stage_remote_prefix(self, req: "Req") -> int:
        return self.maybe_stage_reuse_plan(req)

    def shutdown(self) -> None:
        for handler in self.handlers:
            shutdown = getattr(handler, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    logger.exception("Router KV reuse handler shutdown failed")

    def release_request(self, rid: str) -> None:
        for handler in self.handlers:
            release = getattr(handler, "release_request", None)
            if release is not None:
                try:
                    release(rid)
                except Exception:
                    logger.exception(
                        "Router KV reuse handler failed to release rid=%s", rid
                    )


G2plusManager = RemoteG2ReuseHandler
