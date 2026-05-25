from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from sglang.srt.mem_cache.shared_hicache.control import start_source_transfer_server
from sglang.srt.mem_cache.shared_hicache.plan import normalize_endpoint


logger = logging.getLogger(__name__)


def endpoint_to_bind(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(normalize_endpoint(endpoint))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported shared HiCache endpoint scheme: {parsed.scheme}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"shared HiCache endpoint must include host and port: {endpoint}"
        )
    return parsed.hostname, parsed.port


def select_dp_endpoint(endpoint_spec: object, dp_rank: int) -> Optional[str]:
    if not endpoint_spec:
        return None
    if not isinstance(endpoint_spec, str):
        raise ValueError("shared_hicache_config.control.endpoint must be a string")
    spec = endpoint_spec.strip()
    if not spec:
        return None
    if "{dp_rank}" in spec:
        spec = spec.format(dp_rank=dp_rank)
    return normalize_endpoint(spec)


class SharedHiCacheSourceService:
    def __init__(
        self,
        *,
        endpoint: str,
        worker_id: Optional[int],
        dp_rank: int,
        worker_limit: int,
        max_body_bytes: Callable[[], int],
        direct_transfer_enabled: Callable[[], bool],
        handle_source_transfer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.dp_rank = dp_rank
        self.max_body_bytes = max_body_bytes
        self.direct_transfer_enabled = direct_transfer_enabled
        self.handle_source_transfer = handle_source_transfer
        self._source_server: Optional[Any] = None
        self._source_thread: Optional[threading.Thread] = None
        self._activity_lock = threading.Lock()
        self._active_ops = 0
        self._resolver_semaphore = threading.BoundedSemaphore(worker_limit)

    def start(self) -> None:
        host, port = endpoint_to_bind(self.endpoint)
        self._source_server, self._source_thread = start_source_transfer_server(
            host=host,
            port=port,
            endpoint=self.endpoint,
            worker_id=self.worker_id,
            dp_rank=self.dp_rank,
            max_body_bytes=self.max_body_bytes,
            try_enter=self.try_enter,
            exit_resolver=self.exit,
            direct_transfer_enabled=self.direct_transfer_enabled,
            handle_source_transfer=self.handle_source_transfer,
        )

    def try_enter(self) -> bool:
        if not self._resolver_semaphore.acquire(blocking=False):
            return False
        with self._activity_lock:
            self._active_ops += 1
        return True

    def exit(self) -> None:
        with self._activity_lock:
            self._active_ops -= 1
        self._resolver_semaphore.release()

    def active_count(self) -> int:
        with self._activity_lock:
            return int(self._active_ops)

    def shutdown(self) -> None:
        server = self._source_server
        self._source_server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                logger.debug(
                    "Shared HiCache source resolver shutdown failed", exc_info=True
                )

        source_thread = self._source_thread
        self._source_thread = None
        if (
            source_thread is not None
            and source_thread is not threading.current_thread()
        ):
            try:
                source_thread.join(timeout=1)
            except Exception:
                logger.debug(
                    "Shared HiCache source resolver join failed", exc_info=True
                )
