from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
    normalize_endpoint,
)
from sglang.srt.mem_cache.shared_hicache.source import ResolvedHostPage
from sglang.srt.mem_cache.shared_hicache.transfer import SharedHiCacheTransferBackend
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


SHARED_HICACHE_TRANSFER_REQUEST = "shared_hicache_transfer_request"
SHARED_HICACHE_TRANSFER_DONE = "shared_hicache_transfer_done"


def endpoint_to_zmq(endpoint: str) -> str:
    parsed = urlparse(normalize_endpoint(endpoint))
    if parsed.scheme != "tcp":
        raise ValueError(f"unsupported shared HiCache endpoint scheme: {parsed.scheme}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"shared HiCache endpoint must include host and port: {endpoint}"
        )
    return NetworkAddress(parsed.hostname, parsed.port).to_tcp()


def is_indeterminate_direct_transfer_reason(reason: str) -> bool:
    reason = str(reason)
    return reason.startswith(SHARED_HICACHE_DIRECT_TIMEOUT_REASON)


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return int(value)
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def pages_from_transfer_result(
    payload: Mapping[str, Any],
    plan: SharedHiCachePlan,
    *,
    start_block: int,
    max_blocks: int,
) -> list[ResolvedHostPage]:
    transferred_blocks = _coerce_int(
        payload.get("transferred_blocks", 0), "transferred_blocks"
    )
    if transferred_blocks < 0:
        raise ValueError("transferred_blocks must be non-negative")
    transferred_blocks = min(transferred_blocks, max_blocks)
    block_hashes = plan.planned_hashes[start_block : start_block + transferred_blocks]
    return [
        ResolvedHostPage(block_hash=block_hash, hash_value="", data=b"")
        for block_hash in block_hashes
    ]


class SharedHiCacheTransferHandle:
    """Decode-disagg-style local poll handle for one Shared HiCache transfer."""

    def __init__(
        self,
        *,
        transfer_backend: SharedHiCacheTransferBackend,
        transfer_id: str,
        plan: SharedHiCachePlan,
        start_block: int,
        max_blocks: int,
        timeout_secs: float,
        pop_source_completion: Callable[[str], Optional[Mapping[str, Any]]],
    ):
        self.transfer_backend = transfer_backend
        self.transfer_id = str(transfer_id)
        self.plan = plan
        self.start_block = int(start_block)
        self.max_blocks = int(max_blocks)
        self.timeout_secs = float(timeout_secs)
        self.pop_source_completion = pop_source_completion
        self.submitted_at = time.perf_counter()
        self.done_at = 0.0
        self._status = KVPoll.Transferring
        self._pages: list[ResolvedHostPage] = []
        self._reason = "source_transfer_pending"
        self._source_terminal_seen = False

    @classmethod
    def failed(
        cls,
        *,
        transfer_backend: SharedHiCacheTransferBackend,
        transfer_id: str,
        plan: SharedHiCachePlan,
        start_block: int,
        max_blocks: int,
        timeout_secs: float,
        reason: str,
    ) -> "SharedHiCacheTransferHandle":
        handle = cls(
            transfer_backend=transfer_backend,
            transfer_id=transfer_id,
            plan=plan,
            start_block=start_block,
            max_blocks=max_blocks,
            timeout_secs=timeout_secs,
            pop_source_completion=lambda _transfer_id: None,
        )
        handle._finish(KVPoll.Failed, [], reason)
        return handle

    def poll(self) -> KVPoll:
        if self._status in (KVPoll.Success, KVPoll.Failed):
            return self._status

        elapsed_secs = time.perf_counter() - self.submitted_at
        if self.timeout_secs >= 0 and elapsed_secs > self.timeout_secs:
            logger.warning(
                "Shared HiCache transfer timed out transfer_id=%s ms=%.3f; target pages will be quarantined",
                self.transfer_id,
                elapsed_secs * 1000,
            )
            self._finish(KVPoll.Failed, [], SHARED_HICACHE_DIRECT_TIMEOUT_REASON)
            return self._status

        notification = self._pop_target_transfer_notification()
        if notification is not None:
            transferred_blocks, reason = notification
            pages = pages_from_transfer_result(
                {"transferred_blocks": transferred_blocks},
                self.plan,
                start_block=self.start_block,
                max_blocks=self.max_blocks,
            )
            logger.debug(
                "Shared HiCache target NIXL notification completed transfer_id=%s pages=%d reason=%s source_terminal_seen=%s",
                self.transfer_id,
                len(pages),
                reason,
                self._source_terminal_seen,
            )
            self._finish(KVPoll.Success, pages, reason)
            return self._status

        completion = self.pop_source_completion(self.transfer_id)
        if completion is not None:
            self._source_terminal_seen = True
            reason = str(completion.get("reason", "ok"))
            if not completion.get("ok"):
                self._finish(KVPoll.Failed, [], reason)
                return self._status
            try:
                transferred_blocks = _coerce_int(
                    completion.get("transferred_blocks", 0),
                    "transferred_blocks",
                )
            except ValueError:
                self._finish(KVPoll.Failed, [], "malformed_source_transfer_done")
                return self._status
            if transferred_blocks <= 0:
                pages = pages_from_transfer_result(
                    completion,
                    self.plan,
                    start_block=self.start_block,
                    max_blocks=self.max_blocks,
                )
                self._finish(KVPoll.Success, pages, reason)
                return self._status
            # Positive source completion is not a readiness signal. Match disagg:
            # target observes the transfer-completion notification locally.
        return self._status

    def result(self) -> tuple[list[ResolvedHostPage], str]:
        status = self.poll()
        if status not in (KVPoll.Success, KVPoll.Failed):
            raise RuntimeError(
                "Shared HiCache transfer result requested before completion"
            )
        return self._pages, self._reason

    def done(self) -> bool:
        return self.poll() in (KVPoll.Success, KVPoll.Failed)

    def _finish(
        self,
        status: KVPoll,
        pages: list[ResolvedHostPage],
        reason: str,
    ) -> None:
        self._status = status
        self._pages = list(pages)
        self._reason = str(reason)
        self.done_at = time.perf_counter()

    def _pop_target_transfer_notification(self) -> Optional[tuple[int, str]]:
        pop = getattr(self.transfer_backend, "pop_target_transfer_notification", None)
        if not callable(pop):
            return None
        return pop(self.transfer_id)


class SharedHiCacheTargetTransferTracker:
    """Tracks source completion messages for target-side direct transfers."""

    def __init__(
        self,
        *,
        transfer_backend: Optional[SharedHiCacheTransferBackend],
    ):
        self.transfer_backend = transfer_backend
        self._lock = threading.Lock()
        self._completions: dict[str, Mapping[str, Any]] = {}
        self._active: set[str] = set()

    def handle_done(self, payload: Mapping[str, Any]) -> None:
        transfer_id = str(payload.get("transfer_id") or "")
        if not transfer_id:
            logger.warning("Ignoring SharedHiCache transfer_done without transfer_id")
            return
        with self._lock:
            if transfer_id not in self._active:
                return
            self._completions[transfer_id] = dict(payload)

    def pop_completion(self, transfer_id: str) -> Optional[Mapping[str, Any]]:
        with self._lock:
            return self._completions.pop(str(transfer_id), None)

    def start(self, transfer_id: str) -> None:
        with self._lock:
            self._active.add(str(transfer_id))

    def finish(self, transfer_id: str) -> None:
        transfer_id = str(transfer_id)
        with self._lock:
            self._active.discard(transfer_id)
            self._completions.pop(transfer_id, None)
        drop_notification = getattr(
            self.transfer_backend, "drop_target_transfer_notification", None
        )
        if callable(drop_notification):
            drop_notification(transfer_id)
