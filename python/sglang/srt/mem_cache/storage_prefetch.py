from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey


@dataclass
class _StoragePrefetchRetry:
    immediate: bool
    storage_hit_end: Optional[int] = None
    due_step: Optional[int] = None


class StoragePrefetchRetries:
    """New L3 attempts only (parked IO and retained staging retry at their owner);
    step deadlines keep TP ranks in lockstep, retries follow queue order, and a
    request past its re-issue budget is admitted with what the device holds."""

    def __init__(self):
        self._pending: dict[str, _StoragePrefetchRetry] = {}
        self._step = 0

    def poll_miss(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(False, storage_hit_end)

    def refetch(self, req_id: str, storage_hit_end: Optional[int] = None) -> None:
        self._pending[req_id] = _StoragePrefetchRetry(True, storage_hit_end)

    def cancel(self, req_id: str) -> None:
        self._pending.pop(req_id, None)

    def clear(self) -> None:
        self._pending.clear()

    def pop_ready(
        self, waiting_queue: list[Req], interval: int, max_attempts: int
    ) -> list[tuple[Req, Optional[int]]]:
        self._step += 1
        if not waiting_queue or not self._pending:
            return []
        # A speculative miss must never delay the queue head.
        head_id = waiting_queue[0].rid
        head_retry = self._pending.get(head_id)
        if head_retry is not None and not head_retry.immediate:
            self.cancel(head_id)
        if not any(
            retry.due_step is None or retry.due_step <= self._step
            for retry in self._pending.values()
        ):
            return []

        ready = []
        for req in waiting_queue:
            retry = self._pending.get(req.rid)
            if retry is None:
                continue
            if req.storage_prefetch_retry_attempts >= max_attempts:
                self.cancel(req.rid)
                continue
            if not retry.immediate:
                if interval <= 0:
                    self.cancel(req.rid)
                    continue
                if retry.due_step is None:
                    retry.due_step = self._step + interval
                if retry.due_step > self._step:
                    continue
            self.cancel(req.rid)
            ready.append((req, retry.storage_hit_end))
        return ready


@dataclass(frozen=True)
class StagedPrefetchPlan:
    """One admission pass, computed before promoting the joint FULL/SWA match."""

    operation_id: int
    key: RadixKey
    device_prefix_len: int
    full_tokens: int
    swa_tokens: int
