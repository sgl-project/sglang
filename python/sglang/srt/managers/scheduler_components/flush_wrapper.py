import logging
import time
from typing import Callable, Optional, Tuple

from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput


class SchedulerFlushWrapper:
    def __init__(
        self,
        *,
        flush_cache: Callable[[], bool],
        is_fully_idle: Callable[[], bool],
        send_output: Callable[[FlushCacheReqOutput, FlushCacheReqInput], None],
        pp_rank: Optional[int] = None,
    ) -> None:
        self._flush_cache = flush_cache
        self._is_fully_idle = is_fully_idle
        # Answers a deferred flush through the scheduler's control-reply path.
        self._send_output = send_output
        self._pp_rank = pp_rank
        self._pending: Optional[Tuple[FlushCacheReqInput, float]] = None
        self._pp_retry = False

    def handle(self, recv_req: FlushCacheReqInput) -> Optional[FlushCacheReqOutput]:
        if self._pp_rank is not None and self._pp_rank != 0:
            return None  # Only PP0 owns the caller and its deadline.
        if self._pending is not None:
            return FlushCacheReqOutput(
                success=False,
                message="Another flush_cache is already in progress.",
            )

        timeout_s = float(recv_req.timeout_s or 0.0)
        if self._pp_rank is not None:
            self._pending = (recv_req, time.monotonic() + timeout_s)
            return None
        if timeout_s <= 0.0:
            return FlushCacheReqOutput(success=self._flush_cache())

        if self._is_fully_idle():
            return FlushCacheReqOutput(success=self._flush_cache())

        self._pending = (recv_req, time.monotonic() + timeout_s)
        return None

    def check_pending(self) -> None:
        if self._pending is None or self._pp_rank is not None:
            return

        pending_req, deadline = self._pending

        if self._is_fully_idle():
            success = self._flush_cache()
            self._pending = None
            self._send_output(FlushCacheReqOutput(success=success), pending_req)
            return

        if time.monotonic() >= deadline:
            logging.warning(
                "Deferred flush_cache timed out while waiting for idle state."
            )
            self._pending = None
            self._send_output(
                FlushCacheReqOutput(
                    success=False, message="Timed out waiting for idle state."
                ),
                pending_req,
            )

    def prepare_pp_probe(self, vote: Callable[[float], list[int]]) -> bool:
        if self._pending is None:
            return False
        _, deadline = self._pending
        idle, self._pp_retry = vote(deadline)
        if not idle:
            self.finish_pp_probe(False)
        return bool(idle)

    def finish_pp_probe(self, success: bool) -> None:
        if self._pending is None or (not success and self._pp_retry):
            return
        req, _ = self._pending
        self._pending = None
        message = (
            "Timed out waiting for idle state." if not success and req.timeout_s else ""
        )
        self._send_output(FlushCacheReqOutput(success=success, message=message), req)
