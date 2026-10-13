"""The lists that own the scheduler's live requests, as the readers see them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


@dataclass(frozen=True, kw_only=True)
class RequestList:
    """One list that owns live requests, with what the readers ask of it.

    Each live request is in exactly one list. `Scheduler.request_lists()`
    gathers the core lists and each owner's lists; abort, idle, the running
    timeout, weight version, the sampling-mask scan, load and metrics read
    the lists through it.
    """

    # Unique among the scheduler's lists; load and metrics look lists up by it.
    name: str
    # The members, in list order (the same on every rank).
    reqs: List[Req]
    # A step may run its requests: abort marks them `to_finish`, and the
    # running timeout and `collect_inflight_reqs` read them.
    inflight: bool = False
    # The members wait for admission: the waiting timeout, the queue length
    # metrics and the load's waiting count read them.
    queued: bool = False
    # A retract-mode pause retracts the unfinished members, which hold a row,
    # so that their KV is rebuilt under the new weights.
    retract_on_pause: bool = False
    # Aborts the members an `AbortReq` matches (beyond `to_finish`); None when
    # abort reaches the list only through `inflight`, or not at all.
    abort: Optional[Callable[[AbortReq], None]] = None
    # `is_fully_idle` counts the members, except:
    # - for the health check, unless they run soon and so carry it (a queued
    #   PD request may wait on a handshake, KV space or a transfer);
    health_check: bool = False
    # - with ignore_waiting, the members that hold no KV;
    waiting: bool = False
    # - with ignore_releasing, all of them.
    releasing: bool = False

    def keeps_busy(
        self, *, for_health_check: bool, ignore_waiting: bool, ignore_releasing: bool
    ) -> bool:
        """Whether its members keep the scheduler from being fully idle."""
        if for_health_check and not self.health_check:
            return False
        if ignore_releasing and self.releasing:
            return False
        if ignore_waiting and self.waiting:
            return any(req.kv.holds_kv for req in self.reqs)
        return len(self.reqs) > 0
