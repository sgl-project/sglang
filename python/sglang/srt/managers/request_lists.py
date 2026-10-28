"""The lists that own the scheduler's live requests, as the readers see them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot


@dataclass(frozen=True, kw_only=True)
class RequestList:
    """One list that owns live requests, with what the readers ask of it.
    Each live request is in exactly one list."""

    # Unique among the scheduler's lists. Readers ask the flags below, never
    # the name, which serves logs and tests.
    name: str
    # The members, in list order (the same on every rank).
    reqs: List[Req]
    # A step may run its requests: abort marks them `to_finish`, and the
    # running timeout and `collect_inflight_reqs` read them.
    inflight: bool = False
    # The members wait for admission: the waiting timeout, the queue length
    # metrics and the load's queue length read them.
    queued: bool = False
    # A retract-mode pause retracts the unfinished members, which hold a row,
    # so that their KV is rebuilt under the new weights.
    retract_on_pause: bool = False
    # Aborts the members an `AbortReq` matches (beyond `to_finish`); None when
    # abort reaches the list only through `inflight`, or not at all.
    abort: Optional[Callable[[AbortReq], None]] = None
    # The members keep the scheduler from being fully idle, except:
    # - for the health check, unless this is set (they run soon; a queued PD
    #   request may instead wait on a handshake, KV space or a transfer);
    health_check: bool = False
    # - with ignore_waiting, when this is set, the members that hold no KV;
    waiting: bool = False
    # - with ignore_releasing, all of them.
    releasing: bool = False
    # The members may own a KV row (`req.kv.holds_kv`): the busy memory check
    # counts their private KV, and the load leaves their tokens out of its
    # pending tokens, as the pool counts them as used.
    holds_kv: bool = False
    # The members' adapters stay in use: LoRA admission fits a new adapter
    # beside them.
    pins_lora: bool = False
    # The members are the running requests of the metrics, the logs and the
    # load.
    counts_running: bool = False
    # The load counts the members among its waiting requests.
    counts_queued: bool = False
    # PD decode: the members' KV has not arrived from the prefill side; the
    # load counts their tokens as awaited.
    awaits_kv: bool = False
    # The member stopped between two prefill chunks: the load counts its
    # prompt tokens still to prefill.
    partly_prefilled: bool = False
    # The members finished and only their response is held (a deferred
    # output). Only the output rank holds them, so they keep nothing busy
    # (every rank must agree on idle), and the finished-request check skips
    # them.
    held_response: bool = False

    def keeps_busy(
        self, *, for_health_check: bool, ignore_waiting: bool, ignore_releasing: bool
    ) -> bool:
        """Whether its members keep the scheduler from being fully idle."""
        if self.held_response:
            return False
        if for_health_check and not self.health_check:
            return False
        if ignore_releasing and self.releasing:
            return False
        if ignore_waiting and self.waiting:
            return any(req.kv.holds_kv for req in self.reqs)
        return len(self.reqs) > 0


class CoreLists:
    """The scheduler's own request lists, one live object shared with the
    components that read them."""

    def __init__(self) -> None:
        # The requests waiting for prefill admission.
        self.waiting_queue: List[Req] = []
        # The requests settled into decode (continuous batching): each from
        # the launch of its last prefill chunk to the commit that finishes it.
        self.decoding: List[Req] = []
        # The request between two chunks of its prefill.
        self.chunked_req: Optional[Req] = None
        # The chunked request a client aborted, released at the next schedule.
        self._pending_chunked_abort_req: Optional[Req] = None

    def join_decoding(self, reqs: List[Req], slot: ScheduleSlot) -> None:
        """Append requests to decoding: those whose last prefill chunk a launch
        runs, and those whose previous step was committed elsewhere (PD decode
        admission, HiSparse ready), which join at schedule. Each takes the PP
        slot that launches it."""
        for req in reqs:
            req.ring_slot = slot.index
        self.decoding.extend(reqs)
