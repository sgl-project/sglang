"""PD decode seats in arrival order (SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER).

A PD decode scheduler admits from the head of its waiting queue, which holds the
requests whose KV has landed. Stock joins that queue in landing order, the order
the prefill side finishes in, so at the running cap a request that arrived early
but landed late waits behind requests that arrived after it.

With the switch each request is numbered once, on its first arrival at the
preallocation queue, and every join of the waiting queue (landed and resumed
requests) keeps the queue sorted by that number. The admission is unchanged, so a
freed seat goes to the landed request that arrived first; a retracted request keeps
its number. Nothing else of the request is read: no length, no clock.

The number comes from a per-scheduler counter. Every rank of a decode instance adds
requests to its preallocation queue in the same order and joins the waiting queue
from the same poll consensus, so every rank holds the same queue. Under attention DP
each DP rank's scheduler orders the requests dispatched to it.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_list import move

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.request_list import IndexedReqList

logger = logging.getLogger(__name__)


def init_seat_age_order(
    *, pp_size: int, priority_scheduling: bool
) -> Optional[SeatAgeOrder]:
    """The arrival order of a PD decode scheduler's waiting queue, or None for
    stock's landing order. Server settings only, so every rank decides alike."""
    if not envs.SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER.get():
        return None
    if pp_size != 1:
        logger.info("SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER declined: pp_size %d", pp_size)
        return None
    if priority_scheduling:
        logger.info(
            "SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER declined: priority scheduling "
            "sorts the waiting queue at admission"
        )
        return None
    logger.info(
        "SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER engaged: a freed decode seat goes to "
        "the landed request that arrived first"
    )
    return SeatAgeOrder()


class SeatAgeOrder:
    """Decode-side arrival numbers and the arrival-ordered waiting queue of one PD
    decode scheduler."""

    def __init__(self):
        self._numbers = itertools.count()
        # For the log line: joined requests placed ahead of a request already
        # waiting, and the joins that placed any.
        self.overtakes = 0
        self.reordered_joins = 0
        self._logged_joins = 0

    def stamp(self, req: Req) -> int:
        """`req`'s arrival number, given on first sight and kept."""
        if req.decode_arrival_seq is None:
            req.decode_arrival_seq = next(self._numbers)
        return req.decode_arrival_seq

    def join(self, waiting_queue: IndexedReqList, reqs: List[Req]) -> None:
        """Move `reqs` into the waiting queue (whose items are the requests) and keep
        it in arrival order. A request not numbered yet is numbered here, in queue
        order and then join order, as on every rank."""
        if not reqs:
            return
        waiting = list(waiting_queue)
        move(reqs, waiting_queue)
        waiting_queue.reorder(sorted(waiting_queue, key=self.stamp))
        if not waiting:
            return
        # The waiting part was in order, so its last request is its newest.
        newest = waiting[-1].decode_arrival_seq
        ahead = sum(1 for req in reqs if req.decode_arrival_seq < newest)
        if ahead:
            self._note(ahead)

    def _note(self, ahead: int) -> None:
        self.overtakes += ahead
        self.reordered_joins += 1
        # Logged at doubling join counts, so a long run logs a few lines only.
        if self.reordered_joins >= 2 * self._logged_joins:
            self._logged_joins = self.reordered_joins
            logger.info(
                "SGLANG_OPT_PD_DECODE_SEAT_AGE_ORDER: %d landed requests queued "
                "ahead of later arrivals in %d joins",
                self.overtakes,
                self.reordered_joins,
            )
