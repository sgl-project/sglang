"""The lists that own the scheduler's live requests, as the readers see them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.deferred_output import ResponseCompletion
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.running_reqs import RunningReqs


@dataclass(frozen=True, kw_only=True)
class RequestList:
    """One list that owns live requests, with what the readers ask of it.

    Each live request is in exactly one list. Batches are step records: a
    request a step runs is also in a list, apart from the gaps noted where the
    lists are built.
    """

    # Unique among the scheduler's lists. Readers ask the flags below, never
    # the name, which serves logs and tests.
    name: str
    # The members, in list order (the same on every rank).
    reqs: List[Req]
    # The health check counts the members as busy: they run soon, so a health
    # check request rides on their output instead of running itself.
    health_check: bool = False
    # A fully idle check that ignores waiting requests counts only the members
    # holding KV.
    waiting: bool = False
    # The fully idle check never counts the members.
    never_busy: bool = False

    def keeps_busy(self, *, for_health_check: bool, ignore_waiting: bool) -> bool:
        """Whether its members keep the scheduler from being fully idle."""
        if self.never_busy:
            return False
        if for_health_check and not self.health_check:
            return False
        if ignore_waiting and self.waiting:
            return any(req.kv.holds_kv for req in self.reqs)
        return len(self.reqs) > 0


def core_request_lists(
    *,
    waiting_queue: List[Req],
    grammar_queue: List[Req],
    running_reqs: RunningReqs,
    chunked_req: Optional[Req],
) -> List[RequestList]:
    """The lists every mode has: waiting, grammar, running and chunked."""
    return [
        # PD decode: the requests whose KV arrived wait here for a row.
        RequestList(
            name="waiting", reqs=waiting_queue, health_check=True, waiting=True
        ),
        RequestList(name="grammar", reqs=grammar_queue),
        RequestList(name="running", reqs=list(running_reqs), health_check=True),
        RequestList(
            name="chunked",
            reqs=[chunked_req] if chunked_req is not None else [],
            health_check=True,
        ),
    ]


def deferred_response_lists(
    deferred_responses: Optional[Dict[str, Tuple[Req, ResponseCompletion]]],
) -> List[RequestList]:
    """The finished requests whose response is held; see deferred_output."""
    return [
        # Only the output rank holds them, and every rank must agree on idle.
        RequestList(
            name="deferred",
            reqs=[req for req, _ in (deferred_responses or {}).values()],
            never_busy=True,
        )
    ]
