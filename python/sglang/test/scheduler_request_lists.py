"""Checks on the scheduler's request lists, for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


def request_lists(scheduler: Scheduler) -> Dict[str, List[Req]]:
    """The lists that own the scheduler's live requests."""
    return {name: entry.reqs for name, entry in scheduler.request_lists().items()}


def assert_one_list_per_request(scheduler: Scheduler) -> None:
    """At an iteration boundary: each live request is in exactly one list, and
    every request a pending step runs is either finished or in a list."""
    owners: Dict[int, str] = {}
    for name, reqs in request_lists(scheduler).items():
        for req in reqs:
            assert id(req) not in owners, (
                f"request {req.rid} is in both {owners[id(req)]} and {name}"
            )
            owners[id(req)] = name
    for req in scheduler.decoding:
        assert not req.finished(), f"finished request {req.rid} is still decoding"
    for req in request_lists(scheduler).get("releasing", []):
        assert req.finished(), f"releasing request {req.rid} is not finished"
    for record in scheduler.pending:
        for req in record.step.reqs:
            assert req.finished() or id(req) in owners, (
                f"request {req.rid} of a pending step is in no list"
            )
