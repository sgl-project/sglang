from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Iterator, Optional, Set

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class RunningReqs:
    """The running requests, in join order (the same on every rank).

    A request runs from the launch of its whole prompt until it finishes; the
    steps still computing for it are tracked apart. It leaves when it finishes,
    is retracted or preempted, or a pause retracts it. Each change rebinds a new
    dict, so leaving while iterating is safe.
    """

    def __init__(self) -> None:
        self._reqs: Dict[Req, None] = {}

    def join(self, reqs: Iterable[Req]) -> None:
        """Add the given requests; ones already running keep their place."""
        joining = [req for req in reqs if req not in self._reqs]
        if joining:
            self._reqs = {**self._reqs, **dict.fromkeys(joining)}

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not running are ignored."""
        self._drop({req for req in reqs if req in self._reqs})

    def leave_finished(self) -> None:
        self._drop({req for req in self._reqs if req.finished()})

    def __iter__(self) -> Iterator[Req]:
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def __contains__(self, req: Req) -> bool:
        return req in self._reqs

    def _drop(self, leaving: Set[Req]) -> None:
        if leaving:
            self._reqs = {req: None for req in self._reqs if req not in leaving}

    def check(
        self,
        batches: Iterable[Optional[ScheduleBatch]],
        chunked_req: Optional[Req],
    ) -> None:
        """Raise unless the running requests are exactly the unfinished ones the
        batches hold, apart from the chunked request."""
        expected = {
            req
            for batch in batches
            if batch is not None
            for req in batch.reqs
            if not req.finished() and req is not chunked_req
        }
        running = self._reqs.keys()
        if expected == running:
            return
        not_running = sorted(req.rid for req in expected - running)
        not_held = sorted(req.rid for req in running - expected)
        raise RuntimeError(
            "Running requests disagree with the batches: "
            f"held by a batch but not running: {not_running}; "
            f"running but held by no batch: {not_held}"
        )
