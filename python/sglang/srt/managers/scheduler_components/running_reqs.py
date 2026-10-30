from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Set, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class RunningReqs:
    """The requests in the running state.

    A request joins once a launched step runs its last prefill chunk, and leaves
    when it finishes, is retracted or preempted, or a pause retracts it. Kept in
    join order, which every rank sees alike. Each change binds a new tuple, so a
    caller holding an earlier ``reqs()`` never sees it change.
    """

    def __init__(self) -> None:
        self._reqs: Tuple[Req, ...] = ()
        self._members: Set[Req] = set()

    def join(self, reqs: Iterable[Req]) -> None:
        """Add the given requests; ones already running keep their place."""
        joining = []
        for req in reqs:
            if req not in self._members:
                self._members.add(req)
                joining.append(req)
        if joining:
            self._reqs = (*self._reqs, *joining)

    def leave_finished(self) -> None:
        self._leave([req for req in self._reqs if req.finished()])

    def remove(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not running are ignored."""
        if self._members:
            self._leave([req for req in reqs if req in self._members])

    def clear(self) -> None:
        self._reqs = ()
        self._members = set()

    def reqs(self) -> Tuple[Req, ...]:
        return self._reqs

    def __iter__(self) -> Iterator[Req]:
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def __contains__(self, req: Req) -> bool:
        return req in self._members

    def _leave(self, leaving: Iterable[Req]) -> None:
        leaving = set(leaving)
        if not leaving:
            return
        self._members -= leaving
        self._reqs = tuple(req for req in self._reqs if req not in leaving)

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
        if expected == self._members:
            return
        not_running = sorted(req.rid for req in expected - self._members)
        not_held = sorted(req.rid for req in self._members - expected)
        raise RuntimeError(
            "Running requests disagree with the batches: "
            f"held by a batch but not running: {not_running}; "
            f"running but held by no batch: {not_held}"
        )
