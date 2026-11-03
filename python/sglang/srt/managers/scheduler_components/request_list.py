from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Iterator, Set

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RequestList:
    """Requests in join order, held by identity. Each change rebinds a new dict,
    so leaving while iterating is safe."""

    def __init__(self) -> None:
        self._reqs: Dict[Req, None] = {}

    def join(self, reqs: Iterable[Req]) -> None:
        """Add the given requests; ones already in keep their place."""
        joining = [req for req in reqs if req not in self._reqs]
        if joining:
            self._reqs = {**self._reqs, **dict.fromkeys(joining)}

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not in are ignored."""
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
