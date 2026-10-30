from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import BaseFinishReason, Req


class PendingFinishes:
    """Finishes decided outside a step commit (an abort), one per request in
    arrival order, until the scheduler's drain applies them."""

    def __init__(self) -> None:
        self._finishes: Dict[Req, BaseFinishReason] = {}

    def add(self, req: Req, reason: BaseFinishReason) -> None:
        """The first reason wins; a finished request ignores it."""
        if not req.finished():
            self._finishes.setdefault(req, reason)

    def take_all(self) -> List[Tuple[Req, BaseFinishReason]]:
        """Pop every finish, in arrival order, but those of the requests that
        finished meanwhile."""
        taken = [(req, r) for req, r in self._finishes.items() if not req.finished()]
        self._finishes.clear()
        return taken

    def pop(self, req: Req) -> BaseFinishReason | None:
        return self._finishes.pop(req, None)

    def reqs(self) -> List[Req]:
        return list(self._finishes)

    def __contains__(self, req: Req) -> bool:
        return req in self._finishes

    def __bool__(self) -> bool:
        return bool(self._finishes)
