from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


class ListKind(enum.Enum):
    """The lists that hold the live requests."""

    WAITING = enum.auto()
    # The running requests.
    DECODING = enum.auto()
    CHUNKED = enum.auto()
    PD_BOOTSTRAP = enum.auto()
    PREFILL_LAUNCHED = enum.auto()
    PD_SENDING = enum.auto()
    PD_PREALLOC = enum.auto()
    PD_RETRACTED = enum.auto()
    PD_HELD = enum.auto()
    PD_RECEIVING = enum.auto()
    PD_RELEASING = enum.auto()
    HISPARSE_STAGING = enum.auto()
    GRAMMAR = enum.auto()
    DLLM = enum.auto()
    UOCR_PARKED = enum.auto()
    UOCR_RELEASED = enum.auto()
    DEFERRED = enum.auto()


class RequestList(ABC):
    """A list of live requests. Every live request is in exactly one list."""

    kind: ListKind

    @abstractmethod
    def __iter__(self) -> Iterator[Req]: ...

    @abstractmethod
    def join(self, items) -> None: ...

    @abstractmethod
    def leave(self, reqs: Iterable[Req]) -> None: ...

    def abort(self, recv_req: AbortReq) -> None:
        """Abort the requests `recv_req` names."""

    def take(self, pred: Optional[Callable[[Req], bool]] = None) -> List[Req]:
        """Drop the requests matching `pred`, or all of them, and return them in
        list order."""
        taken = [req for req in self if pred is None or pred(req)]
        self.leave(taken)
        return taken

    def clear(self) -> None:
        """Drop every request."""
        self.leave(list(self))

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _RequestList(RequestList):
    """Requests in join order, held by identity and changed in place: a loop that
    changes the list iterates a copy. An owner that wraps each request joins the
    wrappers and passes `req_of` to unwrap them."""

    def __init__(
        self,
        kind: ListKind,
        *,
        abort: Optional[Callable[[AbortReq], None]] = None,
        req_of: Callable[[Any], Req] = lambda x: x,
    ) -> None:
        self.kind = kind
        self._abort = abort
        self._req_of = req_of
        self._items: Dict[Req, Any] = {}

    def join(self, items: Iterable) -> None:
        """Add the given items; ones whose request is already in keep their place."""
        for item in items:
            self._items.setdefault(self._req_of(item), item)

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not in are ignored."""
        for req in reqs:
            self._items.pop(req, None)

    def leave_finished(self) -> None:
        self.leave([req for req in self._items if req.finished()])

    def clear(self) -> None:
        self._items.clear()

    def abort(self, recv_req: AbortReq) -> None:
        if self._abort is not None:
            self._abort(recv_req)

    def items(self) -> Iterator:
        """The joined items, in list order."""
        return iter(self._items.values())

    def __iter__(self) -> Iterator[Req]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, req: Req) -> bool:
        return req in self._items


class WaitingQueue(RequestList):
    """The requests waiting for prefill admission, in queue order. Held in a list,
    which joining appends to in place: requests join one at a time and the policy
    reorders the queue every pass, which a rebound dict would make linear per
    request."""

    kind = ListKind.WAITING

    def __init__(self, *, abort: Optional[Callable[[AbortReq], None]] = None) -> None:
        self._abort = abort
        self._reqs: List[Req] = []

    def join(self, reqs: Iterable[Req]) -> None:
        """Add the given requests at the back, in order."""
        self._reqs.extend(reqs)

    def prepend(self, reqs: Iterable[Req]) -> None:
        """Add the given requests at the front, in order."""
        self._reqs[:0] = reqs

    def replace(self, reqs: Iterable[Req]) -> None:
        """Make the queue the given requests, in order."""
        self._reqs = list(reqs)

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not in are ignored."""
        leaving = set(reqs)
        if leaving:
            self._reqs = [req for req in self._reqs if req not in leaving]

    def clear(self) -> None:
        self._reqs = []

    def abort(self, recv_req: AbortReq) -> None:
        if self._abort is not None:
            self._abort(recv_req)

    def __iter__(self) -> Iterator[Req]:
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def __contains__(self, req: Req) -> bool:
        return req in self._reqs

    def __getitem__(self, index):
        return self._reqs[index]


def move(reqs: Iterable[Req], src: RequestList, dst: RequestList) -> None:
    """Move requests from `src` to `dst`; each must be in `src`."""
    reqs = list(reqs)
    missing = [req.rid for req in reqs if req not in src]
    if missing:
        raise RuntimeError(f"Requests {missing} are not in {src.kind.name}")
    src.leave(reqs)
    dst.join(reqs)
