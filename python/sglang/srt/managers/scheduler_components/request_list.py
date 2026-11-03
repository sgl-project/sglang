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


class ReqList(ABC):
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


class KeyedReqList(ReqList):
    """Requests in join order, held in a dict by identity: for a list asked `in`
    often. Changed in place, so a loop that changes the list iterates a copy."""

    def __init__(self, kind: ListKind) -> None:
        self.kind = kind
        self._reqs: Dict[Req, None] = {}

    def join(self, reqs: Iterable[Req]) -> None:
        """Add the given requests; ones already in keep their place."""
        for req in reqs:
            self._reqs.setdefault(req, None)

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not in are ignored."""
        for req in reqs:
            self._reqs.pop(req, None)

    def leave_finished(self) -> None:
        self.leave([req for req in self._reqs if req.finished()])

    def clear(self) -> None:
        self._reqs.clear()

    def __iter__(self) -> Iterator[Req]:
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def __contains__(self, req: Req) -> bool:
        return req in self._reqs


class IndexedReqList(ReqList):
    """The default list: items in an order the owner sets, held in a list, for a
    queue scanned in order and dropped from in batches. An owner that wraps each
    request stores the wrappers and passes `req_of` to unwrap them."""

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
        self._items: List[Any] = []

    def join(self, items: Iterable) -> None:
        """Add the given items at the back, in order."""
        self._items.extend(items)

    def prepend(self, items: Iterable) -> None:
        """Add the given items at the front, in order."""
        self._items[:0] = items

    def replace(self, items: Iterable) -> None:
        self._items = list(items)

    def leave(self, reqs: Iterable[Req]) -> None:
        """Drop the items of the given requests; ones not in are ignored."""
        leaving = set(reqs)
        if leaving:
            self._items = [
                item for item in self._items if self._req_of(item) not in leaving
            ]

    def clear(self) -> None:
        self._items = []

    def abort(self, recv_req: AbortReq) -> None:
        if self._abort is not None:
            self._abort(recv_req)

    def items(self) -> Iterator:
        return iter(self._items)

    def __iter__(self) -> Iterator[Req]:
        return map(self._req_of, self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, req: Req) -> bool:
        return any(self._req_of(item) is req for item in self._items)

    def __getitem__(self, index):
        return self._items[index]


def move(reqs: Iterable[Req], src: ReqList, dst: ReqList) -> None:
    """Move requests from `src` to `dst`; each must be in `src`."""
    reqs = list(reqs)
    missing = [req.rid for req in reqs if req not in src]
    if missing:
        raise RuntimeError(f"Requests {missing} are not in {src.kind.name}")
    src.leave(reqs)
    dst.join(reqs)
