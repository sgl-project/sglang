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
    Mapping,
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
    PD_OFFLOADING = enum.auto()
    HISPARSE_STAGING = enum.auto()
    GRAMMAR = enum.auto()
    DLLM = enum.auto()
    UOCR_PARKED = enum.auto()
    UOCR_RELEASED = enum.auto()
    DEFERRED = enum.auto()


PD_KINDS = frozenset(
    {
        ListKind.PD_BOOTSTRAP,
        ListKind.PD_SENDING,
        ListKind.PD_PREALLOC,
        ListKind.PD_RETRACTED,
        ListKind.PD_HELD,
        ListKind.PD_RECEIVING,
        ListKind.PD_RELEASING,
        ListKind.PD_OFFLOADING,
    }
)


def assert_empty(lists: Mapping[ListKind, ReqList], kinds: Iterable[ListKind]) -> None:
    """A teardown runs behind an idle gate that leaves these lists empty; a
    request still in one would be dropped and never answered."""
    nonempty = sorted(kind.name for kind in kinds if lists.get(kind))
    if nonempty:
        raise RuntimeError(f"Requests still in {nonempty} at teardown")


class ReqList(ABC):
    """A list of live requests. Every live request is in at most one list,
    which its `owner_list` names: joining a second list raises. Requests come
    in and move between lists through `move`, and an ended request leaves
    through the retirer; the underscored edits are for those two only."""

    kind: ListKind

    @abstractmethod
    def __iter__(self) -> Iterator[Req]: ...

    @abstractmethod
    def _join(self, items, *, at: Optional[int] = None) -> None: ...

    @abstractmethod
    def _leave(self, reqs: Iterable[Req]) -> None: ...

    def req_of(self, item) -> Req:
        """The request an item of this list stands for."""
        return item

    def abort(self, recv_req: AbortReq) -> None:
        """Abort the requests `recv_req` names."""

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _adopt(self, reqs: List[Req]) -> None:
        for req in reqs:
            if req.owner_list is not None:
                raise RuntimeError(
                    f"Request {req.rid} is in {req.owner_list.kind.name}, "
                    f"cannot join {self.kind.name}"
                )
        for req in reqs:
            req.owner_list = self


class KeyedReqList(ReqList):
    """Requests in join order, held in a dict by identity: for a list asked `in`
    often. Changed in place, so a loop that changes the list iterates a copy."""

    def __init__(self, kind: ListKind) -> None:
        self.kind = kind
        self._reqs: Dict[Req, None] = {}

    def _join(self, reqs: Iterable[Req], *, at: Optional[int] = None) -> None:
        if at is not None:
            raise TypeError(f"{self.kind.name} has no positions to insert at")
        reqs = list(reqs)
        self._adopt(reqs)
        for req in reqs:
            self._reqs[req] = None

    def _leave(self, reqs: Iterable[Req]) -> None:
        """Drop the given requests; ones not in are ignored."""
        for req in reqs:
            if req in self._reqs:
                del self._reqs[req]
                req.owner_list = None

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
        # May still hold items whose request left: a leave only clears the
        # owner; the next iteration, index, join or reorder drops them in one
        # pass.
        self._items: List[Any] = []
        self._num_left = 0

    def _compact(self) -> None:
        if self._num_left:
            self._items = [
                item for item in self._items if self._req_of(item).owner_list is self
            ]
            self._num_left = 0

    def _join(self, items: Iterable, *, at: Optional[int] = None) -> None:
        """Add the given items in order: at the back, or before position `at`."""
        items = list(items)
        # Before the adopt: a request rejoining would keep its old item.
        self._compact()
        self._adopt([self._req_of(item) for item in items])
        if at is None:
            self._items.extend(items)
        else:
            self._items[at:at] = items

    def reorder(self, items: Iterable) -> None:
        """Hold the same items in the order given."""
        items = list(items)
        self._compact()
        if len(items) != len(self._items) or {
            id(self._req_of(item)) for item in items
        } != {id(self._req_of(item)) for item in self._items}:
            raise RuntimeError(f"A reorder of {self.kind.name} changed its requests")
        self._items = items

    def _leave(self, reqs: Iterable[Req]) -> None:
        """Drop the items of the given requests; ones not in are ignored."""
        for req in reqs:
            if req.owner_list is self:
                req.owner_list = None
                self._num_left += 1

    def abort(self, recv_req: AbortReq) -> None:
        if self._abort is not None:
            self._abort(recv_req)

    def req_of(self, item) -> Req:
        return self._req_of(item)

    def items(self) -> Iterator:
        self._compact()
        return iter(self._items)

    def __iter__(self) -> Iterator[Req]:
        self._compact()
        return map(self._req_of, self._items)

    def __len__(self) -> int:
        return len(self._items) - self._num_left

    def __contains__(self, req: Req) -> bool:
        return req.owner_list is self

    def __getitem__(self, index):
        self._compact()
        return self._items[index]


def move(items: Iterable, dst: ReqList, *, at: Optional[int] = None) -> None:
    """Move requests into `dst` in one step, out of the list each is in, if
    any. `items` are what `dst` holds: its wrappers for a list that wraps
    requests, else the requests. `at` inserts them before that position of an
    `IndexedReqList` instead of at the back."""
    items = list(items)
    if not items:
        return
    if at is not None and not isinstance(dst, IndexedReqList):
        raise TypeError(f"{dst.kind.name} has no positions to insert at")
    by_owner: Dict[ReqList, List[Req]] = {}
    for item in items:
        req = dst.req_of(item)
        if req.owner_list is dst:
            raise RuntimeError(f"Request {req.rid} is already in {dst.kind.name}")
        if req.owner_list is not None:
            by_owner.setdefault(req.owner_list, []).append(req)
    for owner, reqs in by_owner.items():
        owner._leave(reqs)
    dst._join(items, at=at)
