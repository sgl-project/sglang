from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import Req


class ListKind(enum.IntEnum):
    """The lists that hold the live requests. Values below 16 are the core's own
    lists; the lists of other owners take values from 16 up."""

    WAITING = 1
    # The running requests.
    DECODING = 2
    CHUNKED = 3
    PD_BOOTSTRAP = 16
    # Prefill requests out of the chunked slot whose launched steps are not done.
    PD_PREFILLING = 17
    PD_SENDING = 18
    PD_PREALLOC = 24
    PD_RETRACTED = 25
    PD_HELD = 26
    PD_RECEIVING = 27
    PD_RELEASING = 29
    HISPARSE_STAGING = 32
    GRAMMAR = 36
    DLLM = 40
    UOCR_PARKED = 48
    DEFERRED = 56


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

    def take(self, pred: Callable[[Req], bool]) -> List[Req]:
        """Drop the requests matching `pred` and return them, in list order."""
        taken = [req for req in self if pred(req)]
        self.leave(taken)
        return taken

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _RequestList(RequestList):
    """Requests in join order, held by identity. Each change rebinds a new dict,
    so leaving while iterating is safe."""

    def __init__(
        self,
        kind: ListKind,
        *,
        abort: Optional[Callable[[AbortReq], None]] = None,
    ) -> None:
        self.kind = kind
        self._abort = abort
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

    def abort(self, recv_req: AbortReq) -> None:
        if self._abort is not None:
            self._abort(recv_req)

    def __iter__(self) -> Iterator[Req]:
        return iter(self._reqs)

    def __len__(self) -> int:
        return len(self._reqs)

    def __contains__(self, req: Req) -> bool:
        return req in self._reqs

    def _drop(self, leaving: Set[Req]) -> None:
        if leaving:
            self._reqs = {req: None for req in self._reqs if req not in leaving}


def move(reqs: Iterable[Req], src: RequestList, dst: RequestList) -> None:
    """Move requests from `src` to `dst`; each must be in `src`."""
    reqs = list(reqs)
    missing = [req.rid for req in reqs if req not in src]
    if missing:
        raise RuntimeError(f"Requests {missing} are not in {src.kind.name}")
    src.leave(reqs)
    dst.join(reqs)
