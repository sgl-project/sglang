"""Scheduler-owned retirement callbacks for source buffers still used by DMA."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from sglang.srt.managers.scheduler_components.request_list import (
    KeyedReqList,
    ListKind,
)

logger = logging.getLogger(__name__)


@dataclass
class _SourceRelease:
    req: Any
    callbacks: dict[str, Callable[[], None]] = field(default_factory=dict)
    release_failed: bool = False


class SourceReleaseTracker:
    """Keep the original request ownership until every sender rank is idle.

    Every retiring request is registered, including locally drained requests,
    so ranks reduce the same list before changing their allocators. Callbacks
    run only on the scheduler owner. A partially failed release is quarantined,
    never retried; destructive operations remain blocked.
    """

    def __init__(self):
        import threading

        self._owner = threading.get_ident()
        self.requests = KeyedReqList(ListKind.PD_RELEASING)
        self._holds: dict[int, _SourceRelease] = {}
        self._releasing: int | None = None

    def _require_owner(self):
        import threading

        if threading.get_ident() != self._owner:
            raise RuntimeError("Source releases must run on the scheduler owner")

    def defer(self, req, key: str, callback: Callable[[], None]) -> bool:
        self._require_owner()
        room = req.bootstrap_room
        if self._releasing == room:
            if self._holds[room].req is not req:
                raise RuntimeError("A retained source room cannot be reused")
            return False
        hold = self._holds.setdefault(room, _SourceRelease(req))
        if hold.req is not req:
            raise RuntimeError("A retained source room cannot be reused")
        hold.callbacks.setdefault(key, callback)
        self.requests.join([req])
        return True

    def collect(self, is_safe, reduce_flags):
        self._require_owner()
        # Callback arrival order is not a collective contract. Bootstrap-room
        # order is deterministic within one attention DP shard.
        holds = sorted(self._holds.items())
        ready = reduce_flags(
            [int(not hold.release_failed and is_safe(room)) for room, hold in holds]
        )
        if len(ready) != len(holds):
            raise RuntimeError("Source release consensus changed its cardinality")
        completed = []
        for (room, hold), release in zip(holds, ready):
            if not release:
                completed.append(0)
                continue
            # A callback may have partially freed native state before raising.
            # Keep the request visible and never repeat any callback in it.
            hold.release_failed = True
            self._releasing = room
            try:
                for callback in tuple(hold.callbacks.values()):
                    callback()
            except Exception:
                logger.exception("Source release failed for room %s; quarantined", room)
                completed.append(0)
            else:
                completed.append(1)
            finally:
                self._releasing = None
        # A native release failure is fatal on every sender rank. Some peers
        # may already have freed slots, so continuing admission would diverge
        # allocators. Keep the holds and never retry possibly partial callbacks.
        completed = reduce_flags(completed)
        if len(completed) != len(holds):
            raise RuntimeError(
                "Source release completion consensus changed cardinality"
            )
        failed = False
        for (room, hold), release, success in zip(holds, ready, completed):
            if release and success:
                self.requests.leave([hold.req])
                del self._holds[room]
            elif release:
                failed = True
        if failed:
            raise RuntimeError(
                "Source release failed on a sender rank; restart required"
            )

    def __bool__(self):
        return bool(self._holds)

    def holds_room(self, room: int) -> bool:
        return room in self._holds
