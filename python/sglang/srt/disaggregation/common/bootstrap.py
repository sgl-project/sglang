"""Bookkeeping for bootstrap rendezvous before decode has reserved KV.

The transfer backend still owns the destination-pointer handshake and abort
draining. This table only joins a completed optimistic prefill with decode's
control endpoint. It performs no I/O, including while holding its lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

BootstrapNotification = tuple[tuple[str, int], bool] | None


@dataclass
class BootstrapRoom:
    owner: object | None = None
    endpoint: tuple[str, int] | None = None
    ready: bool = False
    failed: bool = False
    expires_at: float = 0.0


class DeferredBootstrap:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.rooms: dict[int, BootstrapRoom] = {}
        self.lock = threading.Lock()
        self._next_cleanup = 0.0

    def _room(self, room: int) -> BootstrapRoom:
        now = time.monotonic()
        if now >= self._next_cleanup:
            # Active senders own their lifetime. Unmatched endpoints and closed
            # rooms expire, including cancellation before the sender exists.
            self.rooms = {
                key: state
                for key, state in self.rooms.items()
                if state.owner is not None or state.expires_at > now
            }
            self._next_cleanup = now + min(1.0, self.timeout)
        return self.rooms.setdefault(room, BootstrapRoom(expires_at=now + self.timeout))

    @staticmethod
    def _notification(state: BootstrapRoom) -> BootstrapNotification:
        if state.endpoint is not None and (state.ready or state.failed):
            return state.endpoint, state.failed
        return None

    def open(self, room: int, owner: object) -> BootstrapRoom | None:
        with self.lock:
            state = self._room(room)
            if state.owner is not None:
                return None
            state.owner = owner
            return state

    def register(self, room: int, endpoint: tuple[str, int]) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            if state.endpoint is not None and state.endpoint != endpoint:
                # A room identifies one request, not a reusable allocation slot.
                # Reject another receiver without replacing the first endpoint.
                return endpoint, True
            state.endpoint = endpoint
            return self._notification(state)

    def complete(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner or state.failed:
                return None
            state.ready = True
            return self._notification(state)

    def fail(self, room: int) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            if state.failed:
                return None
            state.failed = True
            return self._notification(state)

    def close(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner:
                return None
            state.owner = None
            already_failed = state.failed
            state.failed = True
            state.expires_at = time.monotonic() + self.timeout
            # Retain a bounded tombstone: a late endpoint must not wait forever
            # or attach to a second request using the same bootstrap room.
            return None if already_failed else self._notification(state)
