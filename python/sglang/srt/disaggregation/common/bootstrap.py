"""Bookkeeping for bootstrap rendezvous before decode has reserved KV.

The transfer backend still owns the destination-pointer handshake and abort
draining. This table only joins a completed optimistic prefill with decode's
control endpoint. It performs no I/O, including while holding its lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import msgspec

BootstrapNotification = tuple[tuple[str, int], bool] | None


class DecodeHandshake(msgspec.Struct, frozen=True):
    endpoint: tuple[str, int]
    # From the readiness message to decode's destination; a wait for decode's
    # registration is not part of it.
    seconds: float
    # time.monotonic() of decode's first registration of the room.
    registered_at: float


@dataclass
class BootstrapRoom:
    owner: object | None = None
    endpoint: tuple[str, int] | None = None
    ready: bool = False
    ready_notified: bool = False
    failed: bool = False
    expires_at: float = 0.0
    # time.monotonic() of decode's first registration, of the readiness message,
    # sent once both the source is ready and decode has registered, and of
    # decode's destination.
    registered_at: float | None = None
    ready_notified_at: float | None = None
    destination_at: float | None = None
    # The transfer succeeded and decode has been told; the owner has not
    # released the room yet.
    concluded: bool = False
    # Decode's registration of the room's next request, which arrived while
    # this one was concluded but not released.
    next_endpoint: tuple[str, int] | None = None
    next_registered_at: float | None = None


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
        if state.endpoint is not None:
            if state.failed:
                return state.endpoint, True
            if state.ready and not state.ready_notified:
                # One readiness message per transfer and PP stage, sent before
                # that stage's Success; so none can trail the room's Success
                # and accidentally admit the next transfer using this room.
                state.ready_notified = True
                state.ready_notified_at = time.monotonic()
                return state.endpoint, False
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
            if state.concluded:
                # Decode heard Success, so this registers the room's next
                # request; it applies once this one is released.
                if state.next_endpoint is None:
                    state.next_endpoint = endpoint
                    state.next_registered_at = time.monotonic()
                return None
            if state.endpoint is not None and state.endpoint != endpoint:
                # A room identifies one request, not a reusable allocation slot.
                # Reject another receiver without replacing the first endpoint.
                return endpoint, True
            if state.endpoint is None:
                state.registered_at = time.monotonic()
            state.endpoint = endpoint
            return self._notification(state)

    def complete(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner or state.failed:
                return None
            state.ready = True
            return self._notification(state)

    def conclude(self, room: int) -> bool:
        """Commit the room's transfer success before decode is told. False when
        a failure or cancellation won, which then owns decode's notification."""
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is None or state.failed:
                return False
            state.concluded = True
            return True

    def awaits_release(self, room: int) -> bool:
        """Whether the room's previous request succeeded but is not released;
        its next request must not open the room until then."""
        with self.lock:
            state = self.rooms.get(room)
            return state is not None and state.owner is not None and state.concluded

    def mark_destination(self, room: int) -> None:
        with self.lock:
            state = self.rooms.get(room)
            if state is not None and state.destination_at is None:
                state.destination_at = time.monotonic()

    def handshake(self, room: int, owner: object) -> DecodeHandshake | None:
        with self.lock:
            state = self.rooms.get(room)
            if (
                state is None
                or state.owner is not owner
                or state.endpoint is None
                or state.registered_at is None
                or state.ready_notified_at is None
                or state.destination_at is None
            ):
                return None
            return DecodeHandshake(
                endpoint=state.endpoint,
                seconds=state.destination_at - state.ready_notified_at,
                registered_at=state.registered_at,
            )

    def fail(self, room: int) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            if state.failed or state.concluded:
                # Decode already holds a concluded transfer's Success.
                return None
            state.failed = True
            return self._notification(state)

    def cancel(self, room: int) -> None:
        """Record decode's ABORT; on a concluded room it cancels the room's
        next request, the only one decode can still abort."""
        with self.lock:
            self._room(room).failed = True

    def close(
        self, room: int, owner: object, *, success: bool = False
    ) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner:
                return None
            if not state.failed and (success or state.concluded):
                # The transfer worker already told decode. A registration of the
                # room's next request that arrived since starts that room.
                if state.next_endpoint is None:
                    del self.rooms[room]
                else:
                    self.rooms[room] = BootstrapRoom(
                        endpoint=state.next_endpoint,
                        registered_at=state.next_registered_at,
                        expires_at=time.monotonic() + self.timeout,
                    )
                return None
            state.owner = None
            state.concluded = False
            already_failed = state.failed
            state.failed = True
            state.expires_at = time.monotonic() + self.timeout
            # Retain a bounded tombstone: a late endpoint must not wait forever
            # or attach to a second request using the same bootstrap room.
            return None if already_failed else self._notification(state)
