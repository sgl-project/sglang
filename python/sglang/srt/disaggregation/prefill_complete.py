"""Readiness control messages for deferred decode KV allocation.

Only live senders and receivers own state. A subscription arriving before its
sender gets a retry response rather than creating an orphan room. Readiness
does not authorize a KV write: the normal destination metadata handshake still
does that, and its existing abort/drain protocol remains authoritative.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Callable

import msgspec

logger = logging.getLogger(__name__)

HEADER = b"PREFILL_COMPLETE_V1"
INITIAL_RETRY_INTERVAL = 0.001
MAX_RETRY_INTERVAL = 0.1
Endpoint = tuple[str, int]


class SourceReadiness(msgspec.Struct, eq=False):
    ready: bool = False
    failure: str | None = None
    subscriber: tuple[Endpoint, bytes] | None = None


class DestinationReadiness(msgspec.Struct, eq=False):
    endpoint: Endpoint
    nonce: bytes
    deadline: float
    ready: bool = False
    failure: str | None = None
    subscribed: bool = False
    next_subscribe: float = 0.0
    retry_interval: float = INITIAL_RETRY_INTERVAL


class PrefillCompleteManager:
    def __init__(
        self,
        *,
        send: Callable[[Endpoint, list[bytes]], None],
        on_cancel: Callable[[int], None],
    ):
        self._send = send
        self._on_cancel = on_cancel
        self._lock = threading.RLock()
        self._sources: dict[int, SourceReadiness] = {}
        self._destinations: dict[int, DestinationReadiness] = {}

    def add_source(self, *, room: int) -> SourceReadiness:
        with self._lock:
            if room in self._sources:
                raise ValueError(f"Duplicate live prefill bootstrap room {room}")
            state = SourceReadiness()
            self._sources[room] = state
            return state

    def mark_ready(self, *, room: int, state: SourceReadiness) -> None:
        with self._lock:
            if self._sources.get(room) is not state or state.failure is not None:
                return
            state.ready = True
            subscriber = state.subscriber
        if subscriber is not None:
            self._reply(room=room, subscriber=subscriber, status=b"READY")

    def fail_source(self, *, room: int, reason: str) -> None:
        with self._lock:
            state = self._sources.get(room)
            if state is None:
                return
            state.failure = reason
            subscriber = state.subscriber
        if subscriber is not None:
            self._reply(
                room=room, subscriber=subscriber, status=b"ERROR", reason=reason
            )

    def remove_source(self, *, room: int, state: SourceReadiness) -> None:
        with self._lock:
            if self._sources.get(room) is state:
                del self._sources[room]

    def add_destination(
        self, *, room: int, endpoint: Endpoint, timeout: float
    ) -> DestinationReadiness:
        with self._lock:
            if room in self._destinations:
                raise ValueError(f"Duplicate live decode bootstrap room {room}")
            state = DestinationReadiness(
                endpoint=endpoint,
                nonce=uuid.uuid4().hex.encode(),
                deadline=time.monotonic() + timeout,
            )
            self._destinations[room] = state
            return state

    def poll(
        self,
        *,
        room: int,
        state: DestinationReadiness,
        local_endpoint: Endpoint,
        awaiting_completion: bool = True,
    ) -> tuple[bool, str | None]:
        now = time.monotonic()
        with self._lock:
            if self._destinations.get(room) is not state:
                return False, "Readiness receiver has already been cleared"
            if state.failure is not None:
                return False, state.failure
            if not awaiting_completion:
                return True, None
            if now >= state.deadline:
                state.failure = "Timed out waiting for prefill completion"
                return False, state.failure
            if state.ready:
                return True, None
            subscribe = not state.subscribed and now >= state.next_subscribe
            if subscribe:
                # Decode may arrive just before the source request is created.
                # Retry that short race promptly, but back off for a missing
                # source. A WAIT/READY reply stops subscription retries entirely.
                state.next_subscribe = now + state.retry_interval
                state.retry_interval = min(state.retry_interval * 2, MAX_RETRY_INTERVAL)
        if subscribe:
            try:
                self._send(
                    state.endpoint,
                    [
                        HEADER,
                        b"SUBSCRIBE",
                        str(room).encode(),
                        state.nonce,
                        local_endpoint[0].encode(),
                        str(local_endpoint[1]).encode(),
                    ],
                )
            except Exception as exc:
                return False, f"Could not subscribe to prefill completion: {exc}"
        return False, None

    def remove_destination(self, *, room: int, state: DestinationReadiness) -> None:
        with self._lock:
            if self._destinations.get(room) is state:
                del self._destinations[room]

    def handle_message(self, msg: list[bytes]) -> bool:
        if not msg or msg[0] != HEADER:
            return False
        try:
            if len(msg) != 6:
                raise ValueError("expected six readiness frames")
            room = int(msg[2])
            if msg[1] == b"SUBSCRIBE":
                self._subscribe(
                    room=room,
                    subscriber=((msg[4].decode(), int(msg[5])), msg[3]),
                )
            elif msg[1] == b"CANCEL":
                with self._lock:
                    source = self._sources.get(room)
                    matches = (
                        source is not None
                        and source.subscriber is not None
                        and source.subscriber[1] == msg[3]
                    )
                    if matches:
                        # Keep the nonce check and cancellation atomic against
                        # clear/reuse of this room. The callback may mark this
                        # same source failed, hence the reentrant lock.
                        self._on_cancel(room)
            elif msg[1] in (b"READY", b"WAIT", b"RETRY", b"ERROR"):
                with self._lock:
                    state = self._destinations.get(room)
                    if state is None or state.nonce != msg[3]:
                        return True
                    if msg[1] == b"ERROR":
                        state.failure = msg[4].decode(errors="replace")
                    elif msg[1] == b"READY":
                        state.ready = True
                        state.subscribed = True
                    elif msg[1] == b"WAIT":
                        state.subscribed = True
            else:
                raise ValueError("unknown readiness message")
        except (ValueError, UnicodeError) as exc:
            logger.warning("Ignoring malformed prefill-complete message: %s", exc)
        return True

    def _subscribe(self, *, room: int, subscriber: tuple[Endpoint, bytes]) -> None:
        reason = ""
        with self._lock:
            state = self._sources.get(room)
            if state is None:
                status = b"RETRY"
            elif state.subscriber not in (None, subscriber):
                status = b"ERROR"
                reason = "Bootstrap room already belongs to another decode request"
            else:
                state.subscriber = subscriber
                if state.failure is not None:
                    status, reason = b"ERROR", state.failure
                else:
                    status = b"READY" if state.ready else b"WAIT"
        self._reply(room=room, subscriber=subscriber, status=status, reason=reason)

    def _reply(
        self,
        *,
        room: int,
        subscriber: tuple[Endpoint, bytes],
        status: bytes,
        reason: str = "",
    ) -> None:
        endpoint, nonce = subscriber
        try:
            self._send(
                endpoint,
                [HEADER, status, str(room).encode(), nonce, reason.encode(), b""],
            )
        except Exception:
            # A lost peer must not kill the shared control thread. The receiver
            # and sender retain their independent bounded handshake deadlines.
            logger.exception("Could not send prefill-complete status for %s", room)
