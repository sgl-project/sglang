"""Bounded, cancellation-safe admission before GPU feature publication."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


class MmMemoryRequestTooLarge(ValueError):
    status_code = 413
    error_code = "mm_memory_request_too_large"


class MmMemoryOverloaded(ValueError):
    status_code = 429
    error_code = "mm_memory_overloaded"


@dataclass(eq=False)
class _PublicationTicket:
    nbytes: int
    changed: asyncio.Event = field(default_factory=asyncio.Event)


class FeaturePublicationAdmission:
    """Bound pending GPU payloads until publication actually finishes.

    Only the FIFO head attempts an atomic pool reservation. Waiting owns no
    pool slices, and an admitted ticket remains charged until its native copy
    finishes, even if the request coroutine has been cancelled.
    """

    def __init__(self, capacity_bytes: int, timeout_ms: int, max_pending: int = 64):
        self.capacity_bytes = capacity_bytes
        self.timeout_ms = timeout_ms
        self.max_pending = max_pending
        self.pending_bytes = 0
        self.pending_count = 0
        self._waiting = deque()
        self._loop = None
        self._closed = False
        self.observer = None

    def _observe(self, outcome=None, wait_seconds=None):
        if self.observer is not None:
            self.observer(outcome, wait_seconds, self.pending_count, self.pending_bytes)

    def notify(self) -> None:
        """Wake the FIFO head from the recycler or publisher thread."""
        if self._loop is not None and not self._closed:
            try:
                self._loop.call_soon_threadsafe(self._wake_head)
            except RuntimeError:
                if not self._loop.is_closed():
                    raise

    def finish_from_thread(self, ticket: _PublicationTicket) -> None:
        """Native completion owns cleanup even after the request loop shuts down."""
        try:
            self._loop.call_soon_threadsafe(self.finish, ticket)
        except RuntimeError:
            if not self._loop.is_closed():
                raise
            # the closed loop cannot run another request or touch these counters
            self.pending_count -= 1
            self.pending_bytes -= ticket.nbytes

    def _wake_head(self) -> None:
        self._observe()
        if self._waiting:
            self._waiting[0].changed.set()

    async def acquire(self, nbytes: int, try_reserve):
        """Return a reservation and its ticket; the caller must finish the ticket."""
        if nbytes > self.capacity_bytes:
            self._observe("oversized")
            raise MmMemoryRequestTooLarge(
                f"Multimodal publication requires {nbytes} bytes, but the worker "
                f"pool can hold only {self.capacity_bytes} bytes"
            )
        if self._closed:
            self._observe("shutdown")
            raise MmMemoryOverloaded("Multimodal feature publisher is shutting down")
        if (
            self.pending_count >= self.max_pending
            or self.pending_bytes + nbytes > self.capacity_bytes
        ):
            self._observe("queue_full")
            raise MmMemoryOverloaded("Multimodal publication queue is full")

        self._loop = asyncio.get_running_loop()
        ticket = _PublicationTicket(nbytes)
        self.pending_count += 1
        self.pending_bytes += nbytes
        self._waiting.append(ticket)
        started = time.monotonic()
        deadline = started + self.timeout_ms / 1000
        self._observe()
        try:
            while True:
                if self._closed:
                    self._observe("shutdown")
                    raise MmMemoryOverloaded(
                        "Multimodal feature publisher is shutting down"
                    )
                # clear before checking capacity so a concurrent recycle cannot
                # disappear between the failed reservation and the wait
                ticket.changed.clear()
                if self._waiting[0] is ticket:
                    reservation = try_reserve()
                    if reservation is not None:
                        self._waiting.popleft()
                        self._observe("admitted", time.monotonic() - started)
                        self._wake_head()
                        return reservation, ticket
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._observe("timeout", time.monotonic() - started)
                    raise MmMemoryOverloaded(
                        "Timed out waiting for multimodal feature pool capacity"
                    )
                try:
                    await asyncio.wait_for(ticket.changed.wait(), remaining)
                except asyncio.TimeoutError as error:
                    self._observe("timeout", time.monotonic() - started)
                    raise MmMemoryOverloaded(
                        "Timed out waiting for multimodal feature pool capacity"
                    ) from error
        except BaseException:
            self._waiting.remove(ticket)
            self.finish(ticket)
            raise

    def finish(self, ticket: _PublicationTicket) -> None:
        """Release payload accounting on the event-loop thread, after native work."""
        self.pending_count -= 1
        self.pending_bytes -= ticket.nbytes
        self._wake_head()

    def close(self) -> None:
        self._closed = True
        for ticket in self._waiting:
            ticket.changed.set()
