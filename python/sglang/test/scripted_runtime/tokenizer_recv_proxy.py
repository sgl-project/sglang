"""Proxy that wraps the scheduler's real ``recv_from_tokenizer`` socket.

Installed in place of the raw zmq PULL socket only when the scripted runtime
is active (see ``SchedulerIpcChannels.create``). Unlike a from-scratch
injector, this proxy *wraps* the real PULL socket: requests still travel
the production path (HTTP server -> tokenizer manager -> ZMQ PUSH), and
the proxy buffers whatever it drains off the underlying socket so a
script can control the exact step at which each request becomes visible
to the scheduler.

``ScriptedContext.start_req`` and the control verbs (flush_cache / abort_all
/ pause / continue) all fire a real HTTP POST and then call
:meth:`wait_until_arrived` with a predicate (matching the rid for start_req,
the message type for the control verbs) to drain the resulting object into
the buffer without yet handing it to the scheduler; the next ``recv_requests``
iteration pops it.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import zmq


class ScriptedTokenizerRecvProxy:
    """Quacks like a ``zmq.Socket``. NOBLOCK semantics mirror a real PULL
    socket: an empty buffer (after draining the underlying socket) raises
    ``zmq.ZMQError`` with ``EAGAIN``.
    """

    def __init__(self, *, underlying: zmq.Socket) -> None:
        self._underlying = underlying
        self._buffer: deque = deque()
        # Monotonic count of items ever drained off the underlying socket, and
        # the count ever popped via ``recv_pyobj``. Together they give each
        # buffered item a stable drain-sequence number (``_popped_count + i``
        # for buffer position ``i``), so a type-based wait can require a
        # *newly*-drained match rather than re-matching a stale buffered object.
        self._total_drained = 0
        self._popped_count = 0

    def recv_pyobj(self, flags: int = 0) -> Any:
        """Pop one buffered request, draining the underlying socket first.

        Non-blocking drain keeps the proxy buffer in sync with whatever
        the tokenizer manager has already pushed, then pops the oldest
        buffered item. An empty buffer under ``NOBLOCK`` raises the same
        ``EAGAIN`` a real PULL socket would.
        """
        self._drain_underlying()

        if self._buffer:
            self._popped_count += 1
            return self._buffer.popleft()

        if flags & zmq.NOBLOCK:
            raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")
        raise RuntimeError(
            "ScriptedTokenizerRecvProxy.recv_pyobj: blocking recv is not supported"
        )

    def wait_until_arrived(
        self,
        predicate: Callable[[Any], bool],
        *,
        timeout_s: float,
        description: str = "matching object",
    ) -> None:
        """Block until a *newly*-drained object satisfies ``predicate``.

        Used by every script verb that injects through the real HTTP path
        (start_req matches the rid; flush_cache / abort_all / pause / continue
        match the message type). Only objects drained at or after this call
        starts are eligible, so a stale buffered object of the same kind from
        an earlier verb is not mistaken for this one. Every drained object
        stays buffered, so the scheduler still observes it on the next
        ``recv_requests``. ``description`` names the awaited object in the
        timeout error.
        """
        start_seq = self._total_drained
        deadline = time.monotonic() + timeout_s
        while True:
            self._drain_underlying()
            for i, obj in enumerate(self._buffer):
                if self._popped_count + i >= start_seq and predicate(obj):
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ScriptedTokenizerRecvProxy: no {description} arrived on the "
                    f"recv_from_tokenizer socket within {timeout_s}s"
                )
            time.sleep(0.005)

    def _drain_underlying(self) -> None:
        while True:
            try:
                req = self._underlying.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            self._buffer.append(req)
            self._total_drained += 1
