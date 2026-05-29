"""Proxy that wraps the scheduler's real ``recv_from_tokenizer`` socket.

Installed in place of the raw zmq PULL socket only when the scripted runtime
is active (see ``SchedulerIpcChannels.create``). Unlike a from-scratch
injector, this proxy *wraps* the real PULL socket: requests still travel
the production path (HTTP server -> tokenizer manager -> ZMQ PUSH), and
the proxy buffers whatever it drains off the underlying socket so a
script can control the exact step at which each request becomes visible
to the scheduler.

``ScriptedContext.start_req`` fires a real ``/generate`` HTTP request and
then calls :meth:`wait_until_rid_arrived` to drain the request into the
buffer without yet handing it to the scheduler; the next ``recv_requests``
iteration pops it. Control-class messages (e.g. ``flush_cache``) may still
be injected straight into the buffer via :meth:`inject`.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

import zmq


class ScriptedRidNotArrivedError(TimeoutError):
    """Raised when a rid does not arrive on the socket within the timeout."""


class ScriptedTokenizerRecvProxy:
    """Quacks like a ``zmq.Socket``. NOBLOCK semantics mirror a real PULL
    socket: an empty buffer (after draining the underlying socket) raises
    ``zmq.ZMQError`` with ``EAGAIN``.
    """

    def __init__(self, *, underlying: zmq.Socket) -> None:
        self._underlying = underlying
        self._buffer: deque = deque()

    def recv_pyobj(self, flags: int = 0) -> Any:
        """Pop one buffered request, draining the underlying socket first.

        Non-blocking drain keeps the proxy buffer in sync with whatever
        the tokenizer manager has already pushed, then pops the oldest
        buffered item. An empty buffer under ``NOBLOCK`` raises the same
        ``EAGAIN`` a real PULL socket would.
        """
        self._drain_underlying()

        if self._buffer:
            return self._buffer.popleft()

        if flags & zmq.NOBLOCK:
            raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")
        raise RuntimeError(
            "ScriptedTokenizerRecvProxy.recv_pyobj: blocking recv is not supported"
        )

    def wait_until_rid_arrived(self, rid: str, *, timeout_s: float) -> None:
        """Block until a request carrying ``rid`` has been buffered.

        Repeatedly drains the underlying socket (with a small sleep between
        polls) until a buffered request exposes ``rid``. Every request read
        in the meantime stays buffered, so the arrival ordering the
        scheduler later observes is preserved. Raises
        :class:`ScriptedRidNotArrivedError` if the rid does not arrive in time.
        """
        deadline = time.monotonic() + timeout_s

        while True:
            self._drain_underlying()
            if any(self._req_rid(req) == rid for req in self._buffer):
                return
            if time.monotonic() >= deadline:
                raise ScriptedRidNotArrivedError(
                    f"ScriptedTokenizerRecvProxy: rid {rid!r} did not arrive on the "
                    f"recv_from_tokenizer socket within {timeout_s}s"
                )
            time.sleep(0.005)

    def inject(self, req: Any) -> None:
        """Queue a control message directly; visible on the next ``recv_requests``.

        Used for non-HTTP control messages such as ``FlushCacheReqInput``.
        """
        self._buffer.append(req)

    def _drain_underlying(self) -> None:
        while True:
            try:
                req = self._underlying.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            self._buffer.append(req)

    @staticmethod
    def _req_rid(req: Any) -> Optional[str]:
        return getattr(req, "rid", None)
