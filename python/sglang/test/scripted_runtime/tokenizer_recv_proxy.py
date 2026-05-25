"""Proxy for the scheduler's ``recv_from_tokenizer`` socket.

In production the proxy is a passthrough wrapper around the real zmq socket.
In test mode (when a :class:`ScriptedRuntime` is active) the proxy ignores the
underlying socket and instead serves requests injected directly into an
in-process queue by ``ScriptedRuntime.start_req``. This bypasses the HTTP
server and tokenizer manager entirely — fine-grained scheduler tests are
not concerned with those layers.
"""

from collections import deque
from typing import Any, Optional

import zmq


class TokenizerRecvProxy:
    """Quacks like a ``zmq.Socket`` for the ``recv_pyobj`` calls in
    ``SchedulerRequestReceiver._pull_raw_reqs``.

    Test-mode behavior matches the real socket's contract used by the
    scheduler: ``recv_pyobj(zmq.NOBLOCK)`` either returns a queued object
    or raises ``zmq.ZMQError`` with ``EAGAIN``, just like a real PULL
    socket with no pending message.
    """

    def __init__(self, *, underlying: Optional[zmq.Socket], test_mode: bool):
        self._underlying = underlying
        self._test_mode = test_mode
        self._queue: deque = deque()

    @property
    def test_mode(self) -> bool:
        return self._test_mode

    def recv_pyobj(self, flags: int = 0) -> Any:
        if not self._test_mode:
            return self._underlying.recv_pyobj(flags)

        if self._queue:
            return self._queue.popleft()

        if flags & zmq.NOBLOCK:
            raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")
        raise RuntimeError(
            "TokenizerRecvProxy.recv_pyobj: blocking recv is not supported in test mode"
        )

    def inject(self, req: Any) -> None:
        """Place a tokenized request into the in-process delivery queue.

        The request becomes visible to the scheduler on its next
        ``recv_requests`` iteration.
        """
        assert self._test_mode, "TokenizerRecvProxy.inject only valid in test mode"
        self._queue.append(req)
