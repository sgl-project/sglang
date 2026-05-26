"""Proxy for the scheduler's ``recv_from_tokenizer`` socket.

Production: passthrough to the zmq socket. Test (ScriptedRuntime
active): serve from an in-process queue populated by
``ScriptedRuntime.start_req``, bypassing the HTTP server and
tokenizer manager.
"""

from collections import deque
from typing import Any, Optional

import zmq


class TokenizerRecvProxy:
    """Quacks like a ``zmq.Socket``. NOBLOCK semantics mirror a real
    PULL socket: empty queue raises ``zmq.ZMQError`` with ``EAGAIN``.
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
        """Queue a tokenized request; visible on the next ``recv_requests``."""
        assert self._test_mode, "TokenizerRecvProxy.inject only valid in test mode"
        self._queue.append(req)
