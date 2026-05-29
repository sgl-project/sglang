"""Proxy for the scheduler's ``recv_from_tokenizer`` socket.

Installed in place of the real zmq PULL socket only when ScriptedRuntime
is active (see ``SchedulerIpcChannels.create``). Serves requests from an
in-process queue populated by ``ScriptedRuntime.start_req``, so a script
fully controls request arrival instead of the HTTP server / tokenizer
manager. The replaced raw socket is retained as ``underlying`` purely so
its bound IPC endpoint stays open for the tokenizer manager to connect to.
"""

from collections import deque
from typing import Any, Optional

import zmq


class TokenizerRecvProxy:
    """Quacks like a ``zmq.Socket``. NOBLOCK semantics mirror a real PULL
    socket: an empty queue raises ``zmq.ZMQError`` with ``EAGAIN``.
    """

    def __init__(self, *, underlying: Optional[zmq.Socket]) -> None:
        self._underlying = underlying
        self._queue: deque = deque()

    def recv_pyobj(self, flags: int = 0) -> Any:
        if self._queue:
            return self._queue.popleft()

        if flags & zmq.NOBLOCK:
            raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")
        raise RuntimeError(
            "TokenizerRecvProxy.recv_pyobj: blocking recv is not supported"
        )

    def inject(self, req: Any) -> None:
        """Queue a tokenized request; visible on the next ``recv_requests``."""
        self._queue.append(req)
