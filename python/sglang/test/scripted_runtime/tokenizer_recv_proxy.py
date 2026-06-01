from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import zmq


class ScriptedTokenizerRecvProxy:

    def __init__(self, *, underlying: zmq.Socket) -> None:
        self._underlying = underlying
        self._buffer: deque = deque()

    def recv_pyobj(self, flags: int = 0) -> Any:
        self._drain_underlying()

        if self._buffer:
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
        start_len = len(self._buffer)
        deadline = time.monotonic() + timeout_s
        while True:
            self._drain_underlying()
            for i, obj in enumerate(self._buffer):
                if i >= start_len and predicate(obj):
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
