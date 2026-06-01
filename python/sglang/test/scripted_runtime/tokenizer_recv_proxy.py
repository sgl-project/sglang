from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, List

import zmq


class ScriptedTokenizerRecvProxy:

    def __init__(self, *, underlying: zmq.Socket) -> None:
        self._underlying = underlying
        self._buffer: deque = deque()

    def buffered_objects_for_rid(self, rid: str) -> List[Any]:
        # Read-only snapshot of every batch output the scheduler has sent toward
        # the tokenizer that carries this rid. Draining here is non-destructive:
        # objects stay in the buffer so the engine's own recv path still sees
        # them. A batch output carries this rid when `rid` is in its `rids` list.
        self._drain_underlying()
        return [
            obj
            for obj in self._buffer
            if rid in (getattr(obj, "rids", None) or [])
        ]

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
