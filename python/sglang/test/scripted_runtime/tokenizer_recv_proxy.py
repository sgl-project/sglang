from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable

import zmq

from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    msgpack_encode,
    sock_recv,
    wrap_as_pickle,
)

_WORK_REQ_TYPES = (
    TokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BatchTokenizedEmbeddingReqInput,
)


class ScriptedTokenizerRecvProxy:

    def __init__(self, *, underlying: zmq.Socket) -> None:
        self._underlying = underlying
        self._buffer: deque = deque()
        self.work_reqs_seen: int = 0

    def recv_pyobj(self, flags: int = 0) -> Any:
        self._drain_underlying()

        return self._pop_buffered(flags, caller="recv_pyobj")

    def recv(self, flags: int = 0) -> bytes:
        self._drain_underlying()

        obj = self._pop_buffered(flags, caller="recv")
        try:
            return msgpack_encode(obj)
        except TypeError:
            return msgpack_encode(wrap_as_pickle(obj))

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
                req = sock_recv(self._underlying, zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            if isinstance(req, _WORK_REQ_TYPES):
                self.work_reqs_seen += 1
            self._buffer.append(req)

    def _pop_buffered(self, flags: int, *, caller: str) -> Any:
        if self._buffer:
            return self._buffer.popleft()

        if flags & zmq.NOBLOCK:
            raise zmq.ZMQError(zmq.EAGAIN, "Resource temporarily unavailable")
        raise RuntimeError(
            f"ScriptedTokenizerRecvProxy.{caller}: blocking recv is not supported"
        )
