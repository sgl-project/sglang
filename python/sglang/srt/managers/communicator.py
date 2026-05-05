from __future__ import annotations

import asyncio
import copy
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

import zmq

from sglang.srt.managers.io_struct import sock_send

T = TypeVar("T")


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Event] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj is not None:
            self.send_obj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj is not None:
                self.send_obj(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        if self._result_event is event:
            self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    def send_obj(self, obj):
        # Hicky solution to compatible with the SendWrapper from TokenizerManager
        if hasattr(self._sender, "send_obj"):
            self._sender.send_obj(obj)
        else:
            sock_send(self._sender, obj)

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
