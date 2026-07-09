from __future__ import annotations

import asyncio
import copy
from typing import Callable, Generic, List, Optional, TypeVar

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

    def __init__(
        self,
        send: Callable[[T], None],
        fan_out: int,
        mode: str = "queueing",
    ):
        self._send = send
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        # Serialize queueing-mode callers. asyncio.Lock is FIFO-fair and atomic, which
        # avoids the slot-handoff race a hand-rolled ready-queue had: a fresh caller
        # could observe the just-freed slot (result_event is None and empty queue) and
        # claim it before a woken waiter resumed, tripping its `assert result_event is
        # None` and returning a 500 (see test_get_server_info_concurrent).
        self._lock = asyncio.Lock()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        # Shield the in-flight lifecycle from caller cancellation. If the caller is
        # cancelled (e.g. client disconnect), _call keeps holding the lock until this
        # request's response is fully received and the state is cleared, so the next
        # caller cannot overwrite result_event/result_values and receive a stale
        # response. Preserves the invariant: only one request is in flight at a time.
        async def _call():
            async with self._lock:
                if obj is not None:
                    self._send(obj)

                self._result_event = asyncio.Event()
                self._result_values = []
                try:
                    await self._result_event.wait()
                    return self._result_values
                finally:
                    self._result_event = self._result_values = None

        return await asyncio.shield(_call())

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj is not None:
                self._send(obj)

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

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
