"""In-process communication channels for ThreadedEngine (no-GIL mode).

Replaces ZMQ IPC sockets with thread-safe queues when scheduler and
detokenizer run as threads instead of processes.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Optional

import zmq

_ZMQ_NOBLOCK = zmq.NOBLOCK


class _Sent:
    """Awaitable no-op so that ``await sender.send_pyobj(...)`` works."""

    __slots__ = ()

    def __await__(self):
        return
        yield  # pragma: no cover — makes this a generator-based coroutine


_SENT = _Sent()


class QueueSender:
    """Drop-in for zmq.Socket.send_pyobj on the sender side.

    Returns an awaitable so both sync and async call-sites work:
      sender.send_pyobj(obj)          # fire-and-forget (FanOutCommunicator)
      await sender.send_pyobj(obj)    # TokenizerManager async path
    """

    def __init__(self, q: queue.SimpleQueue):
        self._q = q

    def send_pyobj(self, obj: Any, flags: int = 0, **kwargs) -> _Sent:
        self._q.put(obj)
        return _SENT


class QueueReceiver:
    """Drop-in for zmq.Socket.recv_pyobj on the receiver side.

    Translates zmq.NOBLOCK to get_nowait and raises zmq.ZMQError on empty
    (matching the scheduler's existing except zmq.ZMQError handling).
    """

    def __init__(self, q: queue.SimpleQueue):
        self._q = q

    def recv_pyobj(self, flags: int = 0) -> Any:
        if flags & _ZMQ_NOBLOCK:
            try:
                return self._q.get_nowait()
            except queue.Empty:
                raise zmq.ZMQError()
        return self._q.get()


class AsyncQueueReceiver:
    """Async-compatible drop-in for zmq.asyncio.Socket.recv_pyobj.

    TokenizerManager uses ``await recv_from_detokenizer.recv_pyobj()``
    inside an asyncio event loop.  A background bridge thread drains the
    sync queue into an asyncio.Queue via call_soon_threadsafe.
    """

    def __init__(self, sync_q: queue.SimpleQueue):
        self._sync_q = sync_q
        self._async_q: Optional[asyncio.Queue] = None
        self._bridge_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_bridge(self):
        if self._bridge_thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._async_q = asyncio.Queue()

        def _bridge():
            loop = self._loop
            async_q = self._async_q
            sync_q = self._sync_q
            while True:
                obj = sync_q.get()
                loop.call_soon_threadsafe(async_q.put_nowait, obj)

        t = threading.Thread(target=_bridge, name="channel-bridge", daemon=True)
        t.start()
        self._bridge_thread = t

    async def recv_pyobj(self, flags: int = 0) -> Any:
        self._ensure_bridge()
        return await self._async_q.get()


class ChannelPair:
    """A unidirectional channel: one sender, one receiver."""

    def __init__(self):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self.sender = QueueSender(self._q)
        self.receiver = QueueReceiver(self._q)
        self.async_receiver = AsyncQueueReceiver(self._q)


class ChannelHub:
    """All channels matching the ZMQ IPC topology for single-node config.

    Attributes expose sender/receiver endpoints that replace ZMQ sockets
    in the respective manager objects.
    """

    def __init__(self):
        self.tokenizer_to_scheduler = ChannelPair()
        self.scheduler_to_detokenizer = ChannelPair()

        self._to_tokenizer = ChannelPair()
        self.detokenizer_to_tokenizer = self._to_tokenizer
        self.scheduler_to_tokenizer = self._to_tokenizer

        self.rpc = ChannelPair()
