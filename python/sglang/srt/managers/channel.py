"""In-process communication channels for free-threaded (no-GIL) mode.

Replaces ZMQ IPC sockets with thread-safe queues when scheduler and
detokenizer run as threads instead of processes.

Usage: create a ChannelHub, then wire its senders/receivers into the
existing Scheduler, DetokenizerManager, and TokenizerManager objects
in place of ZMQ sockets.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, Optional

import zmq

# Cache zmq.NOBLOCK at module level to avoid per-call attribute lookup
_ZMQ_NOBLOCK = zmq.NOBLOCK


class QueueSender:
    """Drop-in for ``zmq.Socket.send_pyobj`` on the sender side."""

    def __init__(self, q: queue.SimpleQueue):
        self._q = q

    def send_pyobj(self, obj: Any) -> None:
        self._q.put(obj)


class QueueReceiver:
    """Drop-in for ``zmq.Socket.recv_pyobj`` on the receiver side.

    The scheduler poll loop passes ``zmq.NOBLOCK`` – we translate that to
    ``get_nowait`` and raise ``zmq.ZMQError`` (caught by ``except zmq.ZMQError``).
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
    """Async-compatible drop-in for ``zmq.asyncio.Socket.recv_pyobj``.

    TokenizerManager uses ``await recv_from_detokenizer.recv_pyobj()``
    inside an asyncio event loop.  We use an ``asyncio.Queue`` bridged
    from the sync side via ``call_soon_threadsafe`` to avoid the
    overhead of ``run_in_executor`` (which creates a thread-pool task
    and an extra context switch per receive).
    """

    def __init__(self, sync_q: queue.SimpleQueue):
        self._sync_q = sync_q
        self._async_q: Optional[asyncio.Queue] = None
        self._bridge_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_bridge(self):
        """Start a background thread that drains the sync queue into the
        asyncio queue via ``call_soon_threadsafe``.  Started lazily on
        first recv so we have a running event loop to target."""
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
    """All channels matching the ZMQ IPC topology for a single-node,
    single-TP, single-PP configuration.

    Attributes expose the sender/receiver endpoints that should be wired
    into the respective manager objects.
    """

    def __init__(self):
        self.tokenizer_to_scheduler = ChannelPair()
        self.scheduler_to_detokenizer = ChannelPair()

        # Scheduler and detokenizer both send to tokenizer via the same
        # queue — mirrors the ZMQ topology where both PUSH to the same port.
        self._to_tokenizer = ChannelPair()
        self.detokenizer_to_tokenizer = self._to_tokenizer
        self.scheduler_to_tokenizer = self._to_tokenizer

        self.rpc = ChannelPair()
