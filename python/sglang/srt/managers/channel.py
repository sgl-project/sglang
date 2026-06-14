"""In-process channels for ThreadedEngine (free-threaded Python / no-GIL).

Replaces ZMQ IPC sockets with thread-safe queues when the scheduler,
detokenizer and tokenizer-manager all run as threads in one process.

Sender/receiver classes are duck-typed against ``zmq.Socket`` /
``zmq.asyncio.Socket`` so that the rest of the codebase keeps calling
``send_pyobj`` / ``recv_pyobj`` unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Optional

import zmq

logger = logging.getLogger(__name__)

_ZMQ_NOBLOCK = zmq.NOBLOCK

# Sentinel pushed through a queue to signal a clean shutdown of any
# receiver bridge thread parked on ``sync_q.get()``.
_SHUTDOWN = object()


class _Sent:
    """Awaitable no-op so ``await sender.send_pyobj(...)`` works.

    The synchronous put has already happened by the time ``send_pyobj``
    returns; the awaitable just lets callers in async code keep their
    ``await`` syntax.
    """

    __slots__ = ()

    def __await__(self):
        return
        yield  # pragma: no cover — makes this a generator-based coroutine


_SENT = _Sent()


class QueueSender:
    """Drop-in for ``zmq.Socket.send_pyobj`` on the sender side.

    The return value is awaitable so both sync and async call sites work:

        sender.send_pyobj(obj)        # fire-and-forget (FanOutCommunicator)
        await sender.send_pyobj(obj)  # TokenizerManager async path
    """

    def __init__(self, q: "queue.SimpleQueue[Any]"):
        self._q = q

    def send_pyobj(self, obj: Any, flags: int = 0, **kwargs) -> _Sent:
        self._q.put(obj)
        return _SENT


class QueueReceiver:
    """Drop-in for ``zmq.Socket.recv_pyobj``.

    Translates ``zmq.NOBLOCK`` to a non-blocking ``get`` and raises
    ``zmq.Again`` (a ``ZMQError`` subclass with ``errno=EAGAIN``) on
    empty queue — matching what real zmq sockets raise, so callers that
    catch either ``zmq.Again`` or the broader ``zmq.ZMQError`` both work.
    """

    def __init__(self, q: "queue.SimpleQueue[Any]"):
        self._q = q

    def recv_pyobj(self, flags: int = 0) -> Any:
        if flags & _ZMQ_NOBLOCK:
            try:
                return self._q.get_nowait()
            except queue.Empty:
                raise zmq.Again(errno=zmq.EAGAIN)
        return self._q.get()


class AsyncQueueReceiver:
    """Async drop-in for ``zmq.asyncio.Socket.recv_pyobj``.

    A background bridge thread drains the sync queue into an
    ``asyncio.Queue`` via ``loop.call_soon_threadsafe``.

    The bridge handles two failure modes that a naive ``while True``
    would not:

    1. The asyncio loop closes while we're parked on ``sync_q.get()``:
       ``call_soon_threadsafe`` raises ``RuntimeError``; we exit cleanly.
    2. ``close()`` is called: we push a sentinel through the sync queue
       to wake the bridge so it can return without leaking the thread.
    """

    def __init__(self, sync_q: "queue.SimpleQueue[Any]"):
        self._sync_q = sync_q
        self._async_q: Optional[asyncio.Queue] = None
        self._bridge_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False

    def _ensure_bridge(self) -> None:
        if self._bridge_thread is not None:
            return
        if self._closed:
            raise RuntimeError("AsyncQueueReceiver is closed")
        self._loop = asyncio.get_running_loop()
        self._async_q = asyncio.Queue()

        loop = self._loop
        async_q = self._async_q
        sync_q = self._sync_q

        def _bridge() -> None:
            while True:
                obj = sync_q.get()
                if obj is _SHUTDOWN:
                    return
                try:
                    loop.call_soon_threadsafe(async_q.put_nowait, obj)
                except RuntimeError:
                    # Event loop already closed — nothing more we can do.
                    return

        t = threading.Thread(target=_bridge, name="channel-bridge", daemon=True)
        t.start()
        self._bridge_thread = t

    async def recv_pyobj(self, flags: int = 0) -> Any:
        self._ensure_bridge()
        return await self._async_q.get()

    def close(self, timeout: float = 1.0) -> None:
        """Stop the bridge thread (idempotent)."""
        if self._closed:
            return
        self._closed = True
        # Push sentinel to unblock the bridge if it is parked on get().
        self._sync_q.put(_SHUTDOWN)
        t = self._bridge_thread
        if t is not None:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(
                    "AsyncQueueReceiver bridge thread did not exit within %.1fs",
                    timeout,
                )


class SyncChannelPair:
    """Unidirectional queue channel with a sync receiver (1 sender, 1 receiver).

    Use when the receiving end is plain synchronous code calling
    ``recv_pyobj()``.
    """

    __slots__ = ("_q", "sender", "receiver")

    def __init__(self) -> None:
        self._q: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self.sender = QueueSender(self._q)
        self.receiver = QueueReceiver(self._q)


class AsyncChannelPair:
    """Unidirectional queue channel with an async receiver (1 sender, 1 receiver).

    Use when the receiving end is inside an ``asyncio`` event loop and
    needs ``await recv_pyobj()``.
    """

    __slots__ = ("_q", "sender", "receiver")

    def __init__(self) -> None:
        self._q: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self.sender = QueueSender(self._q)
        self.receiver = AsyncQueueReceiver(self._q)


class _NullSender:
    """Sender that discards every payload.

    Used by ``QueueDealer`` in threaded mode where the dealer's send
    side has no consumer; queueing every response would leak memory
    unboundedly. Discarding matches the practical "nobody is listening"
    semantics without changing call-site contracts.
    """

    __slots__ = ()

    def send_pyobj(self, obj: Any, flags: int = 0, **kwargs) -> _Sent:
        return _SENT


class QueueDealer:
    """Drop-in for a ``zmq.DEALER`` socket — both ``send_pyobj`` and ``recv_pyobj``.

    The receive side is queue-backed (``request_receiver`` polls it with
    ``NOBLOCK`` and expects ``zmq.Again`` on empty). The send side is a
    sink: in threaded mode no peer drives the dealer, so anything the
    scheduler writes would accumulate forever.
    """

    __slots__ = ("_recv_q", "_recv", "_send")

    def __init__(self, recv_q: "queue.SimpleQueue[Any]"):
        self._recv_q = recv_q
        self._recv = QueueReceiver(recv_q)
        self._send = _NullSender()

    def recv_pyobj(self, flags: int = 0) -> Any:
        return self._recv.recv_pyobj(flags)

    def send_pyobj(self, obj: Any, flags: int = 0, **kwargs) -> _Sent:
        return self._send.send_pyobj(obj, flags, **kwargs)


class ChannelHub:
    """All channels matching the ZMQ topology for single-node tp=1 config.

    Members map directly onto the IPC endpoints the schedulers /
    tokenizer / detokenizer would otherwise create with ZMQ:

      tokenizer_to_scheduler:  TokenizerManager -> Scheduler (PUSH/PULL)
      scheduler_to_detokenizer: Scheduler -> DetokenizerManager (PUSH/PULL)
      detokenizer_to_tokenizer: Detokenizer/Scheduler -> TokenizerManager (PUSH/PULL,
                                consumed inside an async event loop)
      rpc_dealer:               DEALER-shaped channel for Scheduler's recv_from_rpc.
                                ``request_receiver`` unconditionally calls
                                ``recv_from_rpc.recv_pyobj(NOBLOCK)``, so we must
                                provide a real object (not None); in threaded mode
                                no caller drives it, so the recv side stays empty.
    """

    __slots__ = (
        "tokenizer_to_scheduler",
        "scheduler_to_detokenizer",
        "detokenizer_to_tokenizer",
        "_rpc_in",
        "rpc_dealer",
    )

    def __init__(self) -> None:
        self.tokenizer_to_scheduler = SyncChannelPair()
        self.scheduler_to_detokenizer = SyncChannelPair()
        self.detokenizer_to_tokenizer = AsyncChannelPair()
        self._rpc_in: "queue.SimpleQueue[Any]" = queue.SimpleQueue()
        self.rpc_dealer = QueueDealer(self._rpc_in)

    def close(self) -> None:
        """Best-effort shutdown for any background bridge threads."""
        self.detokenizer_to_tokenizer.receiver.close()
