# SPDX-License-Identifier: Apache-2.0

"""Session-fenced media routing primitives for the realtime Gateway."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import msgspec.msgpack

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
)


class OutputProtocolError(ProtocolViolation):
    pass


class OutputBackpressureError(RuntimeError):
    pass


class OutputRouteClosed(RuntimeError):
    pass


class AdmissionQueueFull(RuntimeError):
    reason = "ADMISSION_QUEUE_FULL"

    def __init__(self) -> None:
        super().__init__(self.reason)


class BoundedAdmissionWaiterGate:
    def __init__(self, *, max_waiters: int = 64) -> None:
        if max_waiters < 1:
            raise ValueError("max_waiters must be positive")
        self.max_waiters = max_waiters
        self.waiters = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def waiter(self):
        async with self._lock:
            if self.waiters >= self.max_waiters:
                raise AdmissionQueueFull()
            self.waiters += 1
        try:
            yield
        finally:
            async with self._lock:
                self.waiters -= 1


_WORKER_CONTROL_MESSAGES = {
    "error",
    "session_ready",
    "control_ack",
    "heartbeat",
}


def worker_message_allowed(wire: bytes) -> bool:
    """Only forward business control data from Denoiser to the browser."""
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError:
        return False
    return (
        isinstance(message, dict)
        and message.get("type") in _WORKER_CONTROL_MESSAGES
    )


def worker_message_type(wire: bytes) -> str:
    try:
        message = msgspec.msgpack.decode(wire)
    except msgspec.DecodeError as exc:
        raise ProtocolViolation("invalid Denoiser control message") from exc
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolViolation("Denoiser control message type is required")
    return message["type"]


def build_denoiser_url(
    endpoint: str,
    *,
    session_id: str,
    generation_id: str,
    coordinator_token: str,
    vae_url: str,
    output_url: str,
    output_token: str,
    trace_id: str,
    worker_epoch: str = "",
    vae_worker_epoch: str = "",
) -> str:
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(
        gateway_managed="1",
        session_id=session_id,
        generation_id=generation_id,
        coordinator_token=coordinator_token,
        worker_epoch=worker_epoch,
        realtime_vae_worker_url=vae_url,
        realtime_vae_worker_epoch=vae_worker_epoch,
        gateway_output_url=output_url,
        gateway_output_token=output_token,
        trace_id=trace_id,
    )
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


@dataclass(slots=True)
class GatewayOutputRoute:
    session_id: str
    generation_id: str
    token: str
    queue_depth: int
    enqueue_timeout_s: float
    _queue: asyncio.Queue[bytes | None] = field(init=False)
    _last_chunk_index: int = field(default=-1, init=False)
    _last_frame_batch_index: int = field(default=-1, init=False)
    _seen_chunks: set[int] = field(default_factory=set, init=False)
    _output_closed: asyncio.Event = field(init=False)
    _chunk_completed: dict[int, asyncio.Event] = field(
        default_factory=dict, init=False
    )
    dropped_messages: int = field(default=0, init=False)
    bound: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.queue_depth)
        self._output_closed = asyncio.Event()
        self._output_closed.set()

    def bind_output(self) -> None:
        self.bound = True
        self._output_closed.clear()

    def unbind_output(self) -> None:
        self.bound = False
        self._output_closed.set()

    async def wait_until_output_closed(self) -> None:
        await self._output_closed.wait()

    async def wait_until_chunk_completed(self, chunk_index: int) -> None:
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        event = self._chunk_completed.setdefault(chunk_index, asyncio.Event())
        await event.wait()

    def token_matches(self, token: str) -> bool:
        return hmac.compare_digest(self.token, token)

    async def put(self, wire: bytes) -> None:
        if self.closed:
            raise OutputRouteClosed("output route is closed")
        message = decode_message(wire)
        message_type = message.get("type")
        if message_type not in {"frame_batch", "media_chunk_complete"}:
            raise OutputProtocolError(
                "Gateway output accepts frame_batch or media_chunk_complete only"
            )
        if message.get("session_id") != self.session_id:
            raise OutputProtocolError("wrong session")
        if message.get("generation_id") != self.generation_id:
            raise OutputProtocolError("stale generation")
        chunk_index = int(message.get("chunk_index", -1))
        if chunk_index < 0:
            raise OutputProtocolError("invalid chunk sequence")
        if message_type == "media_chunk_complete":
            if chunk_index not in self._seen_chunks:
                raise OutputProtocolError("completion before frame batch")
            completed = self._chunk_completed.setdefault(
                chunk_index, asyncio.Event()
            )
            if completed.is_set():
                raise OutputProtocolError("duplicate completion")
            await self._put_with_bounded_drop(wire)
            completed.set()
            return

        frame_batch_index = int(message.get("frame_batch_index", -1))
        if frame_batch_index < 0:
            raise OutputProtocolError("invalid frame sequence")
        if chunk_index < self._last_chunk_index:
            raise OutputProtocolError("stale chunk")
        if chunk_index > self._last_chunk_index + 1:
            raise OutputProtocolError("out-of-order chunk")
        if chunk_index == self._last_chunk_index:
            if frame_batch_index <= self._last_frame_batch_index:
                raise OutputProtocolError("duplicate frame batch")
        elif frame_batch_index != 0:
            raise OutputProtocolError("new chunk must start at frame batch zero")
        await self._put_with_bounded_drop(wire)
        self._last_chunk_index = chunk_index
        self._last_frame_batch_index = frame_batch_index
        self._seen_chunks.add(chunk_index)
        if message.get("is_final_frame_batch") is True:
            self._chunk_completed.setdefault(
                chunk_index, asyncio.Event()
            ).set()

    async def _put_with_bounded_drop(self, wire: bytes) -> None:
        try:
            await asyncio.wait_for(
                self._queue.put(wire), timeout=self.enqueue_timeout_s
            )
            return
        except TimeoutError:
            pass

        # A slow browser/network path should not tear down the model session.
        # Keep Gateway memory bounded by discarding the oldest unsent message and
        # enqueueing the newest data after the timeout budget is exhausted.
        while self._queue.full():
            self._drop_oldest_queued_message()
        try:
            self._queue.put_nowait(wire)
        except asyncio.QueueFull as exc:
            raise OutputBackpressureError(
                "Gateway output queue remained full"
            ) from exc

    def _drop_oldest_queued_message(self) -> None:
        try:
            dropped = self._queue.get_nowait()
        except asyncio.QueueEmpty as exc:
            raise OutputBackpressureError(
                "Gateway output queue remained full"
            ) from exc
        if dropped is None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            raise OutputRouteClosed("output route is closed")
        self._queue.task_done()
        self.dropped_messages += 1

    async def get(self) -> bytes:
        wire = await self._queue.get()
        if wire is None:
            raise OutputRouteClosed("output route is closed")
        return wire

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.unbind_output()
        for event in self._chunk_completed.values():
            event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


class GatewayOutputRegistry:
    def __init__(
        self, *, queue_depth: int = 2, enqueue_timeout_s: float = 1.0
    ) -> None:
        if queue_depth < 1:
            raise ValueError("queue_depth must be positive")
        if enqueue_timeout_s <= 0:
            raise ValueError("enqueue_timeout_s must be positive")
        self.queue_depth = queue_depth
        self.enqueue_timeout_s = enqueue_timeout_s
        self._routes: dict[str, GatewayOutputRoute] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> GatewayOutputRoute:
        if not session_id or not generation_id or not token:
            raise OutputProtocolError("output route identity is required")
        async with self._lock:
            current = self._routes.get(session_id)
            if current is not None and not current.closed:
                raise OutputProtocolError("session output route is already registered")
            route = GatewayOutputRoute(
                session_id=session_id,
                generation_id=generation_id,
                token=token,
                queue_depth=self.queue_depth,
                enqueue_timeout_s=self.enqueue_timeout_s,
            )
            self._routes[session_id] = route
            return route

    async def bind(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> GatewayOutputRoute:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None or route.closed:
                raise OutputProtocolError("unknown output route")
            if route.generation_id != generation_id:
                raise OutputProtocolError("stale generation")
            if not route.token_matches(token):
                raise OutputProtocolError("invalid output token")
            if route.bound:
                raise OutputProtocolError("output route is already bound")
            route.bind_output()
            return route

    async def unbind(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> None:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None:
                return
            if route.generation_id != generation_id or not route.token_matches(token):
                return
            route.unbind_output()

    async def unregister(
        self,
        session_id: str,
        generation_id: str,
        *,
        token: str,
    ) -> None:
        async with self._lock:
            route = self._routes.get(session_id)
            if route is None:
                return
            if route.generation_id != generation_id or not route.token_matches(token):
                return
            self._routes.pop(session_id, None)
        await route.close()
