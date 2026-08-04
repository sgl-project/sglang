# SPDX-License-Identifier: Apache-2.0

"""Persistent Gateway client for the bounded realtime VAE worker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import torch
from websockets.asyncio.client import connect

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    LatentChunkHeader,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    encode_message,
)


class RemoteVAEError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RemoteFrameBatch:
    session_id: str
    generation_id: str
    request_id: str
    chunk_index: int
    event_id: int | None
    payloads: tuple[bytes, ...]
    content_type: str
    width: int
    height: int
    frame_batch_index: int
    is_final: bool
    encode_ms: float


@dataclass(frozen=True, slots=True)
class RemoteDecodeResult:
    request_id: str
    chunk_index: int
    num_frames: int
    queue_wait_ms: float
    decode_ms: float
    encode_ms: float
    transfer_ms: float
    serialize_ms: float
    latent_send_ms: float
    credit_wait_ms: float
    first_frame_ms: float | None
    completed_at: float


FrameBatchHandler = Callable[[RemoteFrameBatch], Awaitable[None]]


class GatewayOutputClient:
    """Session-bound media connection from a VAE worker to one Gateway Pod."""

    def __init__(
        self,
        url: str,
        *,
        session_id: str,
        generation_id: str,
        token: str,
        timeout_s: float = 5.0,
        max_message_bytes: int = 64 * 1024 * 1024,
        connect_factory=connect,
    ) -> None:
        if not url or not session_id or not generation_id or not token:
            raise ValueError("Gateway output identity is required")
        self.url = url
        self.session_id = session_id
        self.generation_id = generation_id
        self.token = token
        self.timeout_s = timeout_s
        self.max_message_bytes = max_message_bytes
        self._connect_factory = connect_factory
        self._ws = None
        self._send_lock = asyncio.Lock()

    async def open(self) -> None:
        if self._ws is not None:
            return
        self._ws = await self._connect_factory(
            self.url,
            max_size=self.max_message_bytes,
            compression=None,
            open_timeout=self.timeout_s,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
        )
        await self._ws.send(
            encode_message(
                "session_output_open",
                session_id=self.session_id,
                generation_id=self.generation_id,
                token=self.token,
            )
        )
        response = decode_message(
            await asyncio.wait_for(self._ws.recv(), self.timeout_s),
            max_message_bytes=self.max_message_bytes,
        )
        if response.get("type") != "session_output_accepted":
            raise RemoteVAEError(
                f"Gateway rejected output route: {response.get('message', response)}"
            )
        if (
            response.get("session_id") != self.session_id
            or response.get("generation_id") != self.generation_id
        ):
            raise ProtocolViolation("Gateway output acceptance identity mismatch")

    async def send(self, wire: bytes) -> None:
        if self._ws is None:
            raise RemoteVAEError("Gateway output client is not open")
        message = decode_message(wire, max_message_bytes=self.max_message_bytes)
        if message.get("type") != "frame_batch":
            raise ProtocolViolation("Gateway output accepts frame_batch only")
        if (
            message.get("session_id") != self.session_id
            or message.get("generation_id") != self.generation_id
        ):
            raise ProtocolViolation("Gateway output frame identity mismatch")
        async with self._send_lock:
            await self._ws.send(wire)

    async def close(self) -> None:
        if self._ws is None:
            return
        await self._ws.close()
        self._ws = None


@dataclass(slots=True)
class _PendingDecode:
    header: LatentChunkHeader
    accepted: asyncio.Event
    result: asyncio.Future[RemoteDecodeResult]
    on_frame_batch: FrameBatchHandler
    sent_at: float
    serialize_ms: float
    latent_send_ms: float = 0.0
    credit_wait_ms: float = 0.0
    first_frame_ms: float | None = None
    callback_tail: asyncio.Task | None = None


@dataclass(frozen=True, slots=True)
class RemoteDecodeHandle:
    _future: asyncio.Future[RemoteDecodeResult]
    timeout_s: float
    serialize_ms: float
    latent_send_ms: float
    credit_wait_ms: float

    async def wait(self) -> RemoteDecodeResult:
        return await asyncio.wait_for(asyncio.shield(self._future), self.timeout_s)

    def cancel(self) -> None:
        self._future.cancel()


class RealtimeVAEClient:
    def __init__(
        self,
        url: str,
        *,
        session_id: str,
        generation_id: str,
        timeout_s: float = 10.0,
        max_message_bytes: int = 64 * 1024 * 1024,
        connect_factory=connect,
    ) -> None:
        self.url = url
        self.session_id = session_id
        self.generation_id = generation_id
        self.timeout_s = timeout_s
        self.max_message_bytes = max_message_bytes
        self._connect_factory = connect_factory
        self._ws = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, _PendingDecode] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._callback_tail: asyncio.Task | None = None
        self._closed = False

    async def open(
        self,
        *,
        output_format: str,
        quality: int,
        preview_max_width: int | None,
        output_url: str | None = None,
        output_token: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        if self._ws is not None:
            return
        self._ws = await self._connect_factory(
            self.url,
            max_size=self.max_message_bytes,
            compression=None,
            open_timeout=self.timeout_s,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
        )
        await self._ws.send(
            encode_message(
                "session_open",
                session_id=self.session_id,
                generation_id=self.generation_id,
                output_format=output_format,
                quality=quality,
                preview_max_width=preview_max_width,
                output_url=output_url,
                output_token=output_token,
                trace_id=trace_id,
            )
        )
        response = decode_message(
            await asyncio.wait_for(self._ws.recv(), self.timeout_s),
            max_message_bytes=self.max_message_bytes,
        )
        if response.get("type") != "session_accepted":
            raise RemoteVAEError(
                f"VAE worker rejected session: {response.get('message', response)}"
            )
        if (
            response.get("session_id") != self.session_id
            or response.get("generation_id") != self.generation_id
        ):
            raise ProtocolViolation("VAE session acceptance identity mismatch")
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"vae-reader-{self.session_id[:8]}"
        )

    async def submit(
        self,
        latents: torch.Tensor,
        handoff: dict,
        *,
        on_frame_batch: FrameBatchHandler,
    ) -> RemoteDecodeHandle:
        if self._ws is None or self._closed:
            raise RemoteVAEError("VAE client is not open")
        serialize_started = time.perf_counter()
        cpu_latents = latents.detach().to(device="cpu").contiguous()
        if cpu_latents.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ProtocolViolation(f"unsupported latent dtype: {cpu_latents.dtype}")
        payload = cpu_latents.view(torch.uint8).numpy().tobytes()
        serialize_ms = (time.perf_counter() - serialize_started) * 1000.0
        header = LatentChunkHeader(
            session_id=str(handoff["session_id"]),
            generation_id=str(handoff["generation_id"]),
            request_id=str(handoff["request_id"]),
            chunk_index=int(handoff["chunk_index"]),
            dtype=str(cpu_latents.dtype).removeprefix("torch."),
            shape=tuple(int(value) for value in cpu_latents.shape),
            byte_length=len(payload),
            checksum=checksum_payload(payload),
            event_id=handoff.get("event_id"),
            action_version=int(handoff.get("action_version") or 0),
            prompt_version=int(handoff.get("prompt_version") or 0),
            deadline_epoch_ms=int(time.time() * 1000 + self.timeout_s * 1000),
            has_reference=bool(handoff.get("has_reference")),
        )
        if header.session_id != self.session_id or header.generation_id != self.generation_id:
            raise ProtocolViolation("VAE latent handoff identity mismatch")
        if header.request_id in self._pending:
            raise ProtocolViolation("duplicate in-flight VAE request ID")

        loop = asyncio.get_running_loop()
        pending = _PendingDecode(
            header=header,
            accepted=asyncio.Event(),
            result=loop.create_future(),
            on_frame_batch=on_frame_batch,
            sent_at=time.perf_counter(),
            serialize_ms=serialize_ms,
        )
        self._pending[header.request_id] = pending
        try:
            wire = encode_message("latent_chunk", header=header, payload=payload)
            if len(wire) > self.max_message_bytes:
                raise ProtocolViolation("encoded latent exceeds VAE message limit")
            send_started = time.perf_counter()
            async with self._send_lock:
                await self._ws.send(wire)
            pending.latent_send_ms = (time.perf_counter() - send_started) * 1000.0
            credit_started = time.perf_counter()
            await asyncio.wait_for(pending.accepted.wait(), self.timeout_s)
            pending.credit_wait_ms = (
                time.perf_counter() - credit_started
            ) * 1000.0
            if pending.result.done():
                pending.result.result()
        except Exception:
            self._pending.pop(header.request_id, None)
            if not pending.result.done():
                pending.result.cancel()
            raise
        return RemoteDecodeHandle(
            pending.result,
            self.timeout_s,
            pending.serialize_ms,
            pending.latent_send_ms,
            pending.credit_wait_ms,
        )

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                message = decode_message(
                    raw,
                    max_message_bytes=self.max_message_bytes,
                )
                message_type = message["type"]
                request_id = message.get("request_id")
                pending = self._pending.get(request_id) if request_id else None
                if message_type == "latent_accepted":
                    if pending is None:
                        raise ProtocolViolation("acceptance for unknown VAE request")
                    self._validate_message_identity(message, pending.header)
                    pending.accepted.set()
                elif message_type == "frame_batch":
                    if pending is None:
                        raise ProtocolViolation("frames for unknown VAE request")
                    self._validate_message_identity(message, pending.header)
                    if pending.first_frame_ms is None:
                        pending.first_frame_ms = (
                            time.perf_counter() - pending.sent_at
                        ) * 1000.0
                    frame_batch = self._decode_frame_batch(message, pending.header)
                    previous = self._callback_tail

                    async def dispatch(
                        previous=previous,
                        frame_batch=frame_batch,
                        pending=pending,
                    ):
                        if previous is not None:
                            await previous
                        await pending.on_frame_batch(frame_batch)

                    pending.callback_tail = asyncio.create_task(dispatch())
                    self._callback_tail = pending.callback_tail
                elif message_type == "chunk_complete":
                    if pending is None:
                        raise ProtocolViolation("completion for unknown VAE request")
                    self._validate_message_identity(message, pending.header)
                    task = asyncio.create_task(self._finish_pending(pending, message))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                elif message_type == "error":
                    error = RemoteVAEError(
                        f"{message.get('error_type', 'RemoteVAEError')}: "
                        f"{message.get('message', 'remote VAE failure')}"
                    )
                    if pending is None:
                        raise error
                    self._fail_pending(pending, error)
                else:
                    raise ProtocolViolation(f"unknown VAE response: {message_type}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for pending in list(self._pending.values()):
                self._fail_pending(pending, exc)

    @staticmethod
    def _validate_message_identity(message: dict, header: LatentChunkHeader) -> None:
        if message.get("session_id") != header.session_id:
            raise ProtocolViolation("VAE response session mismatch")
        if message.get("generation_id") != header.generation_id:
            raise ProtocolViolation("VAE response generation mismatch")
        if int(message.get("chunk_index", -1)) != header.chunk_index:
            raise ProtocolViolation("VAE response chunk mismatch")

    @staticmethod
    def _decode_frame_batch(
        message: dict,
        header: LatentChunkHeader,
    ) -> RemoteFrameBatch:
        payload = message.get("payload")
        lengths = message.get("payload_lengths")
        if not isinstance(payload, bytes) or not isinstance(lengths, list):
            raise ProtocolViolation("invalid remote frame payload")
        payloads = []
        offset = 0
        for raw_length in lengths:
            length = int(raw_length)
            if length < 0 or offset + length > len(payload):
                raise ProtocolViolation("invalid remote frame payload lengths")
            payloads.append(payload[offset : offset + length])
            offset += length
        if offset != len(payload):
            raise ProtocolViolation("remote frame payload length mismatch")
        return RemoteFrameBatch(
            session_id=header.session_id,
            generation_id=header.generation_id,
            request_id=header.request_id,
            chunk_index=header.chunk_index,
            event_id=message.get("event_id"),
            payloads=tuple(payloads),
            content_type=str(message["content_type"]),
            width=int(message["width"]),
            height=int(message["height"]),
            frame_batch_index=int(message.get("frame_batch_index") or 0),
            is_final=bool(message.get("is_final_frame_batch")),
            encode_ms=float(message.get("encode_ms") or 0.0),
        )

    async def _finish_pending(self, pending: _PendingDecode, message: dict) -> None:
        completed_at = time.perf_counter()
        try:
            if pending.callback_tail is not None:
                await pending.callback_tail
        except asyncio.CancelledError:
            self._pending.pop(pending.header.request_id, None)
            if not pending.result.done():
                pending.result.cancel()
            raise
        except Exception as exc:
            self._fail_pending(pending, exc)
            return
        self._pending.pop(pending.header.request_id, None)
        if pending.result.done():
            return
        pending.result.set_result(
            RemoteDecodeResult(
                request_id=pending.header.request_id,
                chunk_index=pending.header.chunk_index,
                num_frames=int(message.get("num_frames") or 0),
                queue_wait_ms=float(message.get("queue_wait_ms") or 0.0),
                decode_ms=float(message.get("decode_ms") or 0.0),
                encode_ms=float(message.get("encode_ms") or 0.0),
                transfer_ms=(completed_at - pending.sent_at) * 1000.0,
                serialize_ms=pending.serialize_ms,
                latent_send_ms=pending.latent_send_ms,
                credit_wait_ms=pending.credit_wait_ms,
                first_frame_ms=pending.first_frame_ms,
                completed_at=completed_at,
            )
        )

    def _fail_pending(self, pending: _PendingDecode, exc: Exception) -> None:
        self._pending.pop(pending.header.request_id, None)
        pending.accepted.set()
        if pending.callback_tail is not None:
            pending.callback_tail.cancel()
        if not pending.result.done():
            pending.result.set_exception(exc)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            try:
                async with self._send_lock:
                    await self._ws.send(
                        encode_message(
                            "abort",
                            session_id=self.session_id,
                            generation_id=self.generation_id,
                        )
                    )
            except Exception:
                pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        for pending in list(self._pending.values()):
            self._fail_pending(pending, RemoteVAEError("VAE client closed"))
        if self._ws is not None:
            await self._ws.close()
        self._ws = None
        self._callback_tail = None
