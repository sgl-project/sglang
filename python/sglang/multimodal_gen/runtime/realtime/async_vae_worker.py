# SPDX-License-Identifier: Apache-2.0

"""Bounded stateful TAEHV worker for realtime MinWM decoding."""

from __future__ import annotations

import asyncio
import inspect
import io
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MinWMWan22VAEArchConfig,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    AcceptDisposition,
    ChunkSequenceTracker,
    LatentChunkHeader,
    ProtocolViolation,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
)


class VAEBackpressureError(RuntimeError):
    pass


class VAESessionCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SessionOpen:
    session_id: str
    generation_id: str
    output_format: str = "raw"
    quality: int = 90
    preview_max_width: int | None = None


@dataclass(frozen=True, slots=True)
class EncodedFrameBatch:
    payloads: tuple[bytes, ...]
    content_type: str
    width: int
    height: int
    frame_batch_index: int
    is_final: bool
    encode_ms: float

    @property
    def num_frames(self) -> int:
        return len(self.payloads)


@dataclass(frozen=True, slots=True)
class DecodeResult:
    disposition: AcceptDisposition
    num_frames: int
    frame_batches: tuple[EncodedFrameBatch, ...]
    queue_wait_ms: float
    decode_ms: float
    encode_ms: float


FrameBatchCallback = Callable[[EncodedFrameBatch], Awaitable[None]]


@dataclass(slots=True)
class _DecodeJob:
    header: LatentChunkHeader
    latents: torch.Tensor
    submitted_at: float
    future: asyncio.Future[DecodeResult]
    on_frame_batch: FrameBatchCallback | None = None


@dataclass(slots=True)
class _WorkerSession:
    opened: SessionOpen
    decoder: Any
    tracker: ChunkSequenceTracker
    queue: asyncio.Queue[_DecodeJob]
    runner: asyncio.Task[None] | None = None
    first_t2v_latent: torch.Tensor | None = None
    last_activity_at: float = field(default_factory=time.monotonic)


class AsyncVAEWorker:
    """Owns shared weights and isolated streaming state for each generation."""

    def __init__(
        self,
        engine: Any,
        *,
        max_sessions: int,
        queue_depth_per_session: int = 1,
        encoded_frames_per_batch: int = 3,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if queue_depth_per_session < 1:
            raise ValueError("queue_depth_per_session must be positive")
        self.engine = engine
        self.max_sessions = max_sessions
        self.queue_depth_per_session = queue_depth_per_session
        self.encoded_frames_per_batch = max(1, encoded_frames_per_batch)
        self._sessions: dict[tuple[str, str], _WorkerSession] = {}
        self._session_lock = asyncio.Lock()
        self._actor_lock = asyncio.Lock()

    async def open(self, request: SessionOpen) -> None:
        if not request.session_id or not request.generation_id:
            raise ProtocolViolation("session generation identity is required")
        identity = (request.session_id, request.generation_id)
        async with self._session_lock:
            if identity in self._sessions:
                return
            if len(self._sessions) >= self.max_sessions:
                raise VAESessionCapacityError(
                    f"VAE session capacity exhausted: {self.max_sessions}"
                )
            decoder = self.engine.create_decoder(identity)
            state = _WorkerSession(
                opened=request,
                decoder=decoder,
                tracker=ChunkSequenceTracker(*identity),
                queue=asyncio.Queue(maxsize=self.queue_depth_per_session),
            )
            self._sessions[identity] = state
            state.runner = asyncio.create_task(
                self._run_session(identity, state),
                name=f"realtime-vae-{request.session_id[:8]}",
            )

    async def submit(
        self,
        header: LatentChunkHeader,
        latents: torch.Tensor,
        *,
        on_frame_batch: FrameBatchCallback | None = None,
    ) -> asyncio.Future[DecodeResult]:
        identity = (header.session_id, header.generation_id)
        state = self._sessions.get(identity)
        if state is None:
            raise ProtocolViolation("unknown VAE session generation")
        if state.queue.full():
            raise VAEBackpressureError("VAE session decode queue is full")

        disposition = state.tracker.accept(header)
        loop = asyncio.get_running_loop()
        if disposition is AcceptDisposition.DUPLICATE:
            future: asyncio.Future[DecodeResult] = loop.create_future()
            future.set_result(
                DecodeResult(
                    disposition=disposition,
                    num_frames=0,
                    frame_batches=(),
                    queue_wait_ms=0.0,
                    decode_ms=0.0,
                    encode_ms=0.0,
                )
            )
            return future

        expected_shape = tuple(int(value) for value in header.shape)
        if tuple(latents.shape) != expected_shape:
            raise ProtocolViolation(
                f"latent shape mismatch: expected {expected_shape}, got {tuple(latents.shape)}"
            )
        if str(latents.dtype).removeprefix("torch.") != header.dtype:
            raise ProtocolViolation(
                f"latent dtype mismatch: expected {header.dtype}, got {latents.dtype}"
            )

        future = loop.create_future()
        job = _DecodeJob(
            header=header,
            latents=latents.detach().contiguous(),
            submitted_at=time.perf_counter(),
            future=future,
            on_frame_batch=on_frame_batch,
        )
        try:
            state.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise VAEBackpressureError("VAE session decode queue is full") from exc
        state.last_activity_at = time.monotonic()
        return future

    async def decode(
        self,
        header: LatentChunkHeader,
        latents: torch.Tensor,
        *,
        on_frame_batch: FrameBatchCallback | None = None,
    ) -> DecodeResult:
        future = await self.submit(
            header,
            latents,
            on_frame_batch=on_frame_batch,
        )
        return await future

    async def _run_session(
        self,
        identity: tuple[str, str],
        state: _WorkerSession,
    ) -> None:
        try:
            while True:
                job = await state.queue.get()
                try:
                    result = await self._decode_job(state, job)
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    raise
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    state.queue.task_done()
                    state.last_activity_at = time.monotonic()
        finally:
            current = self._sessions.get(identity)
            if current is state:
                self._sessions.pop(identity, None)

    async def _decode_job(
        self,
        state: _WorkerSession,
        job: _DecodeJob,
    ) -> DecodeResult:
        queue_wait_ms = (time.perf_counter() - job.submitted_at) * 1000.0
        header = job.header
        source = job.latents
        first_chunk = header.chunk_index == 0
        drop_leading_frames = 0

        if header.chunk_index == 0 and not header.has_reference:
            state.first_t2v_latent = source.detach().clone()
        elif header.chunk_index == 1 and not header.has_reference:
            first_latent = state.first_t2v_latent
            state.first_t2v_latent = None
            if first_latent is not None:
                source = torch.cat([first_latent, source], dim=2).contiguous()
                first_chunk = True
                drop_leading_frames = 1

        decode_started = time.perf_counter()
        async with self._actor_lock:
            frames = self.engine.decode(
                state.decoder,
                source,
                first_chunk=first_chunk,
            )
            if inspect.isawaitable(frames):
                frames = await frames
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        frames = self._normalize_frames(frames)
        if drop_leading_frames:
            frames = frames[:, :, drop_leading_frames:]

        encode_started = time.perf_counter()
        encoded = await asyncio.to_thread(
            self._encode_frames,
            frames,
            state.opened,
        )
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        if job.on_frame_batch is not None:
            for frame_batch in encoded:
                await job.on_frame_batch(frame_batch)

        return DecodeResult(
            disposition=AcceptDisposition.ACCEPT,
            num_frames=sum(batch.num_frames for batch in encoded),
            frame_batches=tuple(encoded),
            queue_wait_ms=queue_wait_ms,
            decode_ms=decode_ms,
            encode_ms=encode_ms,
        )

    @staticmethod
    def _normalize_frames(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
            raise RuntimeError("VAE engine must return a five-dimensional frame tensor")
        if frames.shape[1] not in (1, 3, 4) and frames.shape[2] in (1, 3, 4):
            frames = frames.permute(0, 2, 1, 3, 4)
        if frames.shape[1] not in (1, 3, 4):
            raise RuntimeError("VAE frame tensor must be BCTHW or BTCHW")
        return frames.detach().clamp(0, 1).contiguous().cpu()

    def _encode_frames(
        self,
        frames: torch.Tensor,
        opened: SessionOpen,
    ) -> list[EncodedFrameBatch]:
        if frames.shape[0] != 1:
            raise RuntimeError("Realtime VAE supports one sample per session")
        array = (
            (frames[0, :3] * 255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
            .numpy()
        )
        raw_frames = [frame.tobytes() for frame in array]
        height = int(array.shape[1]) if len(array) else int(frames.shape[-2])
        width = int(array.shape[2]) if len(array) else int(frames.shape[-1])
        output_format = opened.output_format.lower()
        content_type = RAW_RGB_CONTENT_TYPE
        encoded_frames = raw_frames
        encode_started = time.perf_counter()

        if output_format in {"webp", "jpeg"}:
            encoded_frames = [
                self._encode_image(
                    frame,
                    width=width,
                    height=height,
                    output_format=output_format,
                    quality=opened.quality,
                    preview_max_width=opened.preview_max_width,
                )
                for frame in raw_frames
            ]
            content_type = (
                WEBP_FRAME_CONTENT_TYPE
                if output_format == "webp"
                else JPEG_FRAME_CONTENT_TYPE
            )
            if opened.preview_max_width and width > opened.preview_max_width:
                height = max(1, round(height * opened.preview_max_width / width))
                width = opened.preview_max_width

        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        chunks = [
            encoded_frames[index : index + self.encoded_frames_per_batch]
            for index in range(0, len(encoded_frames), self.encoded_frames_per_batch)
        ]
        return [
            EncodedFrameBatch(
                payloads=tuple(payloads),
                content_type=content_type,
                width=width,
                height=height,
                frame_batch_index=index,
                is_final=index == len(chunks) - 1,
                encode_ms=encode_ms if index == len(chunks) - 1 else 0.0,
            )
            for index, payloads in enumerate(chunks)
        ]

    @staticmethod
    def _encode_image(
        frame: bytes,
        *,
        width: int,
        height: int,
        output_format: str,
        quality: int,
        preview_max_width: int | None,
    ) -> bytes:
        image = Image.frombytes("RGB", (width, height), frame)
        if preview_max_width and width > preview_max_width:
            preview_height = max(1, round(height * preview_max_width / width))
            image = image.resize(
                (preview_max_width, preview_height), Image.Resampling.BICUBIC
            )
        buffer = io.BytesIO()
        if output_format == "webp":
            image.save(buffer, format="WEBP", quality=quality, method=0)
        else:
            image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        return buffer.getvalue()

    async def close(self, session_id: str, generation_id: str) -> None:
        identity = (session_id, generation_id)
        async with self._session_lock:
            state = self._sessions.pop(identity, None)
        if state is None:
            return
        if state.runner is not None:
            state.runner.cancel()
        while not state.queue.empty():
            try:
                job = state.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not job.future.done():
                job.future.cancel()
            state.queue.task_done()
        if state.runner is not None:
            await asyncio.gather(state.runner, return_exceptions=True)
        reset = getattr(state.decoder, "reset", None)
        if callable(reset):
            reset()

    async def close_all(self) -> None:
        for session_id, generation_id in list(self._sessions):
            await self.close(session_id, generation_id)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


class TAEHVEngine:
    """Shared immutable TAEHV weights with per-generation decoder objects."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        try:
            from taehv import TAEHV
        except ImportError as exc:
            raise RuntimeError("The taehv package is required by the VAE worker") from exc
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = (
            TAEHV(checkpoint_path=checkpoint_path)
            .eval()
            .to(device=self.device, dtype=dtype)
            .requires_grad_(False)
        )
        config = MinWMWan22VAEArchConfig()
        self.mean = torch.tensor(config.latents_mean, device=self.device, dtype=dtype)[
            None, :, None, None, None
        ]
        self.std = torch.tensor(config.latents_std, device=self.device, dtype=dtype)[
            None, :, None, None, None
        ]

    def create_decoder(self, identity):
        del identity
        from taehv import StreamingTAEHV

        return StreamingTAEHV(self.model).eval()

    @torch.no_grad()
    def decode(self, decoder, latents: torch.Tensor, *, first_chunk: bool):
        if first_chunk:
            decoder.reset()
        source = latents.to(device=self.device, dtype=self.dtype, non_blocking=True)
        source = (source * self.std + self.mean).permute(0, 2, 1, 3, 4).contiguous()
        decoded_frames = []
        frame = decoder.decode(source)
        while frame is not None:
            decoded_frames.append(frame)
            frame = decoder.decode()
        if not decoded_frames:
            return source.new_empty((1, 3, 0, source.shape[-2] * 16, source.shape[-1] * 16))
        frames = torch.cat(decoded_frames, dim=1)
        return frames.permute(0, 2, 1, 3, 4).contiguous().clamp(0, 1)
