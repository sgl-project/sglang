"""WebSocket transport for OpenAI Realtime API-style transcription.

Protocol:
    Client -> Server:
        {"type": "session.start"}                                              # minimum
        {"type": "session.start", "model": "<model-name>", "language": "en"}   # with hints
        <binary PCM16 frame>     # 16 kHz mono LE, length must be % 2 == 0
        {"type": "session.end"}
    Server -> Client:
        {"type": "session.started", "session_id": "sess_...", "model": "<model-name>"}
        {"type": "transcript.delta", "delta": "hello"}
        {"type": "transcript.final", "text": "hello world", "duration_sec": 1.0,
         "model": "<model-name>"}
        {"type": "error", "code": "invalid_state", "message": "..."}

The ``model`` field is echoed back unchanged; sglang serves a single model
per process, so this field exists for client convenience and does not affect
routing. ``language`` is an ISO 639-1 hint (e.g. ``"en"``, ``"zh"``) and is
adapter-specific — adapters that ignore language hints will not be affected.

This protocol is sglang-specific; it does not align with OpenAI's Realtime
API spec (vLLM's ``/v1/realtime``). Clients written against OpenAI Realtime
will not work as-is. See ``test/manual/models/test_qwen3_asr.py`` for a
reference client and the Pydantic event models below for full schema.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from typing_extensions import Literal

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    normalize_whitespace,
    process_asr_chunk,
)

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_transcription import (
        OpenAIServingTranscription,
    )

logger = logging.getLogger(__name__)


# Realtime transcription protocol-fixed audio format: PCM16, 16 kHz, mono, LE.
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # bytes per sample (int16)
_BYTES_PER_SECOND = _SAMPLE_RATE * _SAMPLE_WIDTH


def _pcm_to_wav(pcm_buffer: bytes) -> bytes:
    """Wrap raw PCM16 mono 16 kHz bytes in a WAV container so
    ``soundfile.read`` (called by the multimodal processor) can decode it.

    Note: callers re-encode the entire cumulative buffer per chunk (M1
    constraint). Cost is bounded by ``--asr-max-buffer-seconds``; M2 plans
    RadixCache prefix caching to remove this overhead.
    """
    if not pcm_buffer:
        raise ValueError("pcm_buffer is empty")
    samples = np.frombuffer(pcm_buffer, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()


# ---- Client -> Server events ----


class SessionStartEvent(BaseModel):
    type: Literal["session.start"] = "session.start"
    model: Optional[str] = None
    language: Optional[str] = None


class SessionEndEvent(BaseModel):
    type: Literal["session.end"] = "session.end"


# ---- Server -> Client events ----


class SessionStartedEvent(BaseModel):
    type: Literal["session.started"] = "session.started"
    session_id: str
    model: Optional[str] = None


class TranscriptDeltaEvent(BaseModel):
    type: Literal["transcript.delta"] = "transcript.delta"
    delta: str


class TranscriptFinalEvent(BaseModel):
    type: Literal["transcript.final"] = "transcript.final"
    text: str
    duration_sec: float
    model: Optional[str] = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


def _format_validation_error(e: ValidationError) -> str:
    """Compact summary of Pydantic validation errors for the error event."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"] if x != "type")
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts) or "Invalid payload"


class RealtimeConnection:
    """Manages a single WebSocket transcription session.

    Single-task: receive and inference share one coroutine; PCM queues in
    OS buffers during inference (capped by ``asr_max_buffer_seconds``).
    ``session.end`` is therefore serialized after any in-flight chunk.
    """

    def __init__(
        self, websocket: WebSocket, serving: OpenAIServingTranscription
    ) -> None:
        self.websocket = websocket
        self.serving = serving
        self.session_id = f"sess_{uuid.uuid4().hex[:12]}"
        # Initialized in ``_init`` once the adapter has accepted the session.
        self.state: Optional[StreamingASRState] = None
        self.chunk_size_bytes = 0
        self.max_buffer_bytes = 0
        self.max_buffer_seconds = 0
        # Per-session mutable state.
        self.pcm_buffer: bytearray = bytearray()
        self.last_inference_offset = 0
        self.total_audio_bytes = 0
        self.started = False
        self.emitted_words: List[str] = []
        self.sampling_params: Optional[dict] = None
        self.model: Optional[str] = None
        self.language: Optional[str] = None

    @property
    def has_new_audio(self) -> bool:
        return len(self.pcm_buffer) > self.last_inference_offset

    @property
    def should_trigger_inference(self) -> bool:
        return (
            len(self.pcm_buffer) - self.last_inference_offset >= self.chunk_size_bytes
        )

    @property
    def duration_sec(self) -> float:
        return round(self.total_audio_bytes / _BYTES_PER_SECOND, 2)

    async def handle(self) -> None:
        """Public entry point. Runs the full session lifecycle."""
        if not await self._init():
            return
        try:
            await self._run_loop()
        except WebSocketDisconnect:
            logger.info(
                f"[realtime_transcription] client disconnected: {self.session_id}"
            )
        except Exception:
            logger.exception(
                f"[realtime_transcription] unrecoverable error: {self.session_id}"
            )
            try:
                await self._send_error("internal_error", "Internal server error")
            except (WebSocketDisconnect, RuntimeError):
                pass
        finally:
            await self._safe_close()

    async def _init(self) -> bool:
        """Accept the WebSocket and initialize per-session state.

        Returns False if the adapter does not support chunked streaming;
        the WS handshake is still accepted so the client receives an
        ``unsupported_model`` error event before close.
        """
        adapter = self.serving._adapter
        if not adapter.supports_chunked_streaming:
            try:
                await self.websocket.accept()
                await self._send_error(
                    "unsupported_model", "Model does not support streaming ASR"
                )
            except (WebSocketDisconnect, RuntimeError):
                pass
            await self._safe_close()
            return False

        self.state = StreamingASRState(**adapter.chunked_streaming_config)
        max_buffer_seconds = (
            self.serving.tokenizer_manager.server_args.asr_max_buffer_seconds
        )
        self.chunk_size_bytes = int(self.state.chunk_size_sec * _BYTES_PER_SECOND)
        self.max_buffer_bytes = max_buffer_seconds * _BYTES_PER_SECOND
        self.max_buffer_seconds = max_buffer_seconds
        await self.websocket.accept()
        return True

    async def _run_loop(self) -> None:
        """Main receive/dispatch loop. Returns when session should terminate."""
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            text = message.get("text")
            data = message.get("bytes")
            if text:
                if await self._on_control(text):
                    return
            elif data:
                if await self._on_audio_frame(data):
                    return

    async def _on_control(self, text: str) -> bool:
        """Dispatch a JSON control message. Returns True if session should end."""
        try:
            ctrl = json.loads(text)
        except json.JSONDecodeError:
            await self._send_error("invalid_json", "Invalid JSON")
            return False
        if not isinstance(ctrl, dict):
            await self._send_error(
                "invalid_payload", "Control message must be a JSON object"
            )
            return False

        msg_type = ctrl.get("type", "")
        if msg_type == "session.start":
            try:
                event = SessionStartEvent.model_validate(ctrl)
            except ValidationError as e:
                await self._send_error("invalid_payload", _format_validation_error(e))
                return False
            await self._on_session_start(event)
            return False
        if msg_type == "session.end":
            await self._on_session_end()
            return True  # session.end always terminates the loop

        await self._send_error("unknown_message", f"Unknown message type: {msg_type}")
        return False

    async def _on_session_start(self, event: SessionStartEvent) -> None:
        if self.started:
            await self._send_error("invalid_state", "Session already started")
            return

        self.model = event.model
        self.language = event.language
        adapter = self.serving._adapter
        self.sampling_params = adapter.build_sampling_params(
            TranscriptionRequest(language=event.language)
            if event.language
            else TranscriptionRequest()
        )
        self.started = True
        await self._send(
            SessionStartedEvent(session_id=self.session_id, model=self.model)
        )

    async def _on_session_end(self) -> None:
        if not self.started:
            await self._send_error("invalid_state", "No active session")
            return

        if self.has_new_audio:
            await self._run_inference(is_last=True)
        elif self.state.full_transcript:
            # Audio length was an exact multiple of chunk_size_bytes; flush any
            # tokens update() held back without running another inference.
            tail = self.state.finalize()
            await self._emit_delta(tail)

        await self._send(
            TranscriptFinalEvent(
                # Re-normalize: punctuation can arrive as its own word, leaving
                # an orphan space before it after " ".join().
                text=normalize_whitespace(" ".join(self.emitted_words)),
                duration_sec=self.duration_sec,
                model=self.model,
            )
        )

    async def _on_audio_frame(self, data: bytes) -> bool:
        """Append an audio frame and maybe trigger inference. Returns True on overflow."""
        if not self.started:
            await self._send_error(
                "invalid_state", "Send session.start before streaming audio"
            )
            return False
        if len(data) % _SAMPLE_WIDTH != 0:
            await self._send_error(
                "invalid_audio_format",
                f"PCM16 frame length must be a multiple of {_SAMPLE_WIDTH} bytes",
            )
            return False

        self.pcm_buffer.extend(data)
        self.total_audio_bytes += len(data)

        if len(self.pcm_buffer) > self.max_buffer_bytes:
            await self._send_error(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.max_buffer_seconds}s; "
                "client is sending faster than inference can keep up",
            )
            return True

        # Cumulative buffer: each inference sees all audio so far,
        # but trigger only once per chunk_size of new audio.
        if self.should_trigger_inference:
            await self._run_inference(is_last=False)
        return False

    async def _run_inference(self, *, is_last: bool) -> None:
        wav_data = _pcm_to_wav(bytes(self.pcm_buffer))
        delta = await process_asr_chunk(
            tokenizer_manager=self.serving.tokenizer_manager,
            adapter=self.serving._adapter,
            state=self.state,
            audio_data=wav_data,
            sampling_params=self.sampling_params,
            is_last=is_last,
        )
        self.last_inference_offset = len(self.pcm_buffer)
        await self._emit_delta(delta)

    async def _emit_delta(self, delta: str) -> None:
        if not delta:
            return
        for word in delta.split(" "):
            if not word:
                continue
            self.emitted_words.append(word)
            await self._send(TranscriptDeltaEvent(delta=word))

    async def _send(self, event: BaseModel) -> None:
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(self, code: str, message: str) -> None:
        await self._send(ErrorEvent(code=code, message=message))

    async def _safe_close(self) -> None:
        """Close the WebSocket, swallowing the "already gone" errors."""
        try:
            await self.websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


async def handle_realtime_transcription(
    websocket: WebSocket, serving: OpenAIServingTranscription
) -> None:
    """Handle a Realtime transcription session over WebSocket."""
    await RealtimeConnection(websocket, serving).handle()
