"""WebSocket transport for OpenAI Realtime API-style transcription.

The wire protocol mirrors OpenAI's Realtime API conventions
(``session.start`` / ``transcript.delta`` / ``transcript.final``) so the
``Realtime*`` symbol prefix refers to the protocol identity, not the
transport. A future gRPC streaming variant for the same OpenAI-style
protocol would live in ``serving_transcription_grpc.py`` and could reuse
the same enum/class names without collision.
"""

import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    normalize_whitespace,
    process_asr_chunk,
)

logger = logging.getLogger(__name__)


async def _safe_close_websocket(websocket: WebSocket) -> None:
    """Close a WebSocket, swallowing the "already gone" errors.

    Single source of truth for "which exceptions mean the peer is already
    gone and close() cannot possibly succeed".
    """
    try:
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError):
        pass

# Realtime transcription protocol-fixed audio format: PCM16, 16 kHz, mono, LE.
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # bytes per sample (int16)
_BYTES_PER_SECOND = _SAMPLE_RATE * _SAMPLE_WIDTH


def _pcm_to_wav(pcm_buffer: bytes) -> bytes:
    """Wrap raw PCM16 mono 16 kHz bytes in a WAV container so
    ``soundfile.read`` (called by the multimodal processor) can decode it.
    """
    if not pcm_buffer:
        raise ValueError("pcm_buffer is empty")
    samples = np.frombuffer(pcm_buffer, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()


class RealtimeMessageType(str, Enum):
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_STARTED = "session.started"
    TRANSCRIPT_DELTA = "transcript.delta"
    TRANSCRIPT_FINAL = "transcript.final"
    ERROR = "error"


class RealtimeErrorCode(str, Enum):
    UNSUPPORTED_MODEL = "unsupported_model"
    INVALID_STATE = "invalid_state"
    INVALID_JSON = "invalid_json"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_AUDIO_FORMAT = "invalid_audio_format"
    UNKNOWN_MESSAGE = "unknown_message"
    BUFFER_OVERFLOW = "buffer_overflow"
    INTERNAL_ERROR = "internal_error"


@dataclass(kw_only=True)
class RealtimeTranscriptionSession:
    """OpenAI Realtime API-style session for live transcription over WebSocket."""

    websocket: WebSocket
    state: StreamingASRState
    chunk_size_bytes: int
    max_buffer_bytes: int
    max_buffer_seconds: int
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    pcm_buffer: bytearray = field(default_factory=bytearray)
    last_inference_offset: int = 0
    total_audio_bytes: int = 0
    started: bool = False
    emitted_words: List[str] = field(default_factory=list)
    sampling_params: Optional[dict] = None
    model: Optional[str] = None
    language: Optional[str] = None

    @property
    def has_new_audio(self) -> bool:
        return len(self.pcm_buffer) > self.last_inference_offset

    @property
    def should_trigger_inference(self) -> bool:
        return (
            len(self.pcm_buffer) - self.last_inference_offset >= self.chunk_size_bytes
        )

    def mark_inferred(self) -> None:
        self.last_inference_offset = len(self.pcm_buffer)

    def duration_sec(self) -> float:
        return round(self.total_audio_bytes / _BYTES_PER_SECOND, 2)

    async def accept(self) -> None:
        await self.websocket.accept()

    async def send_json(self, data: dict) -> None:
        await self.websocket.send_text(json.dumps(data))

    async def safe_close(self) -> None:
        await _safe_close_websocket(self.websocket)

    async def send_error(self, code: RealtimeErrorCode, message: str) -> None:
        """Send an OpenAI Realtime-style flat error event."""
        await self.send_json(
            {"type": RealtimeMessageType.ERROR, "code": code, "message": message}
        )


async def handle_realtime_transcription(websocket: WebSocket, serving) -> None:
    """Handle a Realtime transcription session over WebSocket.

    Single-task: receive and inference share one coroutine; PCM queues in OS
    buffers during inference (capped by ``asr_max_buffer_seconds``).
    ``session.end`` is therefore serialized after any in-flight chunk.
    """
    session = await _init_session(websocket, serving)
    if session is None:
        return
    try:
        await _run_session_loop(serving, session)
    except WebSocketDisconnect:
        logger.info(
            "[realtime_transcription] client disconnected: %s", session.session_id
        )
    except Exception:
        logger.exception(
            "[realtime_transcription] unrecoverable error: %s", session.session_id
        )
        try:
            await session.send_error(
                RealtimeErrorCode.INTERNAL_ERROR, "Internal server error"
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        await session.safe_close()


async def _init_session(
    websocket: WebSocket, serving
) -> Optional[RealtimeTranscriptionSession]:
    """Construct and accept the session. Returns None if the adapter rejects."""
    adapter = serving._adapter
    if not adapter.supports_chunked_streaming:
        # Client may disconnect between accept and send; swallow and still close.
        try:
            await websocket.accept()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": RealtimeMessageType.ERROR,
                        "code": RealtimeErrorCode.UNSUPPORTED_MODEL,
                        "message": "Model does not support streaming ASR",
                    }
                )
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
        await _safe_close_websocket(websocket)
        return None

    state = StreamingASRState(**adapter.chunked_streaming_config)
    max_buffer_seconds = serving.tokenizer_manager.server_args.asr_max_buffer_seconds
    session = RealtimeTranscriptionSession(
        websocket=websocket,
        state=state,
        chunk_size_bytes=int(state.chunk_size_sec * _BYTES_PER_SECOND),
        max_buffer_bytes=max_buffer_seconds * _BYTES_PER_SECOND,
        max_buffer_seconds=max_buffer_seconds,
    )
    await session.accept()
    return session


async def _run_session_loop(serving, session: RealtimeTranscriptionSession) -> None:
    """Main receive/dispatch loop. Returns when session should terminate."""
    while True:
        message = await session.websocket.receive()
        if message["type"] == "websocket.disconnect":
            return

        text = message.get("text")
        data = message.get("bytes")
        if text:
            if await _handle_control_message(serving, session, text):
                return
        elif data:
            if await _handle_audio_frame(serving, session, data):
                return


async def _handle_control_message(
    serving, session: RealtimeTranscriptionSession, text: str
) -> bool:
    """Process a JSON control message. Returns True if the session should end."""
    try:
        ctrl = json.loads(text)
    except json.JSONDecodeError:
        await session.send_error(RealtimeErrorCode.INVALID_JSON, "Invalid JSON")
        return False
    if not isinstance(ctrl, dict):
        await session.send_error(
            RealtimeErrorCode.INVALID_PAYLOAD,
            "Control message must be a JSON object",
        )
        return False

    msg_type = ctrl.get("type", "")
    if msg_type == RealtimeMessageType.SESSION_START:
        await _handle_session_start(serving, session, ctrl)
        return False
    if msg_type == RealtimeMessageType.SESSION_END:
        await _handle_session_end(serving, session)
        return True  # session.end always terminates the loop

    await session.send_error(
        RealtimeErrorCode.UNKNOWN_MESSAGE, f"Unknown message type: {msg_type}"
    )
    return False


async def _handle_session_start(
    serving, session: RealtimeTranscriptionSession, ctrl: dict
) -> None:
    if session.started:
        await session.send_error(
            RealtimeErrorCode.INVALID_STATE, "Session already started"
        )
        return

    raw_model = ctrl.get("model")
    raw_language = ctrl.get("language")
    if raw_model is not None and not isinstance(raw_model, str):
        await session.send_error(
            RealtimeErrorCode.INVALID_PAYLOAD,
            "session.start.model must be a string",
        )
        return
    if raw_language is not None and not isinstance(raw_language, str):
        await session.send_error(
            RealtimeErrorCode.INVALID_PAYLOAD,
            "session.start.language must be a string",
        )
        return

    session.model = raw_model
    session.language = raw_language
    adapter = serving._adapter
    session.sampling_params = adapter.build_sampling_params(
        TranscriptionRequest(language=raw_language)
        if raw_language
        else TranscriptionRequest()
    )
    session.started = True
    await session.send_json(
        {
            "type": RealtimeMessageType.SESSION_STARTED,
            "session_id": session.session_id,
            "model": session.model,
        }
    )


async def _handle_session_end(serving, session: RealtimeTranscriptionSession) -> None:
    if not session.started:
        await session.send_error(RealtimeErrorCode.INVALID_STATE, "No active session")
        return

    if session.has_new_audio:
        await _run_inference(serving, session, is_last=True)
    elif session.state.full_transcript:
        # Audio length was an exact multiple of chunk_size_bytes; flush any
        # tokens update() held back without running another inference.
        tail = session.state.finalize()
        await _emit_delta(session, tail)

    await session.send_json(
        {
            "type": RealtimeMessageType.TRANSCRIPT_FINAL,
            # Re-normalize: punctuation can arrive as its own word, leaving
            # an orphan space before it after " ".join().
            "text": normalize_whitespace(" ".join(session.emitted_words)),
            "duration_sec": session.duration_sec(),
            "model": session.model,
        }
    )


async def _handle_audio_frame(
    serving, session: RealtimeTranscriptionSession, data: bytes
) -> bool:
    """Append an audio frame and maybe trigger inference. Returns True on overflow."""
    if not session.started:
        await session.send_error(
            RealtimeErrorCode.INVALID_STATE,
            "Send session.start before streaming audio",
        )
        return False
    if len(data) % _SAMPLE_WIDTH != 0:
        await session.send_error(
            RealtimeErrorCode.INVALID_AUDIO_FORMAT,
            f"PCM16 frame length must be a multiple of {_SAMPLE_WIDTH} bytes",
        )
        return False

    session.pcm_buffer.extend(data)
    session.total_audio_bytes += len(data)

    if len(session.pcm_buffer) > session.max_buffer_bytes:
        await session.send_error(
            RealtimeErrorCode.BUFFER_OVERFLOW,
            f"Accumulated audio exceeded {session.max_buffer_seconds}s; "
            "client is sending faster than inference can keep up",
        )
        return True

    # Cumulative buffer: each inference sees all audio so far,
    # but trigger only once per chunk_size of new audio.
    if session.should_trigger_inference:
        await _run_inference(serving, session, is_last=False)
    return False


async def _run_inference(
    serving, session: RealtimeTranscriptionSession, *, is_last: bool
) -> None:
    wav_data = _pcm_to_wav(bytes(session.pcm_buffer))
    delta = await process_asr_chunk(
        tokenizer_manager=serving.tokenizer_manager,
        adapter=serving._adapter,
        state=session.state,
        audio_data=wav_data,
        sampling_params=session.sampling_params,
        is_last=is_last,
    )
    session.mark_inferred()
    await _emit_delta(session, delta)


async def _emit_delta(session: RealtimeTranscriptionSession, delta: str) -> None:
    if not delta:
        return
    for word in delta.split(" "):
        if not word:
            continue
        session.emitted_words.append(word)
        await session.send_json(
            {
                "type": RealtimeMessageType.TRANSCRIPT_DELTA,
                "delta": word,
            }
        )
