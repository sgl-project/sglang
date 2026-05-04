"""WebSocket transport for OpenAI Realtime transcription mode.

Endpoint: ``WS /v1/realtime``
https://platform.openai.com/docs/guides/realtime-transcription

Notable deviations: ``audio.input.sample_rate`` accepts 16/24/48 kHz with
internal resample to 16 kHz; ``turn_detection`` and ``noise_reduction``
must be ``null`` (no server-side VAD); ``include[]`` is dropped; the
model field is echo-only.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    normalize_whitespace,
    process_asr_chunk,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


_MODEL_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2
# pcm_buffer is always at the model rate (incoming frames are resampled
# in _on_audio_append) so all buffer math is rate-independent.
_BYTES_PER_SECOND = _MODEL_SAMPLE_RATE * _SAMPLE_WIDTH
_DEFAULT_INPUT_SAMPLE_RATE = 24000  # OpenAI default for audio/pcm
_SUPPORTED_INPUT_SAMPLE_RATES = (16000, 24000, 48000)


def _resample_to_model_rate(pcm: bytes, src_rate: int) -> bytes:
    # assumes int16 LE per audio/pcm spec. Normalize by 2^15 (so int16
    # maps to [-1, 1]); re-encode by 2^15 - 1 so a clipped 1.0 stays in
    # int16 range.
    if src_rate == _MODEL_SAMPLE_RATE or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    samples = librosa.resample(samples, orig_sr=src_rate, target_sr=_MODEL_SAMPLE_RATE)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _pcm_to_wav(pcm: bytes) -> bytes:
    if not pcm:
        raise ValueError("pcm is empty")
    samples = np.frombuffer(pcm, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, _MODEL_SAMPLE_RATE, format="WAV")
    return buf.getvalue()


class RealtimeConnection:
    """One realtime transcription session.

    Single-task: a single coroutine alternates receive + inference. Frames
    that arrive during inference queue at the WS transport (asyncio + TCP)
    and only land in pcm_buffer after inference returns;
    --asr-max-buffer-seconds bounds the catch-up.
    """

    def __init__(
        self,
        websocket: WebSocket,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        server_args: ServerArgs,
    ) -> None:
        self.websocket = websocket
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.server_args = server_args
        self.session_id = f"sess_{uuid.uuid4().hex[:24]}"

        self.input_sample_rate = _DEFAULT_INPUT_SAMPLE_RATE
        self.language: Optional[str] = None
        self.client_model: Optional[str] = None
        self.sampling_params: Optional[Dict[str, Any]] = None
        self.session_configured = False
        self._current_client_event_id: Optional[str] = None

        self.state: Optional[StreamingASRState] = None
        self.pcm_buffer = bytearray()
        self.last_inference_offset = 0
        self.emitted_words: List[str] = []
        self.current_item_id = f"item_{uuid.uuid4().hex[:24]}"
        self.previous_item_id: Optional[str] = None

        self.max_buffer_seconds = server_args.asr_max_buffer_seconds
        self.max_buffer_bytes = self.max_buffer_seconds * _BYTES_PER_SECOND

    async def handle(self) -> None:
        if not self.adapter.supports_chunked_streaming:
            await self._send_error(
                "not_supported", "Model does not support streaming ASR"
            )
            return

        await self._send_session("session.created", configured=False)

        try:
            await self._run_loop()
        except WebSocketDisconnect:
            logger.info("[realtime] client disconnected: %s", self.session_id)
        except Exception:
            logger.exception("[realtime] unrecoverable error: %s", self.session_id)
            try:
                await self._send_error(
                    "server_error",
                    "Internal server error",
                    error_type="server_error",
                )
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def _run_loop(self) -> None:
        while True:
            self._current_client_event_id = None
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            text = message.get("text")
            if not text:
                if message.get("bytes") is not None:
                    await self._send_error(
                        "invalid_payload",
                        "Binary frames are not supported on /v1/realtime; "
                        "use input_audio_buffer.append with base64 audio.",
                    )
                continue

            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                await self._send_error("invalid_payload", "Invalid JSON")
                continue
            if not isinstance(event, dict):
                await self._send_error(
                    "invalid_payload", "Top-level event must be a JSON object"
                )
                continue

            self._current_client_event_id = event.get("event_id")
            evt_type = event.get("type", "")
            terminate = await self._dispatch(evt_type, event)
            if terminate:
                return

    async def _dispatch(self, evt_type: str, raw: Dict[str, Any]) -> bool:
        """Route one client event. Returns True iff the session should terminate."""
        if evt_type == "session.update":
            await self._on_session_update(raw)
        elif evt_type == "input_audio_buffer.append":
            return await self._on_audio_append(raw)
        elif evt_type == "input_audio_buffer.commit":
            await self._on_audio_commit()
        elif evt_type == "input_audio_buffer.clear":
            await self._on_audio_clear()
        else:
            await self._send_error("unknown_event", f"Unknown event type: {evt_type!r}")
        return False

    async def _on_session_update(self, raw: Dict[str, Any]) -> None:
        cfg = raw.get("session") or {}
        if not isinstance(cfg, dict):
            await self._send_error(
                "invalid_value",
                "session must be a JSON object",
                param="session",
            )
            return

        sess_type = cfg.get("type")
        if sess_type is not None and sess_type != "transcription":
            await self._send_error(
                "invalid_value",
                f"session.type must be 'transcription', got {sess_type!r}",
                param="session.type",
            )
            return

        audio = (cfg.get("audio") or {}).get("input") or {}

        fmt = audio.get("format")
        if fmt and fmt != "audio/pcm":
            await self._send_error(
                "invalid_value",
                f"audio.input.format must be 'audio/pcm', got {fmt!r}",
                param="session.audio.input.format",
            )
            return

        sample_rate = audio.get("sample_rate")
        if sample_rate is not None:
            if sample_rate not in _SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    "invalid_value",
                    f"audio.input.sample_rate must be one of "
                    f"{_SUPPORTED_INPUT_SAMPLE_RATES}, got {sample_rate}",
                    param="session.audio.input.sample_rate",
                )
                return
            self.input_sample_rate = sample_rate
        else:
            self.input_sample_rate = _DEFAULT_INPUT_SAMPLE_RATE

        if audio.get("turn_detection") is not None:
            await self._send_error(
                "not_supported",
                "Server-side VAD is not implemented; "
                "set audio.input.turn_detection: null and commit explicitly.",
                param="session.audio.input.turn_detection",
            )
            return

        if audio.get("noise_reduction") is not None:
            await self._send_error(
                "not_supported",
                "audio.input.noise_reduction is not supported; set to null.",
                param="session.audio.input.noise_reduction",
            )
            return

        transcription = audio.get("transcription") or {}
        if transcription.get("prompt") is not None:
            await self._send_error(
                "not_supported",
                "audio.input.transcription.prompt is not supported.",
                param="session.audio.input.transcription.prompt",
            )
            return

        self.client_model = transcription.get("model")
        self.language = transcription.get("language")

        if self.state is None:
            self.state = StreamingASRState(**self.adapter.chunked_streaming_config)

        self.sampling_params = self.adapter.build_sampling_params(
            TranscriptionRequest(language=self.language)
        )

        self.session_configured = True
        if self.input_sample_rate != _MODEL_SAMPLE_RATE:
            logger.info(
                "[realtime] %s configured: resample %d→%d (ratio %.2f), language=%s",
                self.session_id,
                self.input_sample_rate,
                _MODEL_SAMPLE_RATE,
                self.input_sample_rate / _MODEL_SAMPLE_RATE,
                self.language,
            )
        await self._send_session("session.updated", configured=True)

    async def _on_audio_append(self, raw: Dict[str, Any]) -> bool:
        if not self.session_configured:
            await self._send_error(
                "invalid_state", "Send session.update before audio frames"
            )
            return False

        audio = raw.get("audio")
        if not isinstance(audio, str):
            await self._send_error(
                "invalid_value",
                "audio field must be a base64-encoded string",
                param="audio",
            )
            return False
        try:
            data = base64.b64decode(audio, validate=True)
        except (ValueError, TypeError):
            await self._send_error(
                "invalid_audio", "audio field is not valid base64", param="audio"
            )
            return False

        if len(data) % _SAMPLE_WIDTH != 0:
            await self._send_error(
                "invalid_audio_format",
                f"PCM16 frame length must be a multiple of {_SAMPLE_WIDTH} bytes",
            )
            return False

        # Resample on append so pcm_buffer stays at 16 kHz; otherwise
        # _run_inference would re-resample the cumulative buffer every
        # call (O(N²)). Offload to a thread because librosa.resample with
        # kaiser_best is synchronous CPU work and 32 concurrent sessions
        # would otherwise serialize on the event loop.
        if self.input_sample_rate != _MODEL_SAMPLE_RATE:
            data = await asyncio.to_thread(
                _resample_to_model_rate, data, self.input_sample_rate
            )
        self.pcm_buffer.extend(data)

        if len(self.pcm_buffer) > self.max_buffer_bytes:
            await self._send_error(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.max_buffer_seconds}s; "
                f"client is sending faster than inference can keep up",
                error_type="rate_limit_exceeded",
            )
            return True

        chunk_size = (
            int(self.state.chunk_size_sec * _BYTES_PER_SECOND) if self.state else 0
        )
        if (
            chunk_size > 0
            and len(self.pcm_buffer) - self.last_inference_offset >= chunk_size
        ):
            await self._run_inference(is_last=False)
        return False

    async def _on_audio_commit(self) -> None:
        if not self.session_configured:
            await self._send_error("invalid_state", "Send session.update before commit")
            return
        if not self.pcm_buffer:
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return

        has_new_audio = len(self.pcm_buffer) > self.last_inference_offset
        if has_new_audio:
            # Final inference failed and emitted transcription.failed +
            # rolled the item; suppress committed/created/completed
            # because they would contradict the failure event per spec.
            if not await self._run_inference(is_last=True):
                return
        elif self.state and self.state.full_transcript:
            # Audio length aligned exactly with chunk_size_bytes — no new
            # inference, but flush the unfixed_token_num tail update()
            # held back.
            tail = self.state.finalize()
            await self._emit_delta(tail)

        transcript = normalize_whitespace(" ".join(self.emitted_words))
        item_id = self.current_item_id
        prev_item_id = self.previous_item_id

        await self._send(
            {
                "type": "input_audio_buffer.committed",
                "item_id": item_id,
                "previous_item_id": prev_item_id,
            }
        )
        await self._send(
            {
                "type": "conversation.item.created",
                "previous_item_id": prev_item_id,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_audio", "transcript": transcript}],
                },
            }
        )
        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "content_index": 0,
                "transcript": transcript,
            }
        )

        self._roll_item()

    async def _on_audio_clear(self) -> None:
        self._reset_buffer_state()
        await self._send({"type": "input_audio_buffer.cleared"})

    def _reset_buffer_state(self) -> None:
        self.pcm_buffer = bytearray()
        self.last_inference_offset = 0
        self.emitted_words = []
        self.state = StreamingASRState(**self.adapter.chunked_streaming_config)

    def _roll_item(self) -> None:
        self.previous_item_id = self.current_item_id
        self.current_item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._reset_buffer_state()

    async def _run_inference(self, *, is_last: bool) -> bool:
        """Run one inference; emit deltas; return False on failure.

        ``is_last`` is purely a state-machine signal for process_asr_chunk
        (toggles state.finalize() vs state.update()); the model invocation
        is identical either way.

        On failure, emit transcription.failed and roll the item over —
        per OpenAI spec ``failed`` is item-terminal, so subsequent audio
        must belong to a fresh item.
        """
        wav_data = _pcm_to_wav(bytes(self.pcm_buffer))
        try:
            delta = await process_asr_chunk(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                state=self.state,
                audio_data=wav_data,
                sampling_params=self.sampling_params,
                is_last=is_last,
            )
        except Exception as e:
            logger.exception("[realtime] inference failed: %s", self.session_id)
            await self._send(
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "item_id": self.current_item_id,
                    "content_index": 0,
                    "error": {
                        "type": "server_error",
                        "code": "inference_failed",
                        "message": str(e),
                    },
                }
            )
            self._roll_item()
            return False
        self.last_inference_offset = len(self.pcm_buffer)
        await self._emit_delta(delta)
        return True

    async def _emit_delta(self, delta: str) -> None:
        if not delta:
            return
        for word in delta.split(" "):
            if not word:
                continue
            self.emitted_words.append(word)
            await self._send(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": self.current_item_id,
                    "content_index": 0,
                    "delta": word,
                }
            )

    async def _send_session(self, event_type: str, *, configured: bool) -> None:
        transcription = (
            {"model": self.client_model, "language": self.language}
            if configured
            else {"model": None, "language": None}
        )
        await self._send(
            {
                "type": event_type,
                "session": {
                    "id": self.session_id,
                    "object": "realtime.transcription_session",
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": "audio/pcm",
                            "sample_rate": self.input_sample_rate,
                            "transcription": transcription,
                            "noise_reduction": None,
                            "turn_detection": None,
                        },
                    },
                },
            }
        )

    async def _send(self, event: Dict[str, Any]) -> None:
        event.setdefault("event_id", f"event_{uuid.uuid4().hex[:24]}")
        await self.websocket.send_text(json.dumps(event, ensure_ascii=False))

    async def _send_error(
        self,
        code: str,
        message: str,
        param: Optional[str] = None,
        error_type: str = "invalid_request_error",
    ) -> None:
        await self._send(
            {
                "type": "error",
                "error": {
                    "type": error_type,
                    "code": code,
                    "message": message,
                    "param": param,
                    "event_id": self._current_client_event_id,
                },
            }
        )


async def handle_realtime_transcription(
    websocket: WebSocket,
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    server_args: ServerArgs,
    session_semaphore: asyncio.Semaphore,
) -> None:
    try:
        await websocket.accept()
    except (WebSocketDisconnect, RuntimeError):
        return

    # locked() == True iff value == 0; check + acquire is atomic in single-threaded asyncio.
    if session_semaphore.locked():
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "event_id": f"event_{uuid.uuid4().hex[:24]}",
                        "error": {
                            "type": "rate_limit_exceeded",
                            "code": "too_many_sessions",
                            "message": (
                                f"Maximum concurrent sessions reached "
                                f"({server_args.asr_max_concurrent_sessions})."
                            ),
                        },
                    }
                )
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass
        return
    await session_semaphore.acquire()

    try:
        await RealtimeConnection(
            websocket, tokenizer_manager, adapter, server_args
        ).handle()
    finally:
        session_semaphore.release()
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass
