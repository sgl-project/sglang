"""WebSocket session for realtime ASR.

One coroutine per session, doing ws.receive() and inference in turn.
librosa.resample runs in a worker thread. handler.py owns transport.

Lifecycle: session.created -> session.update -> input_audio_buffer.append(*)
-> input_audio_buffer.commit -> (committed, created, delta(s), completed)
-> next item. See _run_loop and _on_* handlers for per-event logic.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.realtime.protocol import (
    CODE_INFERENCE_FAILED,
    CODE_INVALID_PAYLOAD,
    CODE_INVALID_STATE,
    CODE_INVALID_VALUE,
    CODE_NOT_SUPPORTED,
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_SERVER,
    AudioInputConfig,
    AudioInputFormatPCM,
    ConversationItemCreated,
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCleared,
    InputAudioBufferCommit,
    InputAudioBufferCommitted,
    InputAudioContent,
    Item,
    SessionCreated,
    SessionInfo,
    SessionUpdate,
    SessionUpdated,
    TranscriptionCompleted,
    TranscriptionDelta,
    TranscriptionFailed,
    TranscriptionFailedError,
    UsageDuration,
    format_error_envelope,
    new_item_id,
    new_session_id,
)
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
_BYTES_PER_SECOND = _MODEL_SAMPLE_RATE * _SAMPLE_WIDTH
_DEFAULT_INPUT_SAMPLE_RATE = 24000  # OpenAI default for audio/pcm
_SUPPORTED_INPUT_SAMPLE_RATES = (16000, 24000, 48000)  # sglang extension


def _resample_to_model_rate(pcm: bytes, src_rate: int) -> bytes:
    """Resample client PCM16 to the model's 16 kHz rate via librosa.

    Runs on a worker thread (via asyncio.to_thread in
    _on_input_audio_buffer_append), so it must not touch
    RealtimeConnection state.
    """
    if src_rate == _MODEL_SAMPLE_RATE or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    samples = librosa.resample(samples, orig_sr=src_rate, target_sr=_MODEL_SAMPLE_RATE)
    # Re-encode by 2^15 - 1 so a clipped 1.0 stays in int16 range.
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _pcm_to_wav(pcm: bytes) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, _MODEL_SAMPLE_RATE, format="WAV")
    return buf.getvalue()


# Manual lookup so unknown event types return unknown_event; a Pydantic
# discriminated union would lump them with invalid_value.
_CLIENT_EVENT_TYPES: Dict[str, type] = {
    "session.update": SessionUpdate,
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.commit": InputAudioBufferCommit,
    "input_audio_buffer.clear": InputAudioBufferClear,
}


def _parse_client_event(raw: Dict[str, Any]) -> Optional[BaseModel]:
    """Parse, returning None if type is unknown. Raises ValidationError on
    a malformed payload of a known type."""
    cls = _CLIENT_EVENT_TYPES.get(raw.get("type"))
    if cls is None:
        return None
    return cls.model_validate(raw)


class RealtimeConnection:
    """One realtime transcription session.

    Owns the WebSocket, the cumulative PCM buffer, StreamingASRState, the
    current and previous item ids, and the emitted-words list for the final
    transcript.
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

        self.session_id = new_session_id()

        # Config: populated from session.update; defaults pre-update.
        self.input_sample_rate = _DEFAULT_INPUT_SAMPLE_RATE
        self.language: Optional[str] = None
        self.client_model: Optional[str] = None
        self.sampling_params: Optional[Dict[str, Any]] = None
        self.session_configured = False
        self._current_client_event_id: Optional[str] = None

        # Inference state.
        self.state = StreamingASRState(**adapter.chunked_streaming_config)
        self.pcm_buffer = bytearray()
        self.last_inference_offset = 0
        self.emitted_words: List[str] = []

        # Item lifecycle: a new current_item_id is generated only after commit.
        self.current_item_id = new_item_id()
        self.previous_item_id: Optional[str] = None

        self.max_buffer_seconds = server_args.asr_max_buffer_seconds
        self.max_buffer_bytes = self.max_buffer_seconds * _BYTES_PER_SECOND
        self.chunk_size_bytes = int(self.state.chunk_size_sec * _BYTES_PER_SECOND)
        # Zero or negative chunk_size_sec would skip the chunk-trigger in
        # `_on_input_audio_buffer_append` entirely. Fail at construction instead.
        if self.chunk_size_bytes <= 0:
            raise RuntimeError(
                f"adapter.chunked_streaming_config produced non-positive "
                f"chunk_size_sec; got {self.state.chunk_size_sec!r}"
            )

    async def run(self) -> None:
        """Entry point. Send session.created, then run the receive loop."""
        # adapter compatibility is checked in handler before construction
        await self._send(SessionCreated(session=self._build_session_info()))

        try:
            await self._run_loop()
        except WebSocketDisconnect:
            logger.info("[realtime] client disconnected: %s", self.session_id)
        except Exception:
            logger.exception("[realtime] unexpected error: %s", self.session_id)
            try:
                await self._send_error(
                    CODE_INFERENCE_FAILED,
                    "Internal server error",
                    error_type=ERROR_TYPE_SERVER,
                )
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug(
                    "[realtime] failed to notify client of unexpected error: %s",
                    e,
                )

    async def _run_loop(self) -> None:
        """Receive-and-dispatch loop. Validates each frame in three layers
        (transport, syntax, semantic). Each layer emits an error event and
        continues; only buffer overflow from `input_audio_buffer.append`
        terminates.
        """
        while True:
            self._current_client_event_id = None
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            text = message.get("text")
            if not text:
                if message.get("bytes") is not None:
                    # OpenAI Realtime is base64 PCM in JSON; binary frames aren't supported.
                    await self._send_error(
                        CODE_INVALID_PAYLOAD,
                        "Binary frames are not supported on /v1/realtime; "
                        "use input_audio_buffer.append with base64 audio.",
                    )
                continue

            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                await self._send_error(CODE_INVALID_PAYLOAD, "Invalid JSON")
                continue
            if not isinstance(raw, dict):
                await self._send_error(
                    CODE_INVALID_PAYLOAD, "Top-level event must be a JSON object"
                )
                continue

            self._current_client_event_id = raw.get("event_id")
            try:
                event = _parse_client_event(raw)
            except ValidationError as e:
                # Report first error only; matches OpenAI server behavior.
                err = e.errors()[0]
                loc = ".".join(str(x) for x in err["loc"])
                await self._send_error(
                    CODE_INVALID_VALUE,
                    err.get("msg") or "Invalid payload",
                    param=loc or None,
                )
                continue
            if event is None:
                await self._send_error(
                    "unknown_event",
                    f"Unknown event type: {raw.get('type')!r}",
                )
                continue
            terminate = await self._dispatch(event)
            if terminate:
                return

    async def _dispatch(self, event: BaseModel) -> bool:
        """Returns True if the session should terminate (buffer overflow)."""
        # Append is the only handler that can signal termination (buffer
        # overflow triggers close 1009). Others fall through to False.
        if isinstance(event, InputAudioBufferAppend):
            return await self._on_input_audio_buffer_append(event)
        if isinstance(event, SessionUpdate):
            await self._on_session_update(event)
        elif isinstance(event, InputAudioBufferCommit):
            await self._on_input_audio_buffer_commit(event)
        elif isinstance(event, InputAudioBufferClear):
            await self._on_input_audio_buffer_clear(event)
        return False

    async def _on_session_update(self, event: SessionUpdate) -> None:
        cfg = event.session

        # Normalize audio to an empty AudioInputConfig if absent so downstream
        # `audio.X is not None` reads as a business rule, not an existence check.
        # transcription stays nullable so partial-update can detect whether
        # the client sent the block.
        audio = (cfg.audio.input if cfg.audio else None) or AudioInputConfig()
        transcription = audio.transcription

        fmt = audio.format
        if fmt is not None:
            if not isinstance(fmt, AudioInputFormatPCM):
                # G.711 (pcmu / pcma): not implemented.
                await self._send_error(
                    CODE_NOT_SUPPORTED,
                    f"audio.input.format.type must be 'audio/pcm'; "
                    f"{fmt.type!r} is not implemented",
                    param="session.audio.input.format.type",
                )
                return
            if fmt.rate is not None and fmt.rate not in _SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    CODE_INVALID_VALUE,
                    f"audio.input.format.rate must be one of "
                    f"{_SUPPORTED_INPUT_SAMPLE_RATES}, got {fmt.rate}",
                    param="session.audio.input.format.rate",
                )
                return
            self.input_sample_rate = fmt.rate or _DEFAULT_INPUT_SAMPLE_RATE

        if audio.turn_detection is not None:
            await self._send_error(
                CODE_NOT_SUPPORTED,
                "Server-side VAD is not implemented; "
                "set audio.input.turn_detection: null and commit explicitly.",
                param="session.audio.input.turn_detection",
            )
            return
        if audio.noise_reduction is not None:
            await self._send_error(
                CODE_NOT_SUPPORTED,
                "audio.input.noise_reduction is not supported; set to null.",
                param="session.audio.input.noise_reduction",
            )
            return
        if transcription is not None and transcription.prompt is not None:
            await self._send_error(
                CODE_NOT_SUPPORTED,
                "audio.input.transcription.prompt is not supported.",
                param="session.audio.input.transcription.prompt",
            )
            return

        if cfg.include:
            logger.info(
                "[realtime] %s: include[] received but not implemented; ignoring: %s",
                self.session_id,
                cfg.include,
            )

        if transcription is not None:
            self.client_model = transcription.model
            self.language = transcription.language

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
        await self._send(SessionUpdated(session=self._build_session_info()))

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppend
    ) -> bool:
        """Returns True if the session should terminate (buffer overflow)."""
        if not self.session_configured:
            await self._send_error(
                CODE_INVALID_STATE, "Send session.update before audio frames"
            )
            return False

        try:
            data = base64.b64decode(event.audio, validate=True)
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

        # librosa.resample is sync CPU work; offload so concurrent sessions
        # don't serialize on the event loop.
        if self.input_sample_rate != _MODEL_SAMPLE_RATE:
            data = await asyncio.to_thread(
                _resample_to_model_rate, data, self.input_sample_rate
            )
        self.pcm_buffer.extend(data)

        if len(self.pcm_buffer) > self.max_buffer_bytes:
            # Close 1009 ("message too big") so clients can distinguish
            # session-resource exhaustion from a normal close.
            await self._send_error(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.max_buffer_seconds}s; "
                f"client is sending faster than inference can keep up",
                error_type=ERROR_TYPE_SERVER,
            )
            try:
                await self.websocket.close(code=1009)
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug(
                    "[realtime] failed to close 1009 after buffer overflow: %s", e
                )
            return True

        # One inference per chunk_size_bytes of NEW audio. A big append
        # spanning multiple chunks fires only one inference.
        new_audio_bytes = len(self.pcm_buffer) - self.last_inference_offset
        if new_audio_bytes >= self.chunk_size_bytes:
            await self._run_inference(is_last=False)
        return False

    async def _on_input_audio_buffer_commit(
        self, event: InputAudioBufferCommit
    ) -> None:
        if not self.session_configured:
            await self._send_error(
                CODE_INVALID_STATE, "Send session.update before commit"
            )
            return
        if not self.pcm_buffer and not self.state.full_transcript:
            await self._send_error(
                CODE_INVALID_STATE, "Cannot commit an empty audio buffer"
            )
            return

        has_new_audio = len(self.pcm_buffer) > self.last_inference_offset
        item_id = self.current_item_id
        prev_item_id = self.previous_item_id

        # OpenAI canonical order: committed, created, delta(s), completed.
        # `created.item.content[0].transcript` carries the partial (pre-final
        # inference); `completed.transcript` carries the full result.
        partial_transcript = normalize_whitespace(" ".join(self.emitted_words))

        await self._send(
            InputAudioBufferCommitted(
                item_id=item_id,
                previous_item_id=prev_item_id,
            )
        )
        await self._send(
            ConversationItemCreated(
                previous_item_id=prev_item_id,
                item=Item(
                    id=item_id,
                    content=[InputAudioContent(transcript=partial_transcript)],
                ),
            )
        )

        # Capture pcm duration before `_start_next_item()` runs: starting
        # the next item clears pcm_buffer, so reading it after gives 0.
        pcm_duration_seconds = len(self.pcm_buffer) / _BYTES_PER_SECOND

        if has_new_audio:
            ok = await self._run_inference(is_last=True)
            if not ok:
                # transcription.failed already emitted by _run_inference; do
                # NOT also emit completed. _start_next_item() ran inside _run_inference.
                return
        elif self.state.full_transcript:
            # Audio length was exactly a chunk_size_bytes multiple. Flush
            # the tail tokens update() held back.
            tail = self.state.finalize()
            await self._emit_transcription_delta(tail)

        # Build from emitted_words, not state.full_transcript: prefix injection
        # means the last chunk's full_transcript is only the continuation tail.
        transcript = normalize_whitespace(" ".join(self.emitted_words))

        await self._send(
            TranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                usage=UsageDuration(seconds=pcm_duration_seconds),
            )
        )

        self._start_next_item()

    async def _on_input_audio_buffer_clear(self, event: InputAudioBufferClear) -> None:
        # current_item_id is NOT rolled: clear is not commit.
        self._reset_inference_state()
        await self._send(InputAudioBufferCleared())

    async def _run_inference(self, is_last: bool) -> bool:
        """Returns False on inference failure (after emitting transcription.failed
        and rolling the item)."""
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
                TranscriptionFailed(
                    item_id=self.current_item_id,
                    error=TranscriptionFailedError(message=str(e)),
                )
            )
            self._start_next_item()
            return False

        self.last_inference_offset = len(self.pcm_buffer)
        await self._emit_transcription_delta(delta)
        return True

    async def _emit_transcription_delta(self, delta: str) -> None:
        """One event per word so clients see streaming cadence."""
        if not delta:
            return
        for word in delta.split(" "):
            if not word:
                continue
            self.emitted_words.append(word)
            await self._send(
                TranscriptionDelta(item_id=self.current_item_id, delta=word)
            )

    def _start_next_item(self) -> None:
        """Move on to the next conversation item:
        - move current_item_id to previous_item_id
        - generate a fresh current_item_id
        - reset all per-item inference state (StreamingASRState, pcm_buffer,
          emitted_words, last_inference_offset)

        Called after transcription.completed (commit path) and
        transcription.failed (inference exception path).
        """
        self.previous_item_id = self.current_item_id
        self.current_item_id = new_item_id()
        self._reset_inference_state()

    def _reset_inference_state(self) -> None:
        """Per-item state shared by clear and commit-roll. Missing any of
        these leaks state across items."""
        self.state = StreamingASRState(**self.adapter.chunked_streaming_config)
        self.pcm_buffer.clear()  # in-place; reuses the buffer's allocation
        self.emitted_words.clear()
        self.last_inference_offset = 0

    def _build_session_info(self) -> SessionInfo:
        # `format` is a nested object {type, rate}; no `sample_rate` sibling.
        # Pre-update, client_model and language are None (set in __init__),
        # so session.created emits the canonical empty transcription block
        # without needing a separate "configured" flag.
        return SessionInfo(
            id=self.session_id,
            audio={
                "input": {
                    "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                    "transcription": {
                        "model": self.client_model,
                        "language": self.language,
                    },
                    "noise_reduction": None,
                    "turn_detection": None,
                }
            },
        )

    async def _send(self, event: BaseModel) -> None:
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = ERROR_TYPE_INVALID_REQUEST,
        param: Optional[str] = None,
    ) -> None:
        await self.websocket.send_text(
            format_error_envelope(
                code,
                message,
                error_type=error_type,
                param=param,
                client_event_id=self._current_client_event_id,
            )
        )
