"""WebSocket session for realtime ASR.

Pre-commit deltas reference the reserved current_item_id that the
subsequent input_audio_buffer.committed and conversation.item.created
events will announce — sglang-specific, deviates from OpenAI's
commit-only delta emission.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pybase64
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    ConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    Error as TranscriptionFailedError,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as InputAudioContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_error import RealtimeError
from openai.types.realtime.realtime_transcription_session_audio_input_turn_detection import (
    ServerVad,
)
from pydantic import BaseModel, ValidationError

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.realtime.protocol import (
    DEFAULT_INPUT_SAMPLE_RATE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    AudioPCM,
    SessionUpdateEvent,
    TranscriptionSessionAudioInput,
    TranscriptionSessionConfig,
)
from sglang.srt.entrypoints.openai.realtime.vad import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    StreamingVAD,
    VADConfig,
    VADEvent,
    offset_to_ms,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    compute_window_drop,
    needs_space,
    normalize_whitespace,
    process_asr_chunk,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)


# PCM16: 16-bit samples → 2 bytes each. Used for frame-length validation
# and bytes/sec arithmetic against `np.frombuffer(..., dtype=np.int16)` below.
_SAMPLE_WIDTH = 2

# Sentinel: turn_detection validation failed and an error was already sent.
_INVALID = object()


def _resample_to_target_rate(pcm: bytes, src_rate: int, target_rate: int) -> bytes:
    if src_rate == target_rate or not pcm:
        return pcm
    import torch
    import torchaudio

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples).unsqueeze(0)
    audio = torchaudio.functional.resample(
        audio, orig_freq=src_rate, new_freq=target_rate
    )
    samples = audio.squeeze(0).numpy()
    # Clip to int16 range via 2^15 - 1 so a clipped 1.0 stays representable.
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


_CLIENT_EVENT_TYPES: Dict[str, type] = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
}


def _parse_client_event(raw: Dict[str, Any]) -> Optional[BaseModel]:
    """Parse, returning None if type is unknown. Raises ValidationError on
    a malformed payload of a known type."""
    cls = _CLIENT_EVENT_TYPES.get(raw.get("type"))
    if cls is None:
        return None
    return cls.model_validate(raw)


@dataclass
class _SessionConfig:
    """Session-level configuration negotiated via session.update: audio
    input format, requested language, sampling params. Persists until
    the session ends; ``configured`` gates audio-frame handling so the
    server doesn't run inference on PCM sent before session.update."""

    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    language: Optional[str] = None
    client_model: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    turn_detection: Optional[VADConfig] = None
    configured: bool = False


@dataclass
class _AudioState:
    """Per-item audio state: PCM buffer accumulated from
    input_audio_buffer.append, the chunked ASR rollback state, and the
    static buffer-size limits set at __init__. pcm_buffer / state /
    last_inference_offset reset on commit-roll and clear; the size limits
    stay constant for the session's lifetime."""

    max_buffer_bytes: int
    chunk_size_bytes: int
    state: StreamingASRState
    pcm_buffer: bytearray = field(default_factory=bytearray)
    last_inference_offset: int = 0


@dataclass
class _ItemState:
    """Per-item conversation-item ids and the wire-formatted deltas
    emitted so far for the current item. current_item_id is reserved at
    __init__ and only announced to the client by
    input_audio_buffer.committed."""

    current_item_id: str
    previous_item_id: Optional[str] = None
    emitted_deltas: List[str] = field(default_factory=list)


class RealtimeConnection:
    """One realtime transcription session. Drives the WS receive loop,
    dispatches typed client events to the matching _on_* handler, and
    triggers chunked ASR inference at audio buffer thresholds."""

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

        self.session_id = f"sess_{random_uuid()}"
        self._current_client_event_id: Optional[str] = None

        self.model_sample_rate = adapter.model_sample_rate
        self.bytes_per_second = self.model_sample_rate * _SAMPLE_WIDTH
        self.max_buffer_seconds = server_args.asr_max_buffer_seconds

        self.config = _SessionConfig()

        state = StreamingASRState(**adapter.chunked_streaming_config)
        chunk_size_bytes = int(state.chunk_size_sec * self.bytes_per_second)
        if chunk_size_bytes <= 0:
            raise RuntimeError(
                f"adapter.chunked_streaming_config produced non-positive "
                f"chunk_size_sec; got {state.chunk_size_sec!r}"
            )
        self.audio = _AudioState(
            max_buffer_bytes=self.max_buffer_seconds * self.bytes_per_second,
            chunk_size_bytes=chunk_size_bytes,
            state=state,
        )

        self.item = _ItemState(current_item_id=f"item_{random_uuid()}")

        # Server-side VAD (turn_detection: server_vad); None = client commits.
        self.vad: Optional[StreamingVAD] = None
        # Session-clock sample offset (model rate) of pcm_buffer byte 0.
        # Advances whenever buffer bytes are consumed or dropped, so VAD
        # sample offsets can be mapped to buffer byte offsets.
        self._buffer_origin_samples = 0

    async def run(self) -> None:
        await self._send(
            SessionCreatedEvent(
                event_id=f"event_{random_uuid()}",
                type="session.created",
                session=self._build_session_info(),
            )
        )

        try:
            await self._run_loop()
        except WebSocketDisconnect:
            logger.info("[realtime] client disconnected: %s", self.session_id)
        except Exception:
            logger.exception("[realtime] unexpected error: %s", self.session_id)
            try:
                await self._send_error(
                    "inference_failed",
                    "Internal server error",
                    error_type="server_error",
                )
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.debug(
                    "[realtime] failed to notify client of unexpected error: %s",
                    e,
                )

    async def _run_loop(self) -> None:
        """Receive-and-dispatch loop. Validation errors emit an error event
        and continue; fatal append-path errors (buffer overflow, append-time
        inference failure) close the WebSocket and terminate the loop.
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
                        "invalid_payload",
                        "Binary frames are not supported on /v1/realtime; "
                        "use input_audio_buffer.append with base64 audio.",
                    )
                continue

            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                await self._send_error("invalid_payload", "Invalid JSON")
                continue
            if not isinstance(raw, dict):
                await self._send_error(
                    "invalid_payload", "Top-level event must be a JSON object"
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
                    "invalid_value",
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
        """Returns True if the session should terminate."""
        if isinstance(event, InputAudioBufferAppendEvent):
            return await self._on_input_audio_buffer_append(event)
        if isinstance(event, SessionUpdateEvent):
            await self._on_session_update(event)
        elif isinstance(event, InputAudioBufferCommitEvent):
            await self._on_input_audio_buffer_commit(event)
        elif isinstance(event, InputAudioBufferClearEvent):
            await self._on_input_audio_buffer_clear(event)
        return False

    async def _on_session_update(self, event: SessionUpdateEvent) -> None:
        cfg = event.session

        # Normalize audio to an empty input cfg if absent so downstream
        # `audio.X is not None` reads as a business rule, not an existence check.
        # transcription stays nullable so partial-update can detect whether
        # the client sent the block.
        audio = (
            cfg.audio.input if cfg.audio else None
        ) or TranscriptionSessionAudioInput()
        transcription = audio.transcription

        # Validate first, then mutate config only after the whole update is accepted.
        # Partial-update: an absent turn_detection keeps the current VAD;
        # only an explicit null disables it (mirrors the transcription block).
        if "turn_detection" in audio.model_fields_set:
            vad = await self._validate_turn_detection(audio.turn_detection)
            if vad is _INVALID:
                return
            vad_cfg = vad.config if isinstance(vad, StreamingVAD) else None
        else:
            vad = self.vad
            vad_cfg = self.config.turn_detection
        if vad_cfg != self.config.turn_detection and self.audio.pcm_buffer:
            await self._send_error(
                "invalid_state",
                "Cannot change turn_detection while audio is buffered; "
                "commit or clear the current item first.",
                param="session.audio.input.turn_detection",
            )
            return
        if audio.noise_reduction is not None:
            await self._send_error(
                "not_supported",
                "audio.input.noise_reduction is not supported; set to null.",
                param="session.audio.input.noise_reduction",
            )
            return
        if transcription is not None and transcription.prompt is not None:
            await self._send_error(
                "not_supported",
                "audio.input.transcription.prompt is not supported.",
                param="session.audio.input.transcription.prompt",
            )
            return
        if (
            transcription is not None
            and transcription.model
            and transcription.model != self.server_args.served_model_name
        ):
            await self._send_error(
                "not_supported",
                f"Model {transcription.model!r} is not served by this endpoint "
                f"(serving {self.server_args.served_model_name!r}); set "
                f"transcription.model to null or to the server's model name.",
                param="session.audio.input.transcription.model",
            )
            return

        new_rate = self.config.input_sample_rate  # default: keep current
        fmt = audio.format
        if fmt is not None:
            if not isinstance(fmt, AudioPCM):
                # G.711 (pcmu / pcma): not implemented.
                await self._send_error(
                    "not_supported",
                    f"audio.input.format.type must be 'audio/pcm'; "
                    f"{fmt.type!r} is not implemented",
                    param="session.audio.input.format.type",
                )
                return
            if fmt.rate is not None and fmt.rate not in SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    "invalid_value",
                    f"audio.input.format.rate must be one of "
                    f"{SUPPORTED_INPUT_SAMPLE_RATES}, got {fmt.rate}",
                    param="session.audio.input.format.rate",
                )
                return
            new_rate = fmt.rate or DEFAULT_INPUT_SAMPLE_RATE
            # Changing the rate mid-item would leave already-buffered PCM
            # at the old rate mixed with new audio at the new rate, so
            # require the client to commit or clear before switching.
            if new_rate != self.config.input_sample_rate and self.audio.pcm_buffer:
                await self._send_error(
                    "invalid_state",
                    "Cannot change audio.input.format.rate while audio is "
                    "buffered; commit or clear the current item first.",
                    param="session.audio.input.format.rate",
                )
                return

        # Mutation pass — no early returns past this point.
        self.config.input_sample_rate = new_rate
        if transcription is not None:
            self.config.client_model = transcription.model
            self.config.language = transcription.language
        self.config.sampling_params = self.adapter.build_sampling_params(
            TranscriptionRequest(language=self.config.language)
        )
        self.config.turn_detection = vad_cfg
        self.vad = vad if isinstance(vad, StreamingVAD) else None
        self.config.configured = True

        # Side effects: log + ack.
        if cfg.include:
            logger.info(
                "[realtime] %s: include[] received but not implemented; ignoring: %s",
                self.session_id,
                cfg.include,
            )
        if self.config.input_sample_rate != self.model_sample_rate:
            logger.info(
                "[realtime] %s configured: resample %d→%d (ratio %.2f), language=%s",
                self.session_id,
                self.config.input_sample_rate,
                self.model_sample_rate,
                self.config.input_sample_rate / self.model_sample_rate,
                self.config.language,
            )
        await self._send(
            SessionUpdatedEvent(
                event_id=f"event_{random_uuid()}",
                type="session.updated",
                session=self._build_session_info(),
            )
        )

    async def _validate_turn_detection(self, td: Any):
        """Validate the requested turn_detection and build its VAD.

        Returns None (VAD off), a ready StreamingVAD, or the _INVALID
        sentinel after sending the error event. Constructing the VAD here
        keeps the mutation pass in _on_session_update free of failure
        paths (silero-vad import / model load happen on this side).
        """
        if td is None:
            return None
        if not isinstance(td, ServerVad):
            await self._send_error(
                "not_supported",
                "turn_detection.type must be 'server_vad'; semantic_vad "
                "is not implemented.",
                param="session.audio.input.turn_detection.type",
            )
            return _INVALID
        if td.idle_timeout_ms is not None:
            await self._send_error(
                "not_supported",
                "turn_detection.idle_timeout_ms is not supported; set to null.",
                param="session.audio.input.turn_detection.idle_timeout_ms",
            )
            return _INVALID
        # silero-vad scores 512-sample windows @ 16 kHz; audio is VAD-scored
        # after resampling to the model rate, so the model rate must match.
        if self.model_sample_rate != VAD_SAMPLE_RATE:
            await self._send_error(
                "not_supported",
                f"server_vad requires a {VAD_SAMPLE_RATE} Hz model input rate; "
                f"this model expects {self.model_sample_rate} Hz.",
                param="session.audio.input.turn_detection",
            )
            return _INVALID

        cfg = VADConfig(
            threshold=td.threshold if td.threshold is not None else 0.5,
            prefix_padding_ms=(
                td.prefix_padding_ms if td.prefix_padding_ms is not None else 300
            ),
            silence_duration_ms=(
                td.silence_duration_ms if td.silence_duration_ms is not None else 500
            ),
        )
        if not 0.0 <= cfg.threshold <= 1.0:
            await self._send_error(
                "invalid_value",
                f"turn_detection.threshold must be in [0, 1], got {cfg.threshold}",
                param="session.audio.input.turn_detection.threshold",
            )
            return _INVALID
        if cfg.prefix_padding_ms < 0 or cfg.silence_duration_ms <= 0:
            await self._send_error(
                "invalid_value",
                "turn_detection.prefix_padding_ms must be >= 0 and "
                "silence_duration_ms must be > 0",
                param="session.audio.input.turn_detection",
            )
            return _INVALID

        if self.vad is not None and self.vad.config == cfg:
            return self.vad
        try:
            vad = StreamingVAD(cfg)
        except ImportError as e:
            await self._send_error(
                "not_supported",
                str(e),
                param="session.audio.input.turn_detection",
            )
            return _INVALID
        # Align the VAD clock with the buffer clock. The caller rejects a
        # turn_detection change while audio is buffered, so buffer byte 0
        # is "now" on the session clock.
        vad.samples_consumed = self._buffer_origin_samples
        return vad

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> bool:
        """Returns True if the session should terminate (buffer overflow or
        append-time inference failure)."""
        if not self.config.configured:
            await self._send_error(
                "invalid_state", "Send session.update before audio frames"
            )
            return False

        # Empty audio is a no-op (heartbeat frames); skip b64decode.
        if not event.audio:
            return False

        try:
            data = pybase64.b64decode(event.audio, validate=True)
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

        # Estimate post-resample size before resampling so oversized frames fail early.
        src_samples = len(data) // _SAMPLE_WIDTH
        target_samples = math.ceil(
            src_samples * self.model_sample_rate / self.config.input_sample_rate
        )
        if (
            len(self.audio.pcm_buffer) + target_samples * _SAMPLE_WIDTH
            > self.audio.max_buffer_bytes
        ):
            # Close 1009 ("message too big") so clients can distinguish
            # session-resource exhaustion from a normal close.
            await self._send_error_and_close(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.max_buffer_seconds}s; "
                f"client is sending faster than inference can keep up",
                close_code=1009,
            )
            return True

        if self.config.input_sample_rate != self.model_sample_rate:
            data = await asyncio.to_thread(
                _resample_to_target_rate,
                data,
                self.config.input_sample_rate,
                self.model_sample_rate,
            )
        self.audio.pcm_buffer.extend(data)

        if self.vad is not None:
            await self._process_vad(data)
            if not self.vad.is_speech:
                # Between utterances: keep only the speech-start prefix
                # window and skip inference, so an open mic idles
                # indefinitely without burning compute on silence or
                # hitting the buffer cap.
                self._trim_pre_speech_buffer()
                return False

        new_audio_bytes = len(self.audio.pcm_buffer) - self.audio.last_inference_offset
        if new_audio_bytes >= self.audio.chunk_size_bytes:
            ok = await self._run_inference(is_last=False)
            if not ok:
                # WS already closed inside _run_inference.
                return True
        return False

    async def _on_input_audio_buffer_commit(
        self, event: InputAudioBufferCommitEvent
    ) -> None:
        if not self.config.configured:
            await self._send_error("invalid_state", "Send session.update before commit")
            return
        if self.vad is not None:
            await self._send_error(
                "invalid_state",
                "Cannot commit: the buffer is committed automatically when "
                "turn_detection is server_vad.",
            )
            return
        if not self.audio.pcm_buffer and not self.audio.state.full_transcript:
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return
        await self._commit_item()

    async def _commit_item(self) -> None:
        """Commit the current item: announce it, run the final inference
        pass, emit transcription.completed, and roll to the next item.
        Shared by the manual commit handler and VAD auto-commit."""
        has_new_audio = len(self.audio.pcm_buffer) > self.audio.last_inference_offset
        item_id = self.item.current_item_id
        prev_item_id = self.item.previous_item_id

        partial_transcript = normalize_whitespace("".join(self.item.emitted_deltas))

        await self._send(
            InputAudioBufferCommittedEvent(
                event_id=f"event_{random_uuid()}",
                type="input_audio_buffer.committed",
                item_id=item_id,
                previous_item_id=prev_item_id,
            )
        )
        await self._send(
            ConversationItemCreatedEvent(
                event_id=f"event_{random_uuid()}",
                type="conversation.item.created",
                previous_item_id=prev_item_id,
                item=RealtimeConversationItemUserMessage(
                    id=item_id,
                    type="message",
                    role="user",
                    status="completed",
                    content=[
                        InputAudioContent(
                            type="input_audio", transcript=partial_transcript
                        )
                    ],
                ),
            )
        )

        # Capture pcm duration before `_start_next_item()` runs: starting
        # the next item clears pcm_buffer, so reading it after gives 0.
        pcm_duration_seconds = len(self.audio.pcm_buffer) / self.bytes_per_second

        if has_new_audio:
            ok = await self._run_inference(is_last=True)
            if not ok:
                # _run_inference already emitted transcription.failed and
                # rolled the item; don't also emit completed.
                return
        elif self.audio.state.full_transcript:
            # Audio length was exactly a chunk_size_bytes multiple. Flush
            # the tail tokens update() held back.
            tail = self.audio.state.finalize()
            await self._emit_transcription_delta(tail)

        # Build from emitted_deltas, not state.full_transcript: prefix injection
        # means the last chunk's full_transcript is only the continuation tail.
        transcript = normalize_whitespace("".join(self.item.emitted_deltas))

        await self._send(
            ConversationItemInputAudioTranscriptionCompletedEvent(
                event_id=f"event_{random_uuid()}",
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                content_index=0,
                transcript=transcript,
                usage=UsageTranscriptTextUsageDuration(
                    type="duration", seconds=pcm_duration_seconds
                ),
            )
        )

        self._start_next_item()

    async def _on_input_audio_buffer_clear(
        self, event: InputAudioBufferClearEvent
    ) -> None:
        # Reserve a fresh current_item_id so post-clear pre-commit deltas
        # don't share an item_id with deltas the client already received
        # for the abandoned audio. previous_item_id is NOT touched — the
        # cleared item was never committed, so the prior-commit chain
        # shouldn't include it.
        self._reset_inference_state()
        self.item.current_item_id = f"item_{random_uuid()}"
        if self.vad is not None:
            # Abandoned speech must not leak a dangling is_speech into the
            # next utterance.
            self.vad.end_utterance()
        await self._send(
            InputAudioBufferClearedEvent(
                event_id=f"event_{random_uuid()}", type="input_audio_buffer.cleared"
            )
        )

    async def _process_vad(self, data: bytes) -> None:
        """Score newly appended model-rate PCM and act on VAD transitions:
        speech_started/speech_stopped events, auto-commit on stop."""
        assert self.vad is not None
        for emit in self.vad.process(data):
            if emit.event_type == VADEvent.SPEECH_STARTED:
                await self._send(
                    InputAudioBufferSpeechStartedEvent(
                        event_id=f"event_{random_uuid()}",
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=offset_to_ms(emit.sample_offset),
                        item_id=self.item.current_item_id,
                    )
                )
            else:
                await self._send(
                    InputAudioBufferSpeechStoppedEvent(
                        event_id=f"event_{random_uuid()}",
                        type="input_audio_buffer.speech_stopped",
                        audio_end_ms=offset_to_ms(emit.sample_offset),
                        item_id=self.item.current_item_id,
                    )
                )
                await self._auto_commit(emit.sample_offset)

    async def _auto_commit(self, stop_offset_samples: int) -> None:
        """Commit the utterance that ended at stop_offset_samples. Audio
        past that offset (the silence run, possibly the head of a rapid
        next turn) is carried over into the next item's buffer."""
        keep_from = (stop_offset_samples - self._buffer_origin_samples) * _SAMPLE_WIDTH
        keep_from = min(max(keep_from, 0), len(self.audio.pcm_buffer))
        tail = bytes(self.audio.pcm_buffer[keep_from:])
        del self.audio.pcm_buffer[keep_from:]
        if self.audio.pcm_buffer or self.audio.state.full_transcript:
            await self._commit_item()
        self.audio.pcm_buffer.extend(tail)

    def _maybe_roll_audio_window(self) -> None:
        """Bound per-inference cost: once the buffered audio exceeds the
        adapter's window, drop the head whose transcript is already
        emitted (it conditions the next inference as the prompt prefix).
        Keeps per-chunk cost O(window) instead of O(utterance)."""
        drop = compute_window_drop(
            buffered=len(self.audio.pcm_buffer),
            inferred=self.audio.last_inference_offset,
            chunk_size=self.audio.chunk_size_bytes,
            state=self.audio.state,
        )
        if drop <= 0:
            return
        del self.audio.pcm_buffer[:drop]
        self._buffer_origin_samples += drop // _SAMPLE_WIDTH
        self.audio.last_inference_offset -= drop
        self.audio.state.start_new_window()

    def _trim_pre_speech_buffer(self) -> None:
        """Drop idle audio, keeping the prefix_padding window plus one VAD
        frame of detection latency ahead of a future speech_started. Only
        called while not in speech, where last_inference_offset is 0, so
        trimming never shifts audio an inference pass already consumed."""
        assert self.vad is not None
        keep_bytes = (
            self.vad.config.prefix_padding_ms * self.model_sample_rate // 1000
            + VAD_FRAME_SAMPLES
        ) * _SAMPLE_WIDTH
        drop = len(self.audio.pcm_buffer) - keep_bytes
        if drop > 0:
            del self.audio.pcm_buffer[:drop]
            self._buffer_origin_samples += drop // _SAMPLE_WIDTH

    async def _run_inference(self, is_last: bool) -> bool:
        """Run ASR on the current cumulative buffer. Returns False on failure:
        commit-time emits transcription.failed and rolls the item; append-time
        emits a generic error envelope and closes the WebSocket."""
        self._maybe_roll_audio_window()
        wav_data = await asyncio.to_thread(
            _pcm_to_wav, bytes(self.audio.pcm_buffer), self.model_sample_rate
        )
        try:
            delta = await process_asr_chunk(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                state=self.audio.state,
                audio_data=wav_data,
                sampling_params=self.config.sampling_params,
                is_last=is_last,
            )
        except Exception:
            logger.exception(
                "[realtime] inference failed: session=%s item=%s buffer_bytes=%d",
                self.session_id,
                self.item.current_item_id,
                len(self.audio.pcm_buffer),
            )
            if is_last:
                # Commit-time failure: committed + created already emitted,
                # so the item exists client-side and transcription.failed
                # can reference it. Wire message is hardcoded "Transcription
                # failed" — don't leak backend traces to the client; full
                # error is in the logger.exception above.
                await self._send(
                    ConversationItemInputAudioTranscriptionFailedEvent(
                        event_id=f"event_{random_uuid()}",
                        type="conversation.item.input_audio_transcription.failed",
                        item_id=self.item.current_item_id,
                        content_index=0,
                        error=TranscriptionFailedError(
                            type="server_error",
                            code="inference_failed",
                            message="Transcription failed",
                        ),
                    )
                )
                self._start_next_item()
            else:
                # Append-time failure: the item isn't visible client-side
                # yet (committed/created fire at commit), so
                # transcription.failed would reference a ghost id.
                await self._send_error_and_close(
                    "inference_failed",
                    "Transcription failed",
                    close_code=1011,
                )
            return False

        self.audio.last_inference_offset = len(self.audio.pcm_buffer)
        await self._emit_transcription_delta(delta)
        return True

    async def _emit_transcription_delta(self, delta: str) -> None:
        """emitted_deltas stores wire-formatted text (with leading
        boundary spaces baked in), so "".join(...) reconstructs the
        cumulative transcript verbatim."""
        if not delta:
            return
        for word in delta.split(" "):
            if not word:
                continue
            prev = self.item.emitted_deltas[-1] if self.item.emitted_deltas else ""
            formatted = f" {word}" if needs_space(prev, word) else word
            self.item.emitted_deltas.append(formatted)
            await self._send(
                ConversationItemInputAudioTranscriptionDeltaEvent(
                    event_id=f"event_{random_uuid()}",
                    type="conversation.item.input_audio_transcription.delta",
                    item_id=self.item.current_item_id,
                    content_index=0,
                    delta=formatted,
                )
            )

    def _start_next_item(self) -> None:
        self.item.previous_item_id = self.item.current_item_id
        self.item.current_item_id = f"item_{random_uuid()}"
        self._reset_inference_state()

    def _reset_inference_state(self) -> None:
        """Missing any of these resets leaks state across items."""
        # Dropped buffer bytes advance the session clock so future VAD
        # offsets still map to buffer byte offsets.
        self._buffer_origin_samples += len(self.audio.pcm_buffer) // _SAMPLE_WIDTH
        self.audio.state = StreamingASRState(**self.adapter.chunked_streaming_config)
        self.audio.pcm_buffer.clear()  # in-place; reuses the buffer's allocation
        self.item.emitted_deltas.clear()
        self.audio.last_inference_offset = 0

    def _build_session_info(self) -> TranscriptionSessionConfig:
        # id / object aren't SDK fields; round-trip via extra='allow' so
        # dumps emit them like the real server.
        return TranscriptionSessionConfig.model_validate(
            {
                "type": "transcription",
                "id": self.session_id,
                "object": "realtime.transcription_session",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": self.config.input_sample_rate,
                        },
                        "transcription": {
                            "model": self.config.client_model,
                            "language": self.config.language,
                        },
                        "noise_reduction": None,
                        "turn_detection": (
                            None
                            if self.config.turn_detection is None
                            else {
                                "type": "server_vad",
                                "threshold": self.config.turn_detection.threshold,
                                "prefix_padding_ms": (
                                    self.config.turn_detection.prefix_padding_ms
                                ),
                                "silence_duration_ms": (
                                    self.config.turn_detection.silence_duration_ms
                                ),
                            }
                        ),
                    }
                },
            }
        )

    async def _send(self, event: BaseModel) -> None:
        await self.websocket.send_text(event.model_dump_json())

    async def _send_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
    ) -> None:
        envelope = RealtimeErrorEvent(
            event_id=f"event_{random_uuid()}",
            type="error",
            error=RealtimeError(
                type=error_type,
                code=code,
                message=message,
                param=param,
                event_id=self._current_client_event_id,
            ),
        )
        await self.websocket.send_text(envelope.model_dump_json())

    async def _send_error_and_close(
        self,
        code: str,
        message: str,
        *,
        close_code: int,
        error_type: str = "server_error",
    ) -> None:
        # Independent try-blocks: a failed send must not skip the close.
        # We still need to release local starlette socket state even when
        # the wire send doesn't reach the peer.
        try:
            await self._send_error(code, message, error_type=error_type)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] send error %s before close failed: %s", code, e)
        try:
            await self.websocket.close(code=close_code)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] close %d after %s failed: %s", close_code, code, e)
