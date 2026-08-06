"""WebSocket session for realtime ASR.

Pre-commit deltas reference the reserved current_item_id that the
subsequent input_audio_buffer.committed and conversation.item.created
events will announce — sglang-specific, deviates from OpenAI's
commit-only delta emission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pybase64
from fastapi import WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreatedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearedEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferCommittedEvent,
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
from pydantic import BaseModel, ValidationError

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
)
from sglang.srt.entrypoints.openai.realtime.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    resample_to_target_rate,
)
from sglang.srt.entrypoints.openai.realtime.protocol import (
    DEFAULT_INPUT_SAMPLE_RATE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    AudioPCM,
    SessionUpdateEvent,
    TranscriptionSessionConfig,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    needs_space,
    normalize_whitespace,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)


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
    configured: bool = False


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
        self.adapter = adapter
        self.server_args = server_args
        self.served_model_name = tokenizer_manager.served_model_name

        self.session_id = f"sess_{random_uuid()}"
        self._current_client_event_id: Optional[str] = None

        self.config = _SessionConfig()
        self.asr_processor = RealtimeASRProcessor(
            tokenizer_manager=tokenizer_manager,
            adapter=adapter,
            server_args=server_args,
        )
        self.asr_state = self.asr_processor.create_state()
        self.model_sample_rate = self.asr_processor.model_sample_rate

        self.item = _ItemState(current_item_id=f"item_{random_uuid()}")

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

        # Realtime session updates are patches. Pydantic preserves which nested
        # fields were sent, so omitted values must not reset earlier settings.
        session_audio = cfg.audio if "audio" in cfg.model_fields_set else None
        audio = (
            session_audio.input
            if session_audio is not None and "input" in session_audio.model_fields_set
            else None
        )
        audio_fields = audio.model_fields_set if audio is not None else set()
        transcription_present = "transcription" in audio_fields
        transcription = audio.transcription if transcription_present else None

        # Validate first, then mutate config only after the whole update is accepted.
        if audio is not None and audio.turn_detection is not None:
            await self._send_error(
                "not_supported",
                "Server-side VAD is not implemented; "
                "set audio.input.turn_detection: null and commit explicitly.",
                param="session.audio.input.turn_detection",
            )
            return
        if audio is not None and audio.noise_reduction is not None:
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
            and transcription.model != self.served_model_name
        ):
            await self._send_error(
                "not_supported",
                f"Model {transcription.model!r} is not served by this endpoint "
                f"(serving {self.served_model_name!r}); set "
                f"transcription.model to null or to the server's model name.",
                param="session.audio.input.transcription.model",
            )
            return

        new_rate = self.config.input_sample_rate  # default: keep current
        format_present = "format" in audio_fields
        fmt = audio.format if format_present else None
        if format_present:
            if fmt is None:
                new_rate = DEFAULT_INPUT_SAMPLE_RATE
            elif not isinstance(fmt, AudioPCM):
                # G.711 (pcmu / pcma): not implemented.
                await self._send_error(
                    "not_supported",
                    f"audio.input.format.type must be 'audio/pcm'; "
                    f"{fmt.type!r} is not implemented",
                    param="session.audio.input.format.type",
                )
                return
            elif fmt.rate is not None and fmt.rate not in SUPPORTED_INPUT_SAMPLE_RATES:
                await self._send_error(
                    "invalid_value",
                    f"audio.input.format.rate must be one of "
                    f"{SUPPORTED_INPUT_SAMPLE_RATES}, got {fmt.rate}",
                    param="session.audio.input.format.rate",
                )
                return
            else:
                new_rate = fmt.rate or DEFAULT_INPUT_SAMPLE_RATE
            # Changing the rate mid-item would leave already-buffered PCM
            # at the old rate mixed with new audio at the new rate, so
            # require the client to commit or clear before switching.
            if new_rate != self.config.input_sample_rate and self.asr_state.has_audio:
                await self._send_error(
                    "invalid_state",
                    "Cannot change audio.input.format.rate while audio is "
                    "buffered; commit or clear the current item first.",
                    param="session.audio.input.format.rate",
                )
                return

        # Mutation pass — no early returns past this point.
        self.config.input_sample_rate = new_rate
        if transcription_present:
            if transcription is None:
                self.config.client_model = None
                self.config.language = None
            else:
                if "model" in transcription.model_fields_set:
                    self.config.client_model = transcription.model
                if "language" in transcription.model_fields_set:
                    self.config.language = transcription.language
        self.config.sampling_params = self.adapter.build_sampling_params(
            TranscriptionRequest(language=self.config.language)
        )
        self.config.configured = True

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

    async def _on_input_audio_buffer_append(
        self, event: InputAudioBufferAppendEvent
    ) -> bool:
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

        if len(data) % PCM_SAMPLE_WIDTH_BYTES != 0:
            await self._send_error(
                "invalid_audio_format",
                "PCM16 frame length must be a multiple of "
                f"{PCM_SAMPLE_WIDTH_BYTES} bytes",
            )
            return False

        # Estimate post-resample size before resampling so oversized frames fail early.
        src_samples = len(data) // PCM_SAMPLE_WIDTH_BYTES
        target_samples = math.ceil(
            src_samples * self.model_sample_rate / self.config.input_sample_rate
        )
        if (
            self.asr_state.audio.received_bytes
            + target_samples * PCM_SAMPLE_WIDTH_BYTES
            > self.asr_processor.max_buffer_bytes
        ):
            # Close 1009 ("message too big") so clients can distinguish
            # session-resource exhaustion from a normal close.
            await self._send_error_and_close(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.asr_processor.max_buffer_seconds}s; "
                f"commit or clear before sending more audio",
                close_code=1009,
            )
            return True

        if self.config.input_sample_rate != self.model_sample_rate:
            data = await asyncio.to_thread(
                resample_to_target_rate,
                data,
                self.config.input_sample_rate,
                self.model_sample_rate,
            )
        self.asr_state.audio.append_pcm(data)
        if self.asr_processor.is_chunk_ready(self.asr_state):
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
        if not self.asr_state.has_audio and not self.asr_state.has_transcript:
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return

        has_new_audio = self.asr_state.has_new_audio
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

        # Capture PCM duration before `_start_next_item()` clears the absolute
        # byte counters for the next item.
        pcm_duration_seconds = (
            self.asr_state.audio.received_bytes
            / self.asr_processor.pcm_bytes_per_second
        )

        if has_new_audio:
            ok = await self._run_inference(is_last=True)
            if not ok:
                # _run_inference already emitted transcription.failed and
                # rolled the item; don't also emit completed.
                return
        elif self.asr_state.has_transcript:
            # Audio length was exactly a chunk_size_bytes multiple. Flush
            # the tail tokens update() held back.
            await self._emit_transcription_delta(
                self.asr_processor.flush_pending_transcript(self.asr_state)
            )

        # Transcript state may retain only the latest decode or pending suffix;
        # emitted deltas are the authoritative wire transcript.
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
        await self._send(
            InputAudioBufferClearedEvent(
                event_id=f"event_{random_uuid()}", type="input_audio_buffer.cleared"
            )
        )

    async def _run_inference(self, is_last: bool) -> bool:
        """Run ASR on the current buffer. Returns False on failure:
        commit-time emits transcription.failed and rolls the item; append-time
        emits a generic error envelope and closes the WebSocket."""
        try:
            delta = await self.asr_processor.process(
                self.asr_state,
                is_last=is_last,
                sampling_params=self.config.sampling_params,
            )
        except Exception:
            return await self._handle_inference_failure(is_last)
        await self._emit_transcription_delta(delta)
        return True

    async def _handle_inference_failure(self, is_last: bool) -> bool:
        """Send the correct failure signal for append-time vs commit-time errors."""
        logger.exception(
            "[realtime] inference failed: session=%s item=%s buffer_bytes=%d",
            self.session_id,
            self.item.current_item_id,
            len(self.asr_state.audio.data),
        )
        if is_last:
            # Already client-visible: transcription.failed can reference the item.
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
            # The item is not announced yet; close instead of referencing a ghost id.
            await self._send_error_and_close(
                "inference_failed",
                "Transcription failed",
                close_code=1011,
            )
        return False

    async def _emit_transcription_delta(self, delta: str) -> None:
        """emitted_deltas stores wire-formatted text (with leading
        boundary spaces baked in), so "".join(...) reconstructs the
        cumulative transcript verbatim."""
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
        self.asr_state = self.asr_processor.create_state()
        self.item.emitted_deltas.clear()

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
                        "turn_detection": None,
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
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            logger.debug(
                "[realtime] send error %s before close failed", code, exc_info=True
            )
        try:
            await self.websocket.close(code=close_code)
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            logger.debug(
                "[realtime] close %d after %s failed",
                close_code,
                code,
                exc_info=True,
            )
