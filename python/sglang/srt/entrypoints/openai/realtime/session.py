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
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
from sglang.srt.entrypoints.openai.realtime.protocol import (
    DEFAULT_INPUT_SAMPLE_RATE,
    SUPPORTED_INPUT_SAMPLE_RATES,
    AudioPCM,
    SessionUpdateEvent,
    TranscriptionSessionAudioInput,
    TranscriptionSessionConfig,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
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


# PCM16: 16-bit samples -> 2 bytes each.
PCM_SAMPLE_WIDTH = 2


def slice_pcm_range(buffer: Union[bytes, bytearray], start: int, end: int) -> bytes:
    """Snapshot PCM bytes for an ASR request.

    The resident buffer is a mutable rolling ``bytearray``. Return immutable
    bytes so later append/compaction cannot change the in-flight audio, and
    fail fast if absolute-to-local offset bookkeeping produced an invalid range.
    """
    if not (0 <= start <= end <= len(buffer)):
        raise ValueError(
            f"slice_pcm_range: range=[{start}, {end}) not in [0, {len(buffer)}]"
        )
    return bytes(memoryview(buffer)[start:end])


def resample_to_target_rate(pcm: bytes, src_rate: int, target_rate: int) -> bytes:
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


def pcm_to_float_samples(pcm: bytes) -> np.ndarray:
    # /32768.0 matches soundfile.read's default int16 normalization so the
    # samples are bit-equal to the prior PCM->WAV->sf.read path.
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


CLIENT_EVENT_TYPES: Dict[str, type] = {
    "session.update": SessionUpdateEvent,
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "input_audio_buffer.commit": InputAudioBufferCommitEvent,
    "input_audio_buffer.clear": InputAudioBufferClearEvent,
}


def parse_client_event(raw: Dict[str, Any]) -> Optional[BaseModel]:
    """Parse known client events, returning None for unknown event types."""
    cls = CLIENT_EVENT_TYPES.get(raw.get("type"))
    if cls is None:
        return None
    return cls.model_validate(raw)


@dataclass
class SessionConfig:
    """Session-level configuration negotiated via session.update."""

    input_sample_rate: int = DEFAULT_INPUT_SAMPLE_RATE
    language: Optional[str] = None
    client_model: Optional[str] = None
    sampling_params: Optional[Dict[str, Any]] = None
    configured: bool = False


@dataclass
class AudioState:
    """Per-item PCM buffer plus absolute byte offsets.

    Realtime ASR can keep one item open for many chunks. After slicing starts,
    old PCM should stop growing resident memory, but inference anchors still
    need to be expressed in the item's original byte timeline.
    ``pcm_buffer`` stores only the resident bytes still needed for inference.
    The offset fields stay absolute within the current item, so compaction can
    drop old bytes without moving inference anchors backward.
    """

    max_buffer_bytes: int
    chunk_size_bytes: int
    left_overlap_bytes: int
    slicing_min_chunk_index: int
    state: StreamingASRState
    # Disabled when overlap leaves no fresh audio window for text dedupe.
    slicing_enabled: bool = True
    pcm_buffer: bytearray = field(default_factory=bytearray)
    pcm_buffer_base_offset_bytes: int = 0
    total_pcm_bytes_received: int = 0
    last_inference_offset_bytes: int = 0
    last_sliced_buffer_end_bytes: int = 0

    def append_pcm(self, pcm: bytes) -> None:
        """Append PCM while keeping total received bytes absolute."""
        self.pcm_buffer.extend(pcm)
        self.total_pcm_bytes_received += len(pcm)

    def compact_after_sliced_inference(self) -> None:
        """Bound resident PCM while preserving the next slice's context."""
        keep_bytes = self.left_overlap_bytes + self.chunk_size_bytes
        keep_start = max(0, self.last_sliced_buffer_end_bytes - keep_bytes)
        drop_bytes = keep_start - self.pcm_buffer_base_offset_bytes
        drop_bytes -= drop_bytes % PCM_SAMPLE_WIDTH
        if drop_bytes <= 0:
            return

        del self.pcm_buffer[:drop_bytes]
        self.pcm_buffer_base_offset_bytes += drop_bytes

    def reset_pcm_offsets(self) -> None:
        """Clear item-local audio and reset all absolute byte cursors."""
        self.pcm_buffer.clear()
        self.pcm_buffer_base_offset_bytes = 0
        self.total_pcm_bytes_received = 0
        self.last_inference_offset_bytes = 0
        self.last_sliced_buffer_end_bytes = 0

    def global_to_local(self, global_offset: int) -> int:
        """Translate item-absolute offsets after old PCM has been compacted."""
        return global_offset - self.pcm_buffer_base_offset_bytes


@dataclass
class ItemState:
    """Current conversation item ids and already emitted transcript deltas."""

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
        self.bytes_per_second = self.model_sample_rate * PCM_SAMPLE_WIDTH
        self.max_buffer_seconds = server_args.asr_max_buffer_seconds

        self.config = SessionConfig()

        slicing_cfg = adapter.realtime_slicing_config
        slicing_opt_in = bool(slicing_cfg.get("enabled", False)) and not getattr(
            server_args, "asr_disable_input_slicing", False
        )
        left_overlap_ms = int(slicing_cfg.get("left_overlap_ms", 0))
        min_audio_sec = float(slicing_cfg.get("min_audio_sec", 0.0))

        state = StreamingASRState(**adapter.chunked_streaming_config)
        chunk_size_bytes = int(state.chunk_size_sec * self.bytes_per_second)
        if chunk_size_bytes <= 0:
            raise RuntimeError(
                f"adapter.chunked_streaming_config produced non-positive "
                f"chunk_size_sec; got {state.chunk_size_sec!r}"
            )
        if state.unfixed_chunk_num < 0 or state.unfixed_token_num < 0:
            raise RuntimeError(
                f"adapter.chunked_streaming_config produced negative holdback "
                f"values; got unfixed_chunk_num={state.unfixed_chunk_num!r}, "
                f"unfixed_token_num={state.unfixed_token_num!r}"
            )

        invalid_slicing_fields = []
        if left_overlap_ms < 0:
            invalid_slicing_fields.append(f"left_overlap_ms={left_overlap_ms!r}")
        if min_audio_sec < 0:
            invalid_slicing_fields.append(f"min_audio_sec={min_audio_sec!r}")
        if slicing_opt_in and invalid_slicing_fields:
            logger.warning(
                "[realtime] invalid realtime_slicing_config (%s); "
                "audio slicing disabled, falling back to cumulative inference",
                ", ".join(invalid_slicing_fields),
            )
            slicing_opt_in = False
        left_overlap_ms = max(left_overlap_ms, 0)
        min_audio_sec = max(min_audio_sec, 0.0)

        left_overlap_bytes = int(left_overlap_ms / 1000 * self.bytes_per_second)
        # Snap to a whole int16 frame: odd model_sample_rate*left_overlap_ms
        # (e.g. 44.1kHz) yields an odd byte count, and an odd slice start splits
        # a PCM sample, which raises in pcm_to_float_samples (np.int16 view).
        left_overlap_bytes -= left_overlap_bytes % PCM_SAMPLE_WIDTH
        slicing_min_chunk_index = (
            math.ceil(min_audio_sec / state.chunk_size_sec) if slicing_opt_in else 0
        )
        slicing_enabled = (
            slicing_opt_in
            and left_overlap_bytes < state.unfixed_chunk_num * chunk_size_bytes
        )
        if slicing_opt_in and not slicing_enabled:
            logger.warning(
                "[realtime] left_overlap=%dms >= unfixed_chunks_duration=%dms; "
                "audio slicing disabled, falling back to cumulative inference",
                left_overlap_ms,
                state.unfixed_chunk_num * int(state.chunk_size_sec * 1000),
            )
        self.audio = AudioState(
            max_buffer_bytes=self.max_buffer_seconds * self.bytes_per_second,
            chunk_size_bytes=chunk_size_bytes,
            state=state,
            left_overlap_bytes=left_overlap_bytes,
            slicing_min_chunk_index=slicing_min_chunk_index,
            slicing_enabled=slicing_enabled,
        )

        self.item = ItemState(current_item_id=f"item_{random_uuid()}")

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
                event = parse_client_event(raw)
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
        if audio.turn_detection is not None:
            await self._send_error(
                "not_supported",
                "Server-side VAD is not implemented; "
                "set audio.input.turn_detection: null and commit explicitly.",
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
            if (
                new_rate != self.config.input_sample_rate
                and self.audio.total_pcm_bytes_received > 0
            ):
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

        if len(data) % PCM_SAMPLE_WIDTH != 0:
            await self._send_error(
                "invalid_audio_format",
                f"PCM16 frame length must be a multiple of {PCM_SAMPLE_WIDTH} bytes",
            )
            return False

        # Estimate post-resample size before resampling so oversized frames fail early.
        src_samples = len(data) // PCM_SAMPLE_WIDTH
        target_samples = math.ceil(
            src_samples * self.model_sample_rate / self.config.input_sample_rate
        )
        if (
            self.audio.total_pcm_bytes_received + target_samples * PCM_SAMPLE_WIDTH
            > self.audio.max_buffer_bytes
        ):
            # Close 1009 ("message too big") so clients can distinguish
            # session-resource exhaustion from a normal close.
            await self._send_error_and_close(
                "buffer_overflow",
                f"Accumulated audio exceeded {self.max_buffer_seconds}s; "
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
        self.audio.append_pcm(data)

        new_audio_bytes = (
            self.audio.total_pcm_bytes_received - self.audio.last_inference_offset_bytes
        )
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
        if (
            self.audio.total_pcm_bytes_received == 0
            and not self.audio.state.full_transcript
        ):
            await self._send_error(
                "invalid_state", "Cannot commit an empty audio buffer"
            )
            return

        has_new_audio = (
            self.audio.total_pcm_bytes_received > self.audio.last_inference_offset_bytes
        )
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
            self.audio.total_pcm_bytes_received / self.bytes_per_second
        )

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

        # Rebuild from emitted_deltas: both paths leave full_transcript only a
        # partial tail, while the deltas together are the whole transcript.
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
        """Run ASR on the current audio window.

        Short/unstable items keep the cumulative path for compatibility. Longer
        items switch to a tail slice so each chunk sends bounded audio instead
        of re-encoding the full item. The left overlap preserves acoustic
        context, and output dedupe removes text repeated from that overlap.
        Returns False on failure -- commit-time emits
        transcription.failed and rolls the item; append-time closes the WS."""
        # Slicing uses a bare prompt: the retained overlap + dedupe replace
        # injecting emitted_text as a continuation prefix.
        committed_text = self.audio.state.get_prefix_text()
        use_slicing = (
            self.audio.slicing_enabled
            and bool(committed_text)
            and self.audio.state.chunk_index >= self.audio.slicing_min_chunk_index
        )
        if use_slicing:
            prompt: Optional[str] = self.adapter.prompt_template
            dedupe_against: Optional[str] = committed_text
            slice_start_global = max(
                0,
                self.audio.last_sliced_buffer_end_bytes - self.audio.left_overlap_bytes,
            )
        else:
            prompt = None
            dedupe_against = None
            slice_start_global = 0

        slice_end_global = self.audio.total_pcm_bytes_received
        # Clamp to the retained buffer. A within-item sliced->cumulative flip
        # (e.g. emitted_text becomes CJK-no-whitespace, so get_prefix_text()
        # returns "" and slice_start_global drops to 0) would otherwise index
        # before the compacted base (negative local start) and crash the live
        # session via slice_pcm_range. Use the retained start because earlier
        # bytes are no longer resident.
        slice_start = max(0, self.audio.global_to_local(slice_start_global))
        slice_end = self.audio.global_to_local(slice_end_global)

        try:
            pcm_slice = slice_pcm_range(self.audio.pcm_buffer, slice_start, slice_end)
            audio_samples = await asyncio.to_thread(pcm_to_float_samples, pcm_slice)
            delta = await process_asr_chunk(
                tokenizer_manager=self.tokenizer_manager,
                adapter=self.adapter,
                state=self.audio.state,
                audio_data=audio_samples,
                sampling_params=self.config.sampling_params,
                is_last=is_last,
                prompt=prompt,
                dedupe_against=dedupe_against,
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

        await self._emit_transcription_delta(delta)
        # Held-back tokens are re-covered only if their audio span fits the
        # left overlap; slower speech can drop the earliest (see known limits).
        self.audio.last_sliced_buffer_end_bytes = slice_end_global

        self.audio.last_inference_offset_bytes = slice_end_global
        if use_slicing and not is_last:
            self.audio.compact_after_sliced_inference()
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
        self.audio.state = StreamingASRState(**self.adapter.chunked_streaming_config)
        self.audio.reset_pcm_offsets()
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
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] send error %s before close failed: %s", code, e)
        try:
            await self.websocket.close(code=close_code)
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.debug("[realtime] close %d after %s failed: %s", close_code, code, e)
