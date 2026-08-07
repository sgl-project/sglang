"""Application service for realtime ASR inference.

The processor is connection-scoped but keeps no stream progress itself. Each call
builds and executes one stateless transcription step, then commits its outcome to
the explicit ``RealtimeASRState`` supplied by the realtime endpoint.

Per-chunk flow, driven by the endpoint::

    append audio -> is_chunk_ready()? -> process()
        _build_transcription_step  resolve mode, audio range, and prompt
        _execute_step              run the backend request, reconcile text
        _commit_outcome            advance cursors, compact old PCM
    -> transcript delta
    commit event -> process(is_last=True), or flush_pending_transcript()
                    if no new audio

Mode lifecycle: requests remain cumulative until the adapter's activation
gate is crossed. Eligible items then switch to encoder-aligned audio windows,
embedding-cache reuse, and a bounded decoder prefix for the rest of the item.
No-whitespace CJK remains cumulative.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Optional

import msgspec
import numpy as np

from sglang.srt.entrypoints.openai.realtime.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    AudioBuffer,
    pcm_to_float_samples,
)
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
    DecoderSuffixUpdate,
    cumulative_handoff_text,
    cumulative_is_suffix_compatible,
    has_no_word_boundaries,
    join_handoff_text,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    GeneratedTranscript,
    StreamingASRState,
    generate_asr_transcript,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RealtimeASRState(msgspec.Struct):
    """Mutable audio and transcript progress for the current input buffer."""

    audio: AudioBuffer
    transcript: StreamingASRState
    decoder_suffix: Optional[DecoderSuffixState] = None
    # No-whitespace CJK cannot safely use the word-based decoder-prefix path.
    encoder_window_disabled: bool = False

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        if self.decoder_suffix is not None:
            return bool(self.decoder_suffix.latest_text)
        return bool(self.transcript.full_transcript)

    @property
    def has_new_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.last_processed_offset_bytes

    @property
    def encoder_window_active(self) -> bool:
        return self.decoder_suffix is not None


class _TranscriptionStep(msgspec.Struct, frozen=True):
    """Resolved mode, audio range, and text context for one transcription."""

    is_last: bool
    uses_encoder_windows: bool
    start_offset_bytes: int
    end_offset_bytes: int
    decoder_prefix: str = ""
    handoff_text: str = ""


class _TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Reconciled text and whether the step's audio may be committed.

    ``audio_processed=False`` keeps the covered audio for the next step when
    the model cannot produce a usable continuation.
    """

    audio_processed: bool
    cumulative_delta: str = ""
    decoder_update: Optional[DecoderSuffixUpdate] = None

    @property
    def delta(self) -> str:
        if self.decoder_update is not None:
            return self.decoder_update.delta
        return self.cumulative_delta


class _EncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Encoder-window settings resolved from connection-invariant adapter
    configuration and model geometry."""

    encoder_window_config: AudioEncoderWindowConfig
    decoder_prefix_max_tokens: int
    decoder_prefix_holdback_words: int
    context_window_count: int
    # Below this many received bytes the endpoint keeps its cumulative path.
    activation_threshold_bytes: int

    @property
    def window_bytes(self) -> int:
        return self.encoder_window_config.window_samples * PCM_SAMPLE_WIDTH_BYTES


class RealtimeASRProcessor:
    """Build and execute realtime ASR steps against explicit stream state."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        server_args: ServerArgs,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.model_sample_rate = adapter.model_sample_rate
        self.pcm_bytes_per_second = self.model_sample_rate * PCM_SAMPLE_WIDTH_BYTES
        self.max_buffer_seconds = server_args.asr_max_buffer_seconds

        state = StreamingASRState(**self.adapter.chunked_streaming_config)
        self.chunk_size_bytes = int(state.chunk_size_sec * self.pcm_bytes_per_second)
        self.max_buffer_bytes = self.max_buffer_seconds * self.pcm_bytes_per_second
        self._encoder_window_policy = self._resolve_encoder_window_policy(
            state, server_args
        )

    def create_state(self) -> RealtimeASRState:
        return RealtimeASRState(
            audio=AudioBuffer(),
            transcript=StreamingASRState(**self.adapter.chunked_streaming_config),
        )

    def is_chunk_ready(self, state: RealtimeASRState) -> bool:
        """True once a full chunk of audio has arrived past the last attempt."""
        return (
            state.audio.received_bytes - state.audio.last_attempted_offset_bytes
            >= self.chunk_size_bytes
        )

    def _resolve_encoder_window_policy(
        self, state: StreamingASRState, server_args: ServerArgs
    ) -> Optional[_EncoderWindowPolicy]:
        """Resolve optional realtime encoder-window settings."""
        declared = self.adapter.realtime_encoder_window_config
        if not isinstance(declared, dict):
            return None
        if server_args.asr_long_audio_strategy != "encoder_window":
            return None
        if "max_audio_context_windows" not in declared:
            return None
        if (
            server_args.tp_size != 1
            or server_args.dp_size != 1
            or server_args.pp_size != 1
            or server_args.nnodes != 1
            or server_args.language_only
            or server_args.disaggregation_mode != "null"
        ):
            logger.warning(
                "[realtime] encoder windowing currently requires a local "
                "single-GPU runtime; using cumulative ASR"
            )
            return None
        mm_processor = self.tokenizer_manager.mm_processor
        if mm_processor is None or self.tokenizer_manager.tokenizer is None:
            return None
        try:
            encoder_window_config = mm_processor.resolve_audio_encoder_window_config(
                self.model_sample_rate
            )
            min_audio_sec = float(declared["min_audio_sec"])
            prefix_max_tokens = int(declared.get("decoder_prefix_max_tokens", 192))
            holdback_words = int(
                declared.get("decoder_prefix_holdback_words", state.unfixed_token_num)
            )
            context_window_count = int(declared["max_audio_context_windows"])
            if (
                encoder_window_config.window_samples <= 0
                or encoder_window_config.window_tokens <= 0
                or min_audio_sec < 0
                or not math.isfinite(min_audio_sec)
                or prefix_max_tokens <= 0
                or holdback_words < 0
                or context_window_count <= 0
            ):
                raise ValueError("encoder-window ASR values must be positive")
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning(
                "[realtime] invalid realtime_encoder_window_config; "
                "encoder windowing disabled",
                exc_info=True,
            )
            return None
        return _EncoderWindowPolicy(
            encoder_window_config=encoder_window_config,
            decoder_prefix_max_tokens=prefix_max_tokens,
            decoder_prefix_holdback_words=holdback_words,
            context_window_count=context_window_count,
            activation_threshold_bytes=(
                math.ceil(min_audio_sec / state.chunk_size_sec) * self.chunk_size_bytes
            ),
        )

    async def process(
        self,
        state: RealtimeASRState,
        *,
        is_last: bool,
        sampling_params: Dict[str, Any],
    ) -> str:
        """Run one transcription step and return its publishable delta.

        The first encoder-window update is computed without mutation so
        unsupported no-whitespace CJK can retry cumulatively."""
        step = self._build_transcription_step(state, is_last)
        if (
            step.uses_encoder_windows
            and not state.encoder_window_active
            and not cumulative_is_suffix_compatible(state.transcript)
        ):
            state.encoder_window_disabled = True
            step = self._build_transcription_step(state, is_last)

        outcome = await self._execute_step(state, step, sampling_params)

        if self._requires_cumulative_retry(state, step, outcome):
            # Nothing was emitted yet; redo this request on the cumulative path.
            state.encoder_window_disabled = True
            step = self._build_transcription_step(state, is_last)
            outcome = await self._execute_step(state, step, sampling_params)

        self._commit_outcome(state, step, outcome)
        return outcome.delta

    def flush_pending_transcript(self, state: RealtimeASRState) -> str:
        """Emit text still held back when the item commits without new audio."""
        if state.decoder_suffix is not None:
            return state.decoder_suffix.flush()
        return state.transcript.finalize()

    def _build_transcription_step(
        self,
        state: RealtimeASRState,
        is_last: bool,
    ) -> _TranscriptionStep:
        """Resolve the mode, audio range, and text context for the next call."""
        transcript = state.transcript
        audio = state.audio
        policy = self._encoder_window_policy
        end_offset_bytes = (
            audio.received_bytes
            if is_last
            else min(
                audio.received_bytes,
                audio.last_attempted_offset_bytes + self.chunk_size_bytes,
            )
        )
        if (
            policy is not None
            and not state.encoder_window_disabled
            and (
                state.encoder_window_active
                or (
                    not is_last and end_offset_bytes > policy.activation_threshold_bytes
                )
            )
        ):
            if state.decoder_suffix is None:
                handoff_text = cumulative_handoff_text(transcript) or ""
                prefix_source = join_handoff_text(transcript.emitted_text, handoff_text)
                decoder_prefix = DecoderSuffixState(
                    emitted_text=prefix_source
                ).get_bounded_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.decoder_prefix_max_tokens,
                )
            else:
                handoff_text = ""
                decoder_prefix = state.decoder_suffix.get_bounded_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.decoder_prefix_max_tokens,
                )
            return _TranscriptionStep(
                is_last=is_last,
                uses_encoder_windows=True,
                start_offset_bytes=self._encoder_window_start_offset(
                    state, end_offset_bytes
                ),
                end_offset_bytes=end_offset_bytes,
                decoder_prefix=decoder_prefix,
                handoff_text=handoff_text,
            )
        return _TranscriptionStep(
            is_last=is_last,
            uses_encoder_windows=False,
            start_offset_bytes=0,
            end_offset_bytes=end_offset_bytes,
            decoder_prefix=transcript.get_prefix_text(),
        )

    def _encoder_window_start_offset(
        self, state: RealtimeASRState, end_offset_bytes: int
    ) -> int:
        """Pick a window-aligned start so resent windows keep their cache
        identity; never advances past deferred audio."""
        policy = self._encoder_window_policy
        assert policy is not None
        audio = state.audio

        # Establish suffix state against the cumulative audio once. Later
        # requests can drop encoder-aligned history represented by the prefix.
        if not state.encoder_window_active:
            return 0
        window_bytes = policy.window_bytes
        complete_end = end_offset_bytes // window_bytes * window_bytes
        # Deferred audio must remain inside the next rolling request even when
        # the nominal context horizon advances.
        start = min(
            complete_end - policy.context_window_count * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        return max(audio.base_offset_bytes, start)

    async def _execute_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        if step.uses_encoder_windows:
            return await self._execute_encoder_window_step(state, step, sampling_params)
        return await self._execute_cumulative_step(state, step, sampling_params)

    async def _execute_cumulative_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        """Re-transcribe accumulated audio and reconcile its cumulative text."""
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )
        generation = await self._generate_transcript(
            samples,
            sampling_params,
            self.adapter.prompt_template + step.decoder_prefix,
        )
        if generation.finish_reason == "length":
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
        if step.is_last:
            state.transcript.full_transcript = generation.text
            delta = state.transcript.finalize()
        else:
            delta = state.transcript.update(generation.text)
        return _TranscriptionOutcome(
            audio_processed=True,
            cumulative_delta=delta,
        )

    async def _execute_encoder_window_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        """Generate continuation text from encoder-aligned rolling context."""
        policy = self._encoder_window_policy
        assert policy is not None
        if step.start_offset_bytes % policy.window_bytes:
            raise RuntimeError("encoder-window request is not window aligned")
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )

        generation = await self._generate_transcript(
            samples,
            sampling_params,
            self.adapter.prompt_template + step.decoder_prefix,
            encoder_window_config=policy.encoder_window_config,
        )
        if generation.finish_reason == "length":
            raise RuntimeError("realtime ASR decode reached max_new_tokens")

        suffix_state = state.decoder_suffix or DecoderSuffixState(
            emitted_text=state.transcript.emitted_text
        )
        text = suffix_state.trim_prefix_echo(
            generation.text,
            step.decoder_prefix,
            trim_short_prefix=not state.encoder_window_active,
            minimum_prefix_words=max(24, int(state.transcript.chunk_size_sec * 16)),
        )
        if (
            not text
            and step.end_offset_bytes > state.audio.last_processed_offset_bytes
            and not step.is_last
            and state.audio.last_attempted_offset_bytes
            == state.audio.last_processed_offset_bytes
        ):
            # Retry one later request before accepting an empty continuation.
            return _TranscriptionOutcome(audio_processed=False)

        if not state.encoder_window_active:
            # The first continuation may replay only the un-emitted cumulative
            # tail rather than the full decoder prefix. Prepend that tail once.
            text = suffix_state.trim_prefix_echo(
                text,
                step.handoff_text,
                trim_short_prefix=True,
            )
            text = join_handoff_text(step.handoff_text, text)

        update = suffix_state.reconcile(
            text,
            is_last=step.is_last,
            holdback_words=policy.decoder_prefix_holdback_words,
        )
        return _TranscriptionOutcome(
            audio_processed=True,
            decoder_update=update,
        )

    async def _snapshot_samples(
        self, audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
    ) -> np.ndarray:
        pcm = audio.snapshot(start_offset_bytes, end_offset_bytes)
        return await asyncio.to_thread(pcm_to_float_samples, pcm)

    async def _generate_transcript(
        self,
        samples: np.ndarray,
        sampling_params: Dict[str, Any],
        prompt: str,
        *,
        encoder_window_config: Optional[AudioEncoderWindowConfig] = None,
    ) -> GeneratedTranscript:
        mm_processor_kwargs = (
            {"audio_encoder_window_config": encoder_window_config}
            if encoder_window_config is not None
            else None
        )
        result = await generate_asr_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            audio_data=samples,
            sampling_params=sampling_params,
            prompt=prompt,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if result is None:
            raise RuntimeError("realtime ASR request returned no response")
        return result

    def _commit_outcome(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        outcome: _TranscriptionOutcome,
    ) -> None:
        """Advance attempted audio unconditionally and processed audio only
        after its decode enters transcript state."""
        audio = state.audio
        audio.last_attempted_offset_bytes = step.end_offset_bytes
        if (
            step.uses_encoder_windows
            and outcome.audio_processed
            and state.decoder_suffix is None
        ):
            state.decoder_suffix = DecoderSuffixState(
                emitted_text=state.transcript.emitted_text
            )
        if (
            outcome.decoder_update is not None
            and outcome.decoder_update.pending_suffix is not None
        ):
            assert state.decoder_suffix is not None
            state.decoder_suffix.apply(
                outcome.decoder_update,
                is_last=step.is_last,
            )
        if not outcome.audio_processed:
            return
        audio.last_processed_offset_bytes = step.end_offset_bytes
        if step.is_last:
            return
        if step.uses_encoder_windows:
            audio.discard_before(step.start_offset_bytes)

    def _requires_cumulative_retry(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        outcome: _TranscriptionOutcome,
    ) -> bool:
        # Decoder-prefix reconciliation is word based; the first encoder-window
        # decode may reveal a no-whitespace transcript that cannot use it.
        return (
            step.uses_encoder_windows
            and not state.encoder_window_active
            and outcome.decoder_update is not None
            and has_no_word_boundaries(outcome.decoder_update.pending_suffix or "")
        )
