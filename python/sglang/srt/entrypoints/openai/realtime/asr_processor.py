"""Application service for realtime ASR inference.

The processor is connection-scoped but keeps no stream progress itself. Each call
builds and executes one backend transcription step, then commits its audio outcome
to the explicit ``RealtimeASRState`` supplied by the realtime endpoint.

Per-chunk flow, driven by the endpoint::

    append audio -> is_chunk_ready()? -> process()
        _build_transcription_step  resolve mode, audio range, and prompt
        _execute_step              run the backend request, reconcile text
                                   (cumulative reconciliation mutates its
                                   state; windowed reconciliation is preview-only)
        _commit_outcome            advance cursors, compact old PCM
    -> transcript delta
    commit event -> process(is_last=True), or flush_pending_transcript()
                    if no new audio

Mode lifecycle: requests remain cumulative until the adapter's activation
gate is crossed. Eligible items then switch to encoder-aligned audio windows,
embedding-cache reuse, and a bounded decoder prefix for the rest of the item.
Only languages explicitly declared by the adapter may enter this mode.
"""

from __future__ import annotations

import asyncio
import logging
import math
from copy import copy
from typing import Any, Awaitable, Callable, Dict, Optional

import msgspec
import numpy as np

from sglang.srt.entrypoints.openai.realtime.audio_buffer import (
    PCM_SAMPLE_WIDTH_BYTES,
    AudioBuffer,
    pcm_to_float_samples,
)
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    MIN_PREFIX_ECHO_WORDS,
    DecoderSuffixState,
    DecoderSuffixUpdate,
    could_be_decoder_prefix_replay,
    cumulative_handoff_text,
    cumulative_is_suffix_compatible,
    has_no_word_boundaries,
    join_text,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    GeneratedTranscript,
    StreamingASRState,
    apply_cumulative_transcript,
    generate_asr_transcript,
    generate_cumulative_transcript,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MAX_PREFIX_ECHO_WORDS_PER_SECOND = 16

_TranscriptDeltaCallback = Callable[[str], Awaitable[None]]


class RealtimeASRState(msgspec.Struct):
    """Mutable per-item progress shared by the endpoint and processor.

    ``transcript`` owns cumulative reconciliation until a successful one-way
    handoff. ``decoder_suffix`` is created only when that handoff commits, so
    its presence is also the source of truth that encoder windowing is active.
    Together with ``encoder_window_disabled``, this represents eligible,
    active, and cumulative-only items without a separate mode field that could
    drift out of sync.
    """

    audio: AudioBuffer
    transcript: StreamingASRState
    decoder_suffix: Optional[DecoderSuffixState] = None
    # Once text proves incompatible with word-based suffix reconciliation, this
    # item stays cumulative because switching back after compaction is unsafe.
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
    """Immutable backend-request snapshot built before generation awaits.

    Execution and commit consume the same instance so the selected mode, PCM
    range, and windowed decoder context cannot be recomputed from state changed
    while the request was in flight.
    """

    is_last: bool
    uses_encoder_windows: bool
    start_offset_bytes: int
    end_offset_bytes: int
    decoder_prefix: str = ""
    handoff_text: str = ""


class _TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Result contract consumed when a transcription step commits.

    Windowed decoder updates remain unapplied so the first handoff can be
    discarded and retried cumulatively if its text is unsafe to reconcile.
    The cumulative path keeps its existing in-place reconciliation behavior.
    In both paths, ``audio_processed=False`` retains the covered audio for the
    next step instead of advancing the processed cursor.
    """

    audio_processed: bool
    cumulative_delta: str = ""
    decoder_update: Optional[DecoderSuffixUpdate] = None
    # Judged on the raw post-echo continuation, before the cumulative handoff
    # is prepended: a whitespace-bearing handoff must not mask a no-boundary
    # model continuation, or the one-way windowed transition would commit a
    # mode whose word-based reconciliation cannot handle the text.
    continuation_without_word_boundaries: bool = False

    @property
    def delta(self) -> str:
        if self.decoder_update is not None:
            return self.decoder_update.delta
        return self.cumulative_delta


class _ResolvedEncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Adapter policy paired with model-resolved encoder geometry."""

    geometry: AudioEncoderWindowConfig
    adapter_policy: RealtimeEncoderWindowPolicy
    # Below this many received bytes the endpoint keeps its cumulative path.
    activation_threshold_bytes: int

    @property
    def window_bytes(self) -> int:
        return self.geometry.window_samples * PCM_SAMPLE_WIDTH_BYTES

    def supports_language(self, language: Optional[str]) -> bool:
        return self.adapter_policy.supports_language(language)


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

        state = StreamingASRState(**self.adapter.chunked_streaming_config)
        if not math.isfinite(state.chunk_size_sec) or state.chunk_size_sec <= 0:
            raise ValueError("realtime ASR chunk_size_sec must be finite and positive")
        self.chunk_size_bytes = int(state.chunk_size_sec * self.pcm_bytes_per_second)
        if self.chunk_size_bytes <= 0:
            raise ValueError(
                "realtime ASR chunk_size_sec is shorter than one PCM sample"
            )
        if self.chunk_size_bytes % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError(
                "realtime ASR chunk_size_sec must resolve to whole PCM samples"
            )
        self.max_buffer_bytes = (
            server_args.asr_max_buffer_seconds * self.pcm_bytes_per_second
        )
        self._encoder_window_policy = self._resolve_encoder_window_policy(
            state, enabled=server_args.enable_asr_encoder_window
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
        self, state: StreamingASRState, *, enabled: bool
    ) -> Optional[_ResolvedEncoderWindowPolicy]:
        """Resolve optional realtime encoder-window settings."""
        declared = self.adapter.realtime_encoder_window_policy
        if declared is None:
            return None
        if not isinstance(declared, RealtimeEncoderWindowPolicy):
            raise TypeError(
                "realtime_encoder_window_policy must return "
                "RealtimeEncoderWindowPolicy or None"
            )
        if not enabled:
            return None
        # Activate on a fixed inference boundary so large appends and normal
        # chunked input make the same cumulative-to-windowed transition.
        activation_threshold_bytes = (
            math.ceil(declared.min_audio_sec / state.chunk_size_sec)
            * self.chunk_size_bytes
        )
        mm_processor = self.tokenizer_manager.mm_processor
        if mm_processor is None or self.tokenizer_manager.tokenizer is None:
            raise RuntimeError(
                "encoder-window ASR requires a tokenizer and multimodal processor"
            )
        encoder_window_config = mm_processor.resolve_audio_encoder_window_config(
            self.model_sample_rate
        )
        if (
            encoder_window_config.window_samples <= 0
            or encoder_window_config.window_tokens <= 0
        ):
            raise ValueError("invalid audio encoder-window geometry")
        if self.max_buffer_bytes <= activation_threshold_bytes:
            logger.warning(
                "[realtime] encoder windowing cannot activate before the "
                "current ASR item limit; raise --asr-max-buffer-seconds"
            )
            return None
        return _ResolvedEncoderWindowPolicy(
            geometry=encoder_window_config,
            adapter_policy=declared,
            activation_threshold_bytes=activation_threshold_bytes,
        )

    def max_item_bytes(
        self, language: Optional[str], *, encoder_window_disabled: bool = False
    ) -> int:
        """Choose the item cap from the language fixed before audio arrives;
        a runtime no-boundary fallback narrows it back to the cumulative cap."""
        policy = self._encoder_window_policy
        if policy is None:
            return self.max_buffer_bytes
        if encoder_window_disabled or not policy.supports_language(language):
            return min(self.max_buffer_bytes, policy.activation_threshold_bytes)
        return self.max_buffer_bytes

    async def process(
        self,
        state: RealtimeASRState,
        *,
        is_last: bool,
        language: Optional[str],
        sampling_params: Dict[str, Any],
        on_transcript_delta: Optional[_TranscriptDeltaCallback] = None,
    ) -> str:
        """Run one transcription step and return its publishable delta.

        The adapter language gate is checked before the one-way transition. The
        first windowed result is still computed without mutation so unexpected
        no-whitespace output can retry cumulatively."""
        policy = self._encoder_window_policy
        if (
            state.encoder_window_active
            and policy is not None
            and not policy.supports_language(language)
        ):
            raise RuntimeError(
                "realtime ASR language cannot change after encoder windowing starts"
            )
        step = self._build_transcription_step(state, is_last, language)
        if (
            step.uses_encoder_windows
            and not state.encoder_window_active
            and not cumulative_is_suffix_compatible(state.transcript)
        ):
            state.encoder_window_disabled = True
            step = self._build_transcription_step(state, is_last, language)

        streamed_delta = ""

        async def publish_delta(candidate: str) -> None:
            nonlocal streamed_delta
            candidate = candidate.strip()
            if not candidate or candidate == streamed_delta:
                return
            # Decoder output grows monotonically within one request. If
            # reconciliation revises an already-published prefix, do not append
            # conflicting text; the completed request will fail closed below.
            if not candidate.startswith(streamed_delta):
                return
            delta = candidate[len(streamed_delta) :].strip()
            if not delta:
                return
            assert on_transcript_delta is not None
            await on_transcript_delta(delta)
            streamed_delta = candidate

        stream_callback = publish_delta if on_transcript_delta is not None else None
        outcome = await self._execute_step(
            state, step, sampling_params, stream_callback
        )

        if self._requires_cumulative_retry(state, step, outcome):
            # Nothing was emitted yet; redo this request on the cumulative path.
            state.encoder_window_disabled = True
            step = self._build_transcription_step(state, is_last, language)
            outcome = await self._execute_step(
                state, step, sampling_params, stream_callback
            )

        remaining_delta = outcome.delta
        if streamed_delta:
            if not remaining_delta.startswith(streamed_delta):
                raise RuntimeError(
                    "realtime ASR revised an append-only streamed prefix"
                )
            remaining_delta = remaining_delta[len(streamed_delta) :].strip()
        self._commit_outcome(state, step, outcome)
        return remaining_delta

    def flush_pending_transcript(self, state: RealtimeASRState) -> str:
        """Emit text still held back when the item commits without new audio."""
        if state.decoder_suffix is not None:
            return state.decoder_suffix.flush()
        return state.transcript.finalize()

    def _build_transcription_step(
        self,
        state: RealtimeASRState,
        is_last: bool,
        language: Optional[str],
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
        if self._should_use_encoder_windows(
            state,
            is_last=is_last,
            language=language,
            end_offset_bytes=end_offset_bytes,
        ):
            assert policy is not None
            if state.decoder_suffix is None:
                handoff_text = cumulative_handoff_text(transcript) or ""
                prefix_source = join_text(transcript.emitted_text, handoff_text)
                decoder_prefix = DecoderSuffixState(
                    emitted_text=prefix_source
                ).get_bounded_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.adapter_policy.decoder_prefix_max_tokens,
                )
            else:
                handoff_text = ""
                decoder_prefix = state.decoder_suffix.get_bounded_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.adapter_policy.decoder_prefix_max_tokens,
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
        )

    def _should_use_encoder_windows(
        self,
        state: RealtimeASRState,
        *,
        is_last: bool,
        language: Optional[str],
        end_offset_bytes: int,
    ) -> bool:
        policy = self._encoder_window_policy
        if policy is None or state.encoder_window_disabled:
            return False
        if state.encoder_window_active:
            return True
        return (
            not is_last
            and policy.supports_language(language)
            and end_offset_bytes > policy.activation_threshold_bytes
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
            complete_end
            - policy.adapter_policy.max_audio_context_windows * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        return max(audio.base_offset_bytes, start)

    async def _execute_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        """Execute one step.

        Cumulative reconciliation preserves the existing in-place state machine.
        Windowed reconciliation returns an unapplied update so its first result
        can be discarded and retried cumulatively when text boundaries are unsafe.
        """
        if step.uses_encoder_windows:
            return await self._execute_encoder_window_step(
                state, step, sampling_params, on_transcript_delta
            )
        return await self._execute_cumulative_step(
            state, step, sampling_params, on_transcript_delta
        )

    async def _execute_cumulative_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        """Re-transcribe accumulated audio and reconcile its cumulative text."""
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )

        async def publish_delta(text: str) -> None:
            assert on_transcript_delta is not None
            preview_state = copy(state.transcript)
            delta = apply_cumulative_transcript(preview_state, text, is_last=False)
            await on_transcript_delta(delta)

        generation = await generate_cumulative_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            state=state.transcript,
            audio_data=samples,
            sampling_params=sampling_params,
            on_update=(
                publish_delta
                if on_transcript_delta is not None and not step.is_last
                else None
            ),
        )
        if generation is None:
            if step.is_last:
                # Completing without the final decode would silently drop the
                # last audio; fail the item instead of publishing it.
                raise RuntimeError("final realtime ASR request returned no response")
            # An empty backend iterator must not commit uncovered audio. Only
            # the attempted cursor advances, so the next step or forced final
            # decode re-covers the audio from offset zero. Explicit aborts raise
            # before this branch and fail the request.
            logger.warning("[realtime] cumulative ASR step returned no response")
            return _TranscriptionOutcome(audio_processed=False)
        recovering_unprocessed_audio = (
            state.audio.last_attempted_offset_bytes
            > state.audio.last_processed_offset_bytes
        )
        if recovering_unprocessed_audio and not generation.text:
            if step.is_last:
                raise RuntimeError("final realtime ASR recovery returned empty text")
            return _TranscriptionOutcome(audio_processed=False)
        if step.is_last and generation.finish_reason == "length":
            # Never publish a known-truncated transcript as completed.
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
        delta = apply_cumulative_transcript(
            state.transcript, generation.text, is_last=step.is_last
        )
        # Keep truncated audio unprocessed so the cursor state forces a clean
        # final decode at commit. The cumulative transcript update above still
        # preserves the existing intermediate streaming behavior.
        return _TranscriptionOutcome(
            audio_processed=generation.finish_reason != "length",
            cumulative_delta=delta,
        )

    async def _execute_encoder_window_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
        on_transcript_delta: Optional[_TranscriptDeltaCallback],
    ) -> _TranscriptionOutcome:
        """Generate continuation text from encoder-aligned rolling context."""
        policy = self._encoder_window_policy
        assert policy is not None
        if step.start_offset_bytes % policy.window_bytes:
            raise RuntimeError("encoder-window request is not window aligned")
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )

        async def publish_delta(text: str) -> None:
            assert on_transcript_delta is not None
            if not text:
                return
            if could_be_decoder_prefix_replay(text, step.decoder_prefix):
                return
            # A decoder snapshot can end in the middle of a word. Keep that
            # trailing word provisional so a later token cannot revise text
            # that has already been published through the append-only API.
            words = text.split()
            if not text[-1].isspace():
                words = words[:-1]
            if not words:
                return
            text = " ".join(words)
            try:
                preview = self._reconcile_encoder_window_text(state, step, text)
            except RuntimeError:
                # A partial token sequence can temporarily be impossible to
                # reconcile; the completed generation still fails closed.
                return
            await on_transcript_delta(preview.delta)

        generation = await self._generate_windowed_transcript(
            samples=samples,
            sampling_params=sampling_params,
            prompt=self.adapter.prompt_template + step.decoder_prefix,
            encoder_window_config=policy.geometry,
            on_update=(
                publish_delta
                if on_transcript_delta is not None
                and not step.is_last
                and state.encoder_window_active
                else None
            ),
        )

        return self._reconcile_encoder_window_text(state, step, generation.text)

    def _reconcile_encoder_window_text(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        text: str,
    ) -> _TranscriptionOutcome:
        """Reconcile one complete or provisional windowed decoder snapshot."""
        policy = self._encoder_window_policy
        assert policy is not None

        suffix_state = state.decoder_suffix or DecoderSuffixState(
            emitted_text=state.transcript.emitted_text
        )
        # Echo evidence is applied from strongest to weakest: a long decoder
        # prefix replay, then a short final replay that reconnects pending text,
        # then the first-window handoff. Reordering these trims can delete a
        # phrase the speaker genuinely repeated.
        text, decoder_prefix_replayed = suffix_state.trim_prefix_echo(
            text,
            step.decoder_prefix,
            # Without pending text there is no anchor to confirm a short replay
            # against, so the request's own prefix is the only evidence — the
            # same judgement the first window makes. An accepted empty
            # continuation reaches steady state in exactly that shape.
            trim_short_prefix=not suffix_state.pending_suffix,
            minimum_prefix_words=max(
                MIN_PREFIX_ECHO_WORDS,
                int(
                    state.transcript.chunk_size_sec * _MAX_PREFIX_ECHO_WORDS_PER_SECOND
                ),
            ),
        )
        if not decoder_prefix_replayed:
            text, decoder_prefix_replayed = suffix_state.trim_pending_prefix_echo(
                text, step.decoder_prefix
            )
        if state.encoder_window_active and not decoder_prefix_replayed:
            _, short_prefix_matched = suffix_state.trim_prefix_echo(
                text, step.decoder_prefix, trim_short_prefix=True
            )
            if short_prefix_matched:
                # Neither anchor can distinguish prompt echo from repeated
                # speech. Keeping it lets Local Agreement re-emit history;
                # trimming it can delete speech, so append-only output must fail.
                raise RuntimeError(
                    "realtime ASR cannot reconcile an ambiguous decoder-prefix "
                    "replay after encoder windowing is active"
                )
        if not state.encoder_window_active and not decoder_prefix_replayed:
            # The first continuation may replay only the un-emitted cumulative
            # tail rather than the full decoder prefix. A full-prefix match has
            # already consumed this request's one permitted echo trim; trimming
            # the handoff again could delete a phrase the user repeated.
            text, _ = suffix_state.trim_prefix_echo(
                text,
                step.handoff_text,
                trim_short_prefix=True,
            )
        # Every supported echo is now removed, so the empty check and the
        # no-boundary flag see the model's own continuation: a handoff replay
        # must not bypass the one-shot empty deferral, and handoff whitespace
        # must not mask a no-boundary continuation past the cumulative retry.
        if (
            not text
            and step.end_offset_bytes > state.audio.last_processed_offset_bytes
            and not step.is_last
            and state.audio.last_attempted_offset_bytes
            == state.audio.last_processed_offset_bytes
        ):
            # Retry one later request before accepting an empty continuation.
            return _TranscriptionOutcome(audio_processed=False)

        continuation_without_word_boundaries = has_no_word_boundaries(text)

        if not state.encoder_window_active:
            text = join_text(step.handoff_text, text)

        update = suffix_state.reconcile(
            text,
            is_last=step.is_last,
            holdback_words=policy.adapter_policy.decoder_prefix_holdback_words,
        )
        return _TranscriptionOutcome(
            audio_processed=True,
            decoder_update=update,
            continuation_without_word_boundaries=continuation_without_word_boundaries,
        )

    async def _snapshot_samples(
        self, audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
    ) -> np.ndarray:
        pcm = audio.snapshot(start_offset_bytes, end_offset_bytes)
        return await asyncio.to_thread(pcm_to_float_samples, pcm)

    async def _generate_windowed_transcript(
        self,
        samples: np.ndarray,
        sampling_params: Dict[str, Any],
        prompt: str,
        *,
        encoder_window_config: AudioEncoderWindowConfig,
        on_update: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> GeneratedTranscript:
        result = await generate_asr_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            audio_data=samples,
            sampling_params=sampling_params,
            prompt=prompt,
            audio_encoder_window_config=encoder_window_config,
            on_update=on_update,
        )
        if result is None:
            raise RuntimeError("realtime ASR request returned no response")
        if result.finish_reason == "length":
            raise RuntimeError("realtime ASR decode reached max_new_tokens")
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
            state.decoder_suffix.apply(outcome.decoder_update)
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
        # decode may reveal a no-whitespace continuation that cannot use it.
        # The flag is computed before the handoff join, so a whitespace-bearing
        # handoff cannot mask it.
        return (
            step.uses_encoder_windows
            and not state.encoder_window_active
            and outcome.decoder_update is not None
            and outcome.continuation_without_word_boundaries
        )
