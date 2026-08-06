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
    is_near_silent_pcm,
    pcm_to_float_samples,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    DecoderSuffixUpdate,
    GeneratedTranscript,
    StreamingASRState,
    generate_asr_transcript,
    has_no_word_boundaries,
    is_cjk_char,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Leave headroom above conversational speech while preventing intermediate
# requests from spending the adapter's full final-commit decode budget.
_MIN_DECODER_SUFFIX_TOKENS = 24
_MAX_DECODER_SUFFIX_TOKENS_PER_SECOND = 16


def estimate_decoder_suffix_token_budget(
    new_pcm_bytes: int, pcm_bytes_per_second: int, pending_suffix: str
) -> int:
    """Estimate an intermediate suffix budget from new audio and pending text.

    This limits repeated work from a looping intermediate decode; final commits
    still use the adapter's full budget. The speech rate and pending-text
    allowances are conservative heuristics, not tokenizer-level guarantees.
    """
    new_audio_tokens = math.ceil(
        new_pcm_bytes / pcm_bytes_per_second * _MAX_DECODER_SUFFIX_TOKENS_PER_SECOND
    )
    pending_words = len(pending_suffix.split())
    pending_cjk_chars = sum(is_cjk_char(char) for char in pending_suffix)
    return max(
        _MIN_DECODER_SUFFIX_TOKENS,
        new_audio_tokens + 2 * pending_words + pending_cjk_chars,
    )


class RealtimeASRState(msgspec.Struct):
    """Mutable audio and transcript progress for the current input buffer."""

    audio: AudioBuffer
    transcript: StreamingASRState
    # No-whitespace CJK cannot safely use the word-based decoder-prefix path.
    encoder_window_disabled: bool = False
    # True after cumulative transcript state has handed off to suffix decoding.
    encoder_window_active: bool = False
    # A length-limited decode must keep audio from this absolute offset until a
    # full-budget commit replay succeeds.
    final_replay_start_offset_bytes: Optional[int] = None

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.latest_text)

    @property
    def has_new_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.last_processed_offset_bytes

    @property
    def final_replay_required(self) -> bool:
        return self.final_replay_start_offset_bytes is not None


class _TranscriptionStep(msgspec.Struct, frozen=True):
    """Resolved mode, audio range, and text context for one transcription."""

    is_last: bool
    uses_encoder_windows: bool
    start_offset_bytes: int
    end_offset_bytes: int
    decoder_prefix: str = ""


class _TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Reconciled text and whether the step's audio may be committed.

    ``audio_processed=False`` keeps the covered audio for the next step when
    the model cannot produce a usable continuation.
    """

    audio_processed: bool
    cumulative_delta: str = ""
    decoder_update: Optional[DecoderSuffixUpdate] = None
    replay_start_offset_bytes: Optional[int] = None

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
        if (
            self._encoder_window_policy is not None
            and not state.encoder_window_disabled
            and not state.encoder_window_active
            and not state.transcript.is_decoder_prefix_compatible()
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
        if state.encoder_window_active:
            return state.transcript.flush_pending_decoder_suffix()
        return state.transcript.flush_cumulative_transcript()

    def _build_transcription_step(
        self,
        state: RealtimeASRState,
        is_last: bool,
    ) -> _TranscriptionStep:
        """Resolve the mode, audio range, and text context for the next call."""
        transcript = state.transcript
        audio = state.audio
        policy = self._encoder_window_policy
        if (
            policy is not None
            and not state.encoder_window_disabled
            and (
                state.encoder_window_active
                or (
                    not is_last
                    and audio.received_bytes > policy.activation_threshold_bytes
                )
            )
        ):
            return _TranscriptionStep(
                is_last=is_last,
                uses_encoder_windows=True,
                start_offset_bytes=self._encoder_window_start_offset(state),
                end_offset_bytes=audio.received_bytes,
                decoder_prefix=transcript.get_bounded_decoder_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.decoder_prefix_max_tokens,
                    include_unconfirmed=not state.encoder_window_active,
                ),
            )
        return _TranscriptionStep(
            is_last=is_last,
            uses_encoder_windows=False,
            start_offset_bytes=0,
            end_offset_bytes=audio.received_bytes,
            decoder_prefix=transcript.get_cumulative_prompt_prefix(),
        )

    def _encoder_window_start_offset(self, state: RealtimeASRState) -> int:
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
        complete_end = audio.received_bytes // window_bytes * window_bytes
        # Deferred audio must remain inside the next rolling request even when
        # the nominal context horizon advances.
        start = min(
            complete_end - policy.context_window_count * window_bytes,
            audio.last_processed_offset_bytes // window_bytes * window_bytes,
        )
        if state.final_replay_start_offset_bytes is not None:
            start = min(start, state.final_replay_start_offset_bytes)
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
        delta = state.transcript.reconcile_cumulative_transcript(
            generation.text, is_last=step.is_last
        )
        return _TranscriptionOutcome(audio_processed=True, cumulative_delta=delta)

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

        # Intermediate requests use the heuristic suffix budget above. The
        # final commit keeps the adapter budget so pending text is not limited
        # by that estimate.
        if not step.is_last:
            max_suffix_tokens = estimate_decoder_suffix_token_budget(
                step.end_offset_bytes - state.audio.last_processed_offset_bytes,
                self.pcm_bytes_per_second,
                state.transcript.pending_suffix,
            )
            sampling_params = dict(sampling_params)
            configured_max = sampling_params.get("max_new_tokens")
            sampling_params["max_new_tokens"] = (
                min(configured_max, max_suffix_tokens)
                if configured_max is not None
                else max_suffix_tokens
            )

        generation = await self._generate_transcript(
            samples,
            sampling_params,
            self.adapter.prompt_template + step.decoder_prefix,
            encoder_window_config=policy.encoder_window_config,
        )
        if step.is_last and generation.finish_reason == "length":
            raise RuntimeError("final realtime ASR decode reached max_new_tokens")
        replay_start_offset_bytes = (
            step.start_offset_bytes if generation.finish_reason == "length" else None
        )
        text = generation.text
        text = state.transcript.trim_decoder_prefix_echo(
            text,
            step.decoder_prefix,
            # During the one-way handoff the complete decoder prefix is known
            # existing context. Its exact replay must not be combined with the
            # cumulative holdback a second time.
            trim_short_prefix=not state.encoder_window_active,
        )

        new_pcm = state.audio.snapshot(
            state.audio.last_processed_offset_bytes, step.end_offset_bytes
        )
        if (
            not text
            and new_pcm
            and not step.is_last
            and not is_near_silent_pcm(new_pcm)
            and len(new_pcm) <= policy.window_bytes
        ):
            # Retry one encoder window with more context. Beyond that horizon,
            # accept an empty transcript so unrecognized audio cannot make
            # per-request work grow without bound.
            return _TranscriptionOutcome(
                audio_processed=False,
                replay_start_offset_bytes=replay_start_offset_bytes,
            )

        if not state.encoder_window_active:
            text = state.transcript.prepend_unemitted_cumulative_text(text)

        update = state.transcript.reconcile_decoder_suffix(
            text,
            is_last=step.is_last,
            holdback_words=policy.decoder_prefix_holdback_words,
        )
        return _TranscriptionOutcome(
            audio_processed=True,
            decoder_update=update,
            replay_start_offset_bytes=replay_start_offset_bytes,
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
        result = await generate_asr_transcript(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            audio_data=samples,
            sampling_params=sampling_params,
            prompt=prompt,
            mm_processor_kwargs={"audio_encoder_window_config": encoder_window_config},
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
            outcome.decoder_update is not None
            and outcome.decoder_update.pending_suffix is not None
        ):
            state.transcript.commit_decoder_suffix_update(
                outcome.decoder_update,
                is_last=step.is_last,
            )
        if outcome.replay_start_offset_bytes is not None:
            replay_start = outcome.replay_start_offset_bytes
            if state.final_replay_start_offset_bytes is not None:
                replay_start = min(replay_start, state.final_replay_start_offset_bytes)
            state.final_replay_start_offset_bytes = replay_start
        elif step.is_last:
            state.final_replay_start_offset_bytes = None
        if not outcome.audio_processed:
            return
        if step.uses_encoder_windows:
            state.encoder_window_active = True
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
