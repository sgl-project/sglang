"""Application service for realtime ASR inference.

The processor is connection-scoped but keeps no stream progress itself. Each call
builds and executes one stateless transcription step, then commits its outcome to
the explicit ``RealtimeASRState`` supplied by the realtime endpoint.

Per-chunk flow, driven by the endpoint::

    append audio -> next_chunk_ready()? -> process()
        _build_transcription_step  resolve mode, audio range, and prompt
        _execute_step              run the backend request, reconcile text
        _commit                    advance cursors, compact windowed PCM
    -> transcript delta
    commit event -> process(is_last=True), or finalize() if no new audio

Mode lifecycle: every item starts cumulative (identical to the pre-existing
behavior). Past the adapter's activation gate it switches to windowed --
encoder-aligned audio windows reused through the embedding cache plus a
bounded decoder prefix -- and stays there for the rest of the item.
No-whitespace CJK disables windowed; the item stays cumulative.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, Optional

import msgspec
import numpy as np

from sglang.srt.entrypoints.openai.realtime.audio_buffer import (
    PCM_SAMPLE_WIDTH,
    AudioBuffer,
    is_near_silent_pcm,
    pcm_to_float_samples,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    DecoderSuffixDecision,
    StreamingASRState,
    generate_asr_text,
    is_cjk_char,
    is_cjk_no_whitespace,
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


def decoder_suffix_token_budget(
    new_audio_bytes: int, bytes_per_second: int, pending_suffix: str
) -> int:
    """Bound suffix decoding without truncating text pending confirmation.

    A degenerate decode (e.g. looping on repetitive audio) must not spend the
    final-commit budget, so intermediate requests are capped. The cap must
    still cover everything a correct decode may emit: 16 tokens/s over-covers
    real speech for the new audio, and because the pending suffix is
    re-decoded on every request it gets ~2 tokens per word plus one per CJK
    char (BPE upper bounds). The floor keeps tiny audio increments decodable.
    """
    new_audio_tokens = math.ceil(
        new_audio_bytes / bytes_per_second * _MAX_DECODER_SUFFIX_TOKENS_PER_SECOND
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
    windowed_disabled: bool = False
    # True after cumulative transcript state has handed off to suffix decoding.
    windowed_started: bool = False

    @property
    def has_audio(self) -> bool:
        return self.audio.received_bytes > 0

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript.full_transcript)

    @property
    def has_new_audio(self) -> bool:
        return self.audio.received_bytes > self.audio.accepted_offset_bytes


class _TranscriptionStep(msgspec.Struct, frozen=True):
    """Resolved mode, audio range, and text context for one transcription."""

    is_last: bool
    windowed: bool
    committed_text: str
    start_offset_bytes: int
    end_offset_bytes: int
    decoder_prefix: str = ""


class _TranscriptionOutcome(msgspec.Struct, frozen=True):
    """Reconciled text and whether the step's audio may be committed.

    ``accept_audio=False`` keeps the covered audio for the next step instead of
    losing it when the model cannot produce a usable continuation.
    """

    accept_audio: bool
    direct_delta: str = ""
    decoder_decision: Optional[DecoderSuffixDecision] = None

    @property
    def delta(self) -> str:
        if self.decoder_decision is not None:
            return self.decoder_decision.delta
        return self.direct_delta


class _WindowedPolicy(msgspec.Struct, frozen=True):
    """Windowed-mode knobs, resolved once per connection because every input
    (adapter config, model window geometry) is fixed for its lifetime."""

    window_config: AudioEncoderWindowConfig
    decoder_prefix_max_tokens: int
    decoder_prefix_holdback_words: int
    context_window_count: int
    # Below this many received bytes the endpoint keeps its cumulative path.
    activation_bytes: int

    @property
    def window_bytes(self) -> int:
        return self.window_config.window_samples * PCM_SAMPLE_WIDTH


class RealtimeASRProcessor:
    """Plan and execute realtime ASR requests for explicit stream state."""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        server_args: ServerArgs,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.model_sample_rate = adapter.model_sample_rate
        self.bytes_per_second = self.model_sample_rate * PCM_SAMPLE_WIDTH
        self.max_buffer_seconds = server_args.asr_max_buffer_seconds

        state = StreamingASRState(**self.adapter.chunked_streaming_config)
        self.chunk_size_bytes = int(state.chunk_size_sec * self.bytes_per_second)
        self.max_buffer_bytes = self.max_buffer_seconds * self.bytes_per_second
        self._windowed_policy = self._resolve_windowed_policy(state)

    def create_state(self) -> RealtimeASRState:
        return RealtimeASRState(
            audio=AudioBuffer(),
            transcript=StreamingASRState(**self.adapter.chunked_streaming_config),
        )

    def next_chunk_ready(self, state: RealtimeASRState) -> bool:
        """True once a full chunk of audio has arrived past the last attempt."""
        return (
            state.audio.received_bytes - state.audio.attempted_offset_bytes
            >= self.chunk_size_bytes
        )

    def _resolve_windowed_policy(
        self, state: StreamingASRState
    ) -> Optional[_WindowedPolicy]:
        """Resolve the windowed policy, or None to keep the cumulative path;
        an invalid or missing config must degrade the connection, not fail it."""
        declared = self.adapter.realtime_long_audio_config
        if not isinstance(declared, dict):
            return None
        if "max_audio_context_windows" not in declared:
            return None
        mm_processor = self.tokenizer_manager.mm_processor
        if mm_processor is None or self.tokenizer_manager.tokenizer is None:
            return None
        try:
            window_config = mm_processor.resolve_audio_encoder_window_config(
                self.model_sample_rate
            )
            min_audio_sec = float(declared["min_audio_sec"])
            prefix_max_tokens = int(declared.get("decoder_prefix_max_tokens", 192))
            holdback_words = int(
                declared.get("decoder_prefix_holdback_words", state.unfixed_token_num)
            )
            context_window_count = int(declared["max_audio_context_windows"])
            if (
                window_config.window_samples <= 0
                or window_config.window_tokens <= 0
                or min_audio_sec < 0
                or not math.isfinite(min_audio_sec)
                or prefix_max_tokens <= 0
                or holdback_words < 0
                or context_window_count <= 0
            ):
                raise ValueError("windowed ASR policy values must be positive")
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning(
                "[realtime] invalid realtime_long_audio_config; windowed mode disabled",
                exc_info=True,
            )
            return None
        return _WindowedPolicy(
            window_config=window_config,
            decoder_prefix_max_tokens=prefix_max_tokens,
            decoder_prefix_holdback_words=holdback_words,
            context_window_count=context_window_count,
            activation_bytes=(
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

        Windowed inference previews without mutating transcript state, so when
        its first decode reveals no-whitespace CJK the same audio is simply
        redone on the cumulative path with nothing to roll back."""
        # No-whitespace CJK cannot use the word-based decoder-prefix path.
        if (
            self._windowed_policy is not None
            and not state.windowed_disabled
            and not state.windowed_started
            and not state.transcript.can_start_decoder_prefix()
        ):
            state.windowed_disabled = True

        step = self._build_transcription_step(state, is_last)
        outcome = await self._execute_step(state, step, sampling_params)

        if self._windowed_text_needs_fallback(state, step, outcome):
            # Nothing was emitted yet; redo this request on the cumulative path.
            state.windowed_disabled = True
            step = self._build_transcription_step(state, is_last)
            outcome = await self._execute_step(state, step, sampling_params)

        self._commit(state, step, outcome)
        return outcome.delta

    def finalize(self, state: RealtimeASRState) -> str:
        """Emit text still held back when the item commits without new audio."""
        if state.windowed_started:
            return state.transcript.finalize_decoder_suffix()
        return state.transcript.finalize()

    def _build_transcription_step(
        self,
        state: RealtimeASRState,
        is_last: bool,
    ) -> _TranscriptionStep:
        """Resolve the mode, audio range, and text context for the next call."""
        transcript = state.transcript
        audio = state.audio
        policy = self._windowed_policy
        if (
            policy is not None
            and not state.windowed_disabled
            and (
                state.windowed_started
                or (not is_last and audio.received_bytes > policy.activation_bytes)
            )
        ):
            return _TranscriptionStep(
                is_last=is_last,
                windowed=True,
                committed_text="",
                start_offset_bytes=self._windowed_start_offset(state),
                end_offset_bytes=audio.received_bytes,
                decoder_prefix=transcript.get_bounded_decoder_prefix(
                    self.tokenizer_manager.tokenizer,
                    policy.decoder_prefix_max_tokens,
                    include_unconfirmed=not state.windowed_started,
                ),
            )
        return _TranscriptionStep(
            is_last=is_last,
            windowed=False,
            committed_text=transcript.get_prefix_text(),
            start_offset_bytes=0,
            end_offset_bytes=audio.received_bytes,
        )

    def _windowed_start_offset(self, state: RealtimeASRState) -> int:
        """Pick a window-aligned start so resent windows keep their cache
        identity; never advances past unaccepted (deferred) audio."""
        policy = self._windowed_policy
        assert policy is not None
        audio = state.audio

        # Establish suffix state against the cumulative audio once. Later
        # requests can drop encoder-aligned history represented by the prefix.
        if not state.windowed_started:
            return 0
        window_bytes = policy.window_bytes
        complete_end = audio.received_bytes // window_bytes * window_bytes
        # Deferred audio must remain inside the next rolling request even when
        # the nominal context horizon advances.
        start = min(
            complete_end - policy.context_window_count * window_bytes,
            audio.accepted_offset_bytes // window_bytes * window_bytes,
        )
        return max(audio.base_offset_bytes, start)

    async def _execute_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        if step.windowed:
            return await self._execute_windowed_step(state, step, sampling_params)
        return await self._execute_cumulative_step(state, step, sampling_params)

    async def _execute_cumulative_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        """Re-transcribe the whole buffer and reconcile it against emitted
        text -- main's behavior, kept byte-identical below the gate."""
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )
        text = await self._generate(
            samples,
            sampling_params,
            self.adapter.prompt_template + step.committed_text,
        )
        delta = state.transcript.apply_hypothesis(text, is_last=step.is_last)
        return _TranscriptionOutcome(accept_audio=True, direct_delta=delta)

    async def _execute_windowed_step(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        sampling_params: Dict[str, Any],
    ) -> _TranscriptionOutcome:
        """Decode only the transcript continuation for the rolling window.

        Flow: budget the intermediate decode by new-audio length (so one stuck
        decode cannot spend the final-commit budget) -> generate with the
        decoder prefix -> the first windowed decode adopts the cumulative
        tail; a later voiced-but-empty decode is deferred at most one window
        -> preview the suffix delta for _commit."""
        policy = self._windowed_policy
        assert policy is not None
        if step.start_offset_bytes % policy.window_bytes:
            raise RuntimeError("encoder-window request is not window aligned")
        samples = await self._snapshot_samples(
            state.audio, step.start_offset_bytes, step.end_offset_bytes
        )

        # Intermediate requests decode only the new suffix; the final commit
        # may replay the prefix, so it keeps the adapter's full budget.
        if not step.is_last:
            max_suffix_tokens = decoder_suffix_token_budget(
                step.end_offset_bytes - state.audio.accepted_offset_bytes,
                self.bytes_per_second,
                state.transcript.pending_suffix,
            )
            sampling_params = dict(sampling_params)
            configured_max = sampling_params.get("max_new_tokens")
            sampling_params["max_new_tokens"] = (
                min(configured_max, max_suffix_tokens)
                if configured_max is not None
                else max_suffix_tokens
            )

        text = await self._generate(
            samples,
            sampling_params,
            self.adapter.prompt_template + step.decoder_prefix,
            window_config=policy.window_config,
        )

        if not state.windowed_started:
            text = state.transcript.prepare_decoder_suffix_transition(text)
        elif not step.is_last and not text:
            new_pcm = state.audio.snapshot(
                state.audio.accepted_offset_bytes, step.end_offset_bytes
            )
            # Defer a voiced-but-empty decode so the next request re-covers
            # it, at most one window so a stuck decoder cannot grow requests.
            if (
                new_pcm
                and not is_near_silent_pcm(new_pcm)
                and len(new_pcm) <= policy.window_bytes
            ):
                return _TranscriptionOutcome(accept_audio=False)

        decision = state.transcript.preview_decoder_suffix(
            text,
            is_last=step.is_last,
            holdback_words=policy.decoder_prefix_holdback_words,
        )
        return _TranscriptionOutcome(
            accept_audio=True,
            decoder_decision=decision,
        )

    async def _snapshot_samples(
        self, audio: AudioBuffer, start_offset_bytes: int, end_offset_bytes: int
    ) -> np.ndarray:
        pcm = audio.snapshot(start_offset_bytes, end_offset_bytes)
        return await asyncio.to_thread(pcm_to_float_samples, pcm)

    async def _generate(
        self,
        samples: np.ndarray,
        sampling_params: Dict[str, Any],
        prompt: str,
        *,
        window_config: Optional[AudioEncoderWindowConfig] = None,
    ) -> str:
        text = await generate_asr_text(
            tokenizer_manager=self.tokenizer_manager,
            adapter=self.adapter,
            audio_data=samples,
            sampling_params=sampling_params,
            prompt=prompt,
            mm_processor_kwargs={"audio_encoder_window_config": window_config},
        )
        if text is None:
            raise RuntimeError("realtime ASR request returned no response")
        return text

    def _commit(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        outcome: _TranscriptionOutcome,
    ) -> None:
        """Apply the outcome: `attempted` always advances so pacing waits for
        new audio; `accepted` and PCM compaction advance only when the outcome
        entered the transcript."""
        audio = state.audio
        audio.attempted_offset_bytes = step.end_offset_bytes
        if outcome.decoder_decision is not None:
            state.transcript.commit_decoder_suffix(
                outcome.decoder_decision,
                is_last=step.is_last,
            )
            state.windowed_started = True
        if not outcome.accept_audio:
            return
        audio.accepted_offset_bytes = step.end_offset_bytes
        if step.is_last:
            return
        if step.windowed:
            audio.discard_before(step.start_offset_bytes)

    def _windowed_text_needs_fallback(
        self,
        state: RealtimeASRState,
        step: _TranscriptionStep,
        outcome: _TranscriptionOutcome,
    ) -> bool:
        # Decoder-prefix reconciliation is word based; the first windowed
        # decode may reveal a no-whitespace transcript that cannot use it.
        return (
            step.windowed
            and not state.windowed_started
            and outcome.decoder_decision is not None
            and is_cjk_no_whitespace(outcome.decoder_decision.pending_suffix or "")
        )
