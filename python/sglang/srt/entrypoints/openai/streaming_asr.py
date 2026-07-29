"""Text-side reconciliation and request helpers for streaming ASR.

Shared by the HTTP streaming transcription endpoint (``process_asr_chunk``)
and the realtime WebSocket endpoint (via ``RealtimeASRProcessor``): decoded
model text goes in, stable transcript deltas come out. ``StreamingASRState``
holds the two reconciliation machines; ``generate_asr_text`` runs the single
stateless backend request both endpoints build on.
"""

import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import msgspec
import numpy as np
import soundfile as sf
from fastapi import Request

from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


_MAX_CJK_CHARS_PER_SECOND = 16
# Over-covers any BPE token, so a character tail always holds enough tokens.
_MAX_CHARS_PER_TOKEN = 64


_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


class DecoderSuffixDecision(msgspec.Struct, frozen=True):
    """A previewed decoder-suffix update that can be committed atomically."""

    delta: str
    # None means the hypothesis made no state transition.
    pending_suffix: Optional[str]


@dataclass
class StreamingASRState:
    """Reconcile decoded text into a stable stream of transcript deltas.

    Text state only: no audio buffer, GPU state, or scheduler state lives here.
    Two machines share the emitted-text anchor and the trim helpers:

    - Cumulative machine (below the gate, and no-whitespace CJK): every decode
      re-transcribes all audio, so ``update()``/``finalize()`` emit only the
      words that stopped changing between hypotheses (word/char rollback).
    - Decoder-suffix machine (windowed mode): every decode continues from a
      text prefix, so ``preview_decoder_suffix()`` computes a delta without
      mutating state and ``commit_decoder_suffix()`` applies it. The one-way
      handoff between the machines is ``prepare_decoder_suffix_transition()``.
    """

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    # Hypothesis prefix the rollback machine treats as stable; may lag emitted
    # text after an implausible CJK jump.
    confirmed_text: str = ""
    # Text actually sent to the client; prompts and dedupe anchor on it.
    emitted_text: str = ""
    # Latest complete hypothesis (cumulative) or emitted + pending (suffix).
    full_transcript: str = ""
    # Decoded text awaiting cross-decode agreement before it may be emitted.
    pending_suffix: str = ""
    # Decode counter; gates cumulative prompt-prefix injection.
    chunk_index: int = 0

    # --- Cumulative machine: reconcile full re-transcription hypotheses ---

    def apply_hypothesis(self, text: str, *, is_last: bool) -> str:
        """Reconcile one complete model hypothesis with already emitted text."""
        if is_last:
            self.full_transcript = text
            return self.finalize()
        return self.update(text)

    def update(self, new_transcript: str) -> str:
        if is_cjk_no_whitespace(new_transcript):
            return self._update_chars(new_transcript)

        old_confirmed = self.confirmed_text
        words = new_transcript.split()
        holdback = self.unfixed_token_num
        if holdback:
            self.confirmed_text = " ".join(words[: max(0, len(words) - holdback)])
        else:
            self.confirmed_text = new_transcript
        self.full_transcript = new_transcript
        self.chunk_index += 1
        return self._emit_word_delta(old_confirmed, self.confirmed_text)

    def _update_chars(self, new_transcript: str) -> str:
        """Use character rollback when whitespace cannot define stable words."""
        old_confirmed = self.confirmed_text
        holdback = max(0, self.unfixed_token_num)
        if holdback == 0:
            cut = len(new_transcript)
        elif len(new_transcript) > holdback:
            cut = len(new_transcript) - holdback
        else:
            cut = 0
        # Do not split an embedded Latin word at the char holdback boundary.
        while (
            0 < cut < len(new_transcript)
            and _is_word_char(new_transcript[cut - 1])
            and _is_word_char(new_transcript[cut])
        ):
            cut -= 1
        candidate_confirmed = new_transcript[:cut]
        self.full_transcript = new_transcript
        self.chunk_index += 1

        common_count = _common_prefix_len(old_confirmed, candidate_confirmed)
        delta = candidate_confirmed[common_count:]
        max_delta_chars = max(24, int(self.chunk_size_sec * _MAX_CJK_CHARS_PER_SECOND))
        if len(delta) > max_delta_chars:
            # A cumulative decode can transiently expand or rewrite a repeated
            # CJK passage. Keep the latest hypothesis for commit rather than
            # publishing a jump that cannot belong to one audio chunk.
            return ""

        self.confirmed_text = candidate_confirmed
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        return self._record_emit(delta)

    def finalize(self) -> str:
        if is_cjk_no_whitespace(self.full_transcript):
            # confirmed_text can intentionally lag after an implausibly large
            # intermediate jump; finalize against what reached the client.
            old_confirmed = self.emitted_text
            self.confirmed_text = self.full_transcript
            common_count = _cjk_common_prefix_end(old_confirmed, self.full_transcript)
            delta = self.full_transcript[common_count:]
            if common_count == 0:
                delta = self._trim_cjk_emitted_overlap(delta)
            return self._record_emit(delta)

        old_confirmed = self.confirmed_text
        self.confirmed_text = self.full_transcript
        return self._emit_word_delta(old_confirmed, self.full_transcript)

    def get_prefix_text(self) -> str:
        if self.chunk_index < self.unfixed_chunk_num or not self.emitted_text:
            return ""
        # Word overlap is unsafe for no-whitespace CJK; keep that path cumulative.
        if is_cjk_no_whitespace(self.emitted_text):
            return ""
        return self.emitted_text

    def _emit_word_delta(self, old_text: str, new_text: str) -> str:
        """Emit the word-level tail of new_text not already covered by old_text."""
        old_words = old_text.split()
        new_words = new_text.split()
        common_count = _norm_common_prefix_len(old_words, new_words)
        delta = " ".join(new_words[common_count:])
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        delta = self._trim_large_prompt_echo(delta)
        return self._record_emit(delta)

    # --- Decoder-suffix machine: reconcile windowed continuations ---

    def can_start_decoder_prefix(self) -> bool:
        """Whether cumulative text can safely hand off to word-based suffix state."""
        return not (
            is_cjk_no_whitespace(self.emitted_text)
            or is_cjk_no_whitespace(self.full_transcript)
        )

    def get_bounded_decoder_prefix(
        self, tokenizer, max_tokens: int, *, include_unconfirmed: bool = False
    ) -> str:
        """Return recent emitted context for a suffix-only decoder request."""
        source_text = self.emitted_text
        if include_unconfirmed:
            source_text = _join_words(
                source_text, self._cumulative_unemitted_tail() or ""
            )
        # Only the trailing max_tokens tokens can survive the cap, so tokenize
        # a bounded character tail instead of the whole growing transcript.
        tail = source_text[-max_tokens * _MAX_CHARS_PER_TOKEN :]
        token_ids = tokenizer.encode(tail, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return tail
        return tokenizer.decode(
            token_ids[-max_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).lstrip()

    def prepare_decoder_suffix_transition(self, candidate: str) -> str:
        """Preserve the cumulative holdback while adopting a decoder prefix."""
        return _join_words(self._cumulative_unemitted_tail() or "", candidate)

    def _cumulative_unemitted_tail(self) -> Optional[str]:
        """Return the current hypothesis tail not yet emitted by rollback."""
        confirmed_words = self.confirmed_text.split()
        full_words = self.full_transcript.split()
        common_count = _norm_common_prefix_len(confirmed_words, full_words)
        if common_count < len(confirmed_words):
            return None
        return " ".join(full_words[common_count:])

    def preview_decoder_suffix(
        self, candidate: str, *, is_last: bool = False, holdback_words: int
    ) -> DecoderSuffixDecision:
        """Calculate a suffix update without changing emitted transcript state."""
        candidate = self._trim_large_prompt_echo(candidate)
        if self.emitted_text and (not self.pending_suffix or is_last):
            candidate, _ = _dedupe_by_word(self.emitted_text, candidate)
        previous = self.pending_suffix

        if is_last:
            return DecoderSuffixDecision(delta=candidate or previous, pending_suffix="")
        if not candidate:
            return DecoderSuffixDecision(delta="", pending_suffix=None)
        if not previous:
            return DecoderSuffixDecision(delta="", pending_suffix=candidate)

        if is_cjk_no_whitespace(previous) or is_cjk_no_whitespace(candidate):
            emit_count = max(
                0, _common_prefix_len(previous, candidate) - holdback_words
            )
            return DecoderSuffixDecision(
                delta=candidate[:emit_count], pending_suffix=candidate[emit_count:]
            )

        previous_words = previous.split()
        candidate_words = candidate.split()
        # Keep the acoustic tail out of the decoder prefix. A premature
        # sentence end there can make the next request stop before newly
        # appended audio, while the retained audio can safely recover it.
        emit_count = max(
            0, _norm_common_prefix_len(previous_words, candidate_words) - holdback_words
        )
        return DecoderSuffixDecision(
            delta=" ".join(candidate_words[:emit_count]),
            pending_suffix=" ".join(candidate_words[emit_count:]),
        )

    def commit_decoder_suffix(
        self, decision: DecoderSuffixDecision, *, is_last: bool
    ) -> str:
        """Apply a previewed decision; preview and commit are split so a mode
        fallback can discard the preview without touching emitted state."""
        self.chunk_index += 1
        if decision.pending_suffix is None:
            return ""
        delta = self._record_emit(decision.delta)
        self._set_suffix_candidate("" if is_last else decision.pending_suffix)
        return delta

    def finalize_decoder_suffix(self) -> str:
        """Emit the pending suffix at item end: no further decode will confirm it."""
        delta = self._record_emit(self.pending_suffix)
        self._set_suffix_candidate("")
        return delta

    def _set_suffix_candidate(self, candidate: str) -> None:
        """Record the latest unconfirmed suffix decoded after emitted_text."""
        self.pending_suffix = candidate
        self.full_transcript = _join_words(self.emitted_text, candidate)
        self.confirmed_text = self.emitted_text

    # --- Shared emit and trim helpers ---

    def _record_emit(self, delta: str) -> str:
        if delta:
            if self.emitted_text:
                sep = " " if needs_space(self.emitted_text, delta) else ""
                self.emitted_text = f"{self.emitted_text}{sep}{delta}".strip()
            else:
                self.emitted_text = delta
        return delta

    def _trim_cjk_emitted_overlap(self, delta: str) -> str:
        """Drop an already-emitted leading CJK run after char/word path flips."""
        if not delta or not self.emitted_text:
            return delta
        # Ignore trailing sentence enders when aligning: a chunk boundary can
        # close a clause ("...停。") that the re-transcription then extends
        # ("...停滞..."), mirroring the word path's edge-punctuation stripping.
        emitted = self.emitted_text.rstrip("。！？.!?")
        max_k = min(len(delta), len(emitted))
        for k in range(max_k, 0, -1):
            if emitted[-k:] != delta[:k]:
                continue
            if all(is_cjk_char(c) for c in delta[:k]):
                return delta[k:].lstrip()
        return delta

    def _trim_large_prompt_echo(self, delta: str) -> str:
        """Drop an obvious full transcript-prefix echo from chunked ASR."""
        if not delta or not self.emitted_text:
            return delta

        delta_words = delta.split()
        # ~16 words/sec outpaces real speech, so a delta this long that also
        # prefix-matches emitted text is a prompt echo, not new audio content.
        max_words_for_chunk = max(24, int(self.chunk_size_sec * 16))
        if len(delta_words) <= max_words_for_chunk:
            return delta
        emitted_words = self.emitted_text.split()

        while len(delta_words) > max_words_for_chunk:
            max_match = min(len(delta_words), len(emitted_words))
            match = 0
            for i in range(max_match):
                if _dedupe_norm(delta_words[i]) != _dedupe_norm(emitted_words[i]):
                    break
                match += 1

            if match < max_words_for_chunk:
                break
            delta_words = delta_words[match:]

        return " ".join(delta_words)


def split_audio_chunks(audio_data: bytes, chunk_size_sec: float) -> List[bytes]:
    if not audio_data:
        raise ValueError("audio_data is empty")
    if chunk_size_sec <= 0:
        raise ValueError(f"chunk_size_sec must be positive, got {chunk_size_sec}")
    audio_file = io.BytesIO(audio_data)
    try:
        data, sample_rate = sf.read(audio_file, dtype="float32")
    except sf.LibsndfileError as e:
        raise ValueError(f"failed to decode audio: {e}") from e
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    total_samples = len(data)
    chunks = []
    for end in range(
        chunk_size_samples, total_samples + chunk_size_samples, chunk_size_samples
    ):
        end = min(end, total_samples)
        buf = io.BytesIO()
        sf.write(buf, data[:end], sample_rate, format="WAV")
        chunks.append(buf.getvalue())
    return chunks


def normalize_whitespace(text: str) -> str:
    return _PUNCT_WS_RE.sub(r"\1", text)


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def is_cjk_char(c: str) -> bool:
    """CJK-context character that takes no inter-word space."""
    cp = ord(c)
    if 0xFFA0 <= cp <= 0xFFDC:  # halfwidth Hangul jamo -- Korean is space-delimited
        return False
    return (
        0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana (incl. ー / ・)
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth & Fullwidth Forms
    )


def _is_cjk_script_char(c: str) -> bool:
    """Han or kana character that requires character-level reconciliation."""
    cp = ord(c)
    return (
        0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
    )


def _is_word_char(c: str) -> bool:
    return c.isalnum() and not is_cjk_char(c)


def is_cjk_no_whitespace(text: str) -> bool:
    return (
        bool(text)
        and not any(c.isspace() for c in text)
        and any(_is_cjk_script_char(c) for c in text)
    )


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for lc, rc in zip(left, right):
        if lc != rc:
            break
        count += 1
    return count


def _cjk_common_prefix_end(left: str, right: str) -> int:
    """Map a punctuation-insensitive CJK prefix to an index in ``right``."""
    left_index = 0
    right_index = 0
    while True:
        while (
            left_index < len(left) and unicodedata.category(left[left_index])[0] == "P"
        ):
            left_index += 1
        while (
            right_index < len(right)
            and unicodedata.category(right[right_index])[0] == "P"
        ):
            right_index += 1
        if left_index >= len(left) or right_index >= len(right):
            break
        if left[left_index] != right[right_index]:
            break
        left_index += 1
        right_index += 1

    while (
        right_index < len(right) and unicodedata.category(right[right_index])[0] == "P"
    ):
        right_index += 1
    return right_index


def needs_space(prev: str, cur: str) -> bool:
    if not prev or not cur:
        return False
    if prev[-1].isspace() or cur[0].isspace():
        return False
    if cur[0] in _NO_SPACE_BEFORE or prev[-1] in _NO_SPACE_AFTER:
        return False
    if is_cjk_char(prev[-1]) and is_cjk_char(cur[0]):
        return False
    return True


def _join_words(prev: str, cur: str) -> str:
    if not prev or not cur:
        return prev or cur
    sep = " " if needs_space(prev, cur) else ""
    return f"{prev}{sep}{cur}"


def _dedupe_norm(word: str) -> str:
    """Normalize a word so recasing and edge-punctuation drift between decodes
    cannot break overlap matching."""
    word = unicodedata.normalize("NFKC", word)
    lo, hi = 0, len(word)
    while lo < hi and unicodedata.category(word[lo])[0] == "P":
        lo += 1
    while hi > lo and unicodedata.category(word[hi - 1])[0] == "P":
        hi -= 1
    return word[lo:hi].lower()


def _norm_common_prefix_len(left_words: List[str], right_words: List[str]) -> int:
    """Common prefix robust to recasing and edge punctuation drift."""
    count = 0
    for lw, rw in zip(left_words, right_words):
        if _dedupe_norm(lw) != _dedupe_norm(rw):
            break
        count += 1
    return count


def _dedupe_by_word(committed_text: str, candidate_out: str) -> "tuple[str, bool]":
    """Trim only a normalized candidate prefix matching the committed tail."""
    candidate_words = candidate_out.split()
    if not candidate_words:
        return candidate_out, False
    committed_tail = committed_text.rsplit(maxsplit=len(candidate_words))[
        -len(candidate_words) :
    ]
    if not committed_tail:
        return candidate_out, False
    committed_tail_norm = [_dedupe_norm(w) for w in committed_tail]
    candidate_norm = [_dedupe_norm(w) for w in candidate_words]
    max_overlap = min(len(committed_tail_norm), len(candidate_norm))

    cut = 0
    for length in range(max_overlap, 0, -1):
        if committed_tail_norm[-length:] == candidate_norm[:length] and any(
            committed_tail_norm[-length:]
        ):
            cut = length
            break
    if cut == 0:
        return candidate_out, False
    return " ".join(candidate_words[cut:]), True


async def generate_asr_text(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    prompt: str,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Run one stateless backend request and return normalized model text."""
    chunk_request = GenerateReqInput(
        text=prompt,
        audio_data=audio_data,
        sampling_params=sampling_params,
        stream=False,
        modalities=["audio"],
        routing_key=routing_key,
    )

    try:
        ret = None
        async for ret in tokenizer_manager.generate_request(
            chunk_request,
            raw_request,
            internal_mm_processor_kwargs=mm_processor_kwargs,
        ):
            break
    except ValueError:
        logger.warning("[streaming_asr] ASR request failed", exc_info=True)
        raise

    if ret is None:
        logger.warning("[streaming_asr] ASR request returned no response")
        return None

    return normalize_whitespace(adapter.postprocess_text(ret.get("text", "")))


async def process_asr_chunk(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
) -> str:
    """Run and reconcile one cumulative chunk for HTTP streaming ASR."""
    text = await generate_asr_text(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        audio_data=audio_data,
        sampling_params=sampling_params,
        prompt=adapter.prompt_template + state.get_prefix_text(),
        raw_request=raw_request,
        routing_key=routing_key,
    )
    if text is None:
        return ""

    return state.apply_hypothesis(text, is_last=is_last)
