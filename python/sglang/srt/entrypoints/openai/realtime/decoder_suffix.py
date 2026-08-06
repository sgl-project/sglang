"""Text reconciliation for realtime ASR decoder continuation."""

import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import msgspec

from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    is_cjk_char,
    needs_space,
)

_MAX_CHARS_PER_TOKEN = 64


class DecoderSuffixUpdate(msgspec.Struct, frozen=True):
    """A suffix update that can be applied after its request is accepted."""

    delta: str
    # None means the decode made no state transition.
    pending_suffix: Optional[str]


@dataclass
class DecoderSuffixState:
    """Reconcile suffix-only decodes after encoder windowing starts."""

    emitted_text: str
    pending_suffix: str = ""

    @property
    def latest_text(self) -> str:
        return _join_text(self.emitted_text, self.pending_suffix)

    def get_bounded_prefix(self, tokenizer, max_tokens: int) -> str:
        """Return recent emitted context for the next decoder request."""
        tail = self.emitted_text[-max_tokens * _MAX_CHARS_PER_TOKEN :]
        token_ids = tokenizer.encode(tail, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return tail
        return tokenizer.decode(
            token_ids[-max_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).lstrip()

    def reconcile(
        self, decoded_suffix: str, *, is_last: bool, holdback_words: int
    ) -> DecoderSuffixUpdate:
        """Compute an update without mutating state."""
        if self.emitted_text and (not self.pending_suffix or is_last):
            decoded_suffix, _ = _trim_word_overlap(self.emitted_text, decoded_suffix)
        previous_suffix = self.pending_suffix

        if is_last:
            return DecoderSuffixUpdate(
                delta=decoded_suffix or previous_suffix,
                pending_suffix="",
            )
        if not decoded_suffix:
            return DecoderSuffixUpdate(delta="", pending_suffix=None)
        if not previous_suffix:
            return DecoderSuffixUpdate(delta="", pending_suffix=decoded_suffix)

        if has_no_word_boundaries(previous_suffix) or has_no_word_boundaries(
            decoded_suffix
        ):
            emit_count = max(
                0,
                _common_prefix_len(previous_suffix, decoded_suffix) - holdback_words,
            )
            return DecoderSuffixUpdate(
                delta=decoded_suffix[:emit_count],
                pending_suffix=decoded_suffix[emit_count:],
            )

        previous_words = previous_suffix.split()
        decoded_words = decoded_suffix.split()
        # Keep the unstable acoustic tail out of the next decoder prefix.
        emit_count = max(
            0,
            _normalized_word_prefix_len(previous_words, decoded_words) - holdback_words,
        )
        return DecoderSuffixUpdate(
            delta=" ".join(decoded_words[:emit_count]),
            pending_suffix=" ".join(decoded_words[emit_count:]),
        )

    def apply(self, update: DecoderSuffixUpdate, *, is_last: bool) -> str:
        if update.pending_suffix is None:
            return ""
        delta = self._append_emitted_text(update.delta)
        self.pending_suffix = "" if is_last else update.pending_suffix
        return delta

    def flush(self) -> str:
        delta = self._append_emitted_text(self.pending_suffix)
        self.pending_suffix = ""
        return delta

    def trim_prefix_echo(
        self,
        decoded_suffix: str,
        decoder_prefix: str,
        *,
        trim_short_prefix: bool = False,
        minimum_prefix_words: int = 24,
    ) -> str:
        """Remove an exact replay of text supplied as decoder context."""
        decoded_words = decoded_suffix.split()
        prefix_words = decoder_prefix.split()
        if (
            not prefix_words
            or (not trim_short_prefix and len(prefix_words) < minimum_prefix_words)
            or len(decoded_words) < len(prefix_words)
        ):
            return decoded_suffix
        if _normalized_word_prefix_len(prefix_words, decoded_words) != len(
            prefix_words
        ):
            return decoded_suffix
        return " ".join(decoded_words[len(prefix_words) :])

    def _append_emitted_text(self, delta: str) -> str:
        if delta:
            self.emitted_text = _join_text(self.emitted_text, delta)
        return delta


def cumulative_handoff_text(state: StreamingASRState) -> Optional[str]:
    """Return cumulative text not emitted before switching decode modes."""
    confirmed_words = state.confirmed_text.split()
    full_words = state.full_transcript.split()
    common_count = _normalized_word_prefix_len(confirmed_words, full_words)
    if common_count < len(confirmed_words):
        return None
    return " ".join(full_words[common_count:])


def cumulative_is_suffix_compatible(state: StreamingASRState) -> bool:
    """Whether cumulative text can safely hand off to word-based suffix state."""
    return (
        not (
            has_no_word_boundaries(state.emitted_text)
            or has_no_word_boundaries(state.full_transcript)
        )
        and cumulative_handoff_text(state) is not None
    )


def has_no_word_boundaries(text: str) -> bool:
    """Whether whitespace-based suffix reconciliation is unsafe."""
    text = text.strip()
    return (
        bool(text)
        and not any(char.isspace() for char in text)
        and any(is_cjk_char(char) or 0x0E00 <= ord(char) <= 0x0E7F for char in text)
    )


def join_handoff_text(pending_text: str, decoded_suffix: str) -> str:
    return _join_text(pending_text, decoded_suffix)


def _join_text(left: str, right: str) -> str:
    if not left or not right:
        return left or right
    separator = " " if needs_space(left, right) else ""
    return f"{left}{separator}{right}"


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def _normalize_overlap_word(word: str) -> str:
    word = unicodedata.normalize("NFKC", word)
    start, end = 0, len(word)
    while start < end and unicodedata.category(word[start])[0] == "P":
        start += 1
    while end > start and unicodedata.category(word[end - 1])[0] == "P":
        end -= 1
    return word[start:end].lower()


def _normalized_word_prefix_len(left_words: List[str], right_words: List[str]) -> int:
    count = 0
    for left_word, right_word in zip(left_words, right_words):
        if _normalize_overlap_word(left_word) != _normalize_overlap_word(right_word):
            break
        count += 1
    return count


def _trim_word_overlap(emitted_text: str, decoded_text: str) -> tuple[str, bool]:
    """Trim only a normalized decoded prefix matching the emitted tail."""
    decoded_words = decoded_text.split()
    if not decoded_words:
        return decoded_text, False
    emitted_tail = emitted_text.rsplit(maxsplit=len(decoded_words))[
        -len(decoded_words) :
    ]
    if not emitted_tail:
        return decoded_text, False
    emitted_tail_norm = [_normalize_overlap_word(word) for word in emitted_tail]
    decoded_norm = [_normalize_overlap_word(word) for word in decoded_words]
    max_overlap = min(len(emitted_tail_norm), len(decoded_norm))

    for overlap in range(max_overlap, 0, -1):
        if emitted_tail_norm[-overlap:] == decoded_norm[:overlap] and any(
            emitted_tail_norm[-overlap:]
        ):
            return " ".join(decoded_words[overlap:]), True
    return decoded_text, False
