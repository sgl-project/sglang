"""Text reconciliation for realtime ASR decoder continuation."""

import unicodedata
from typing import List, Optional

import msgspec

from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    is_cjk_char,
    needs_space,
)

_MAX_CHARS_PER_TOKEN = 64
MIN_PREFIX_ECHO_WORDS = 24
_THAI_CODEPOINT_START = 0x0E00
_THAI_CODEPOINT_END = 0x0E7F


class DecoderSuffixUpdate(msgspec.Struct, frozen=True):
    """A suffix update that can be applied after its request is accepted."""

    delta: str
    # None means the decode made no state transition.
    pending_suffix: Optional[str]


class DecoderSuffixState(msgspec.Struct):
    """Reconcile suffix-only decodes after encoder windowing starts."""

    emitted_text: str
    pending_suffix: str = ""

    @property
    def latest_text(self) -> str:
        return join_text(self.emitted_text, self.pending_suffix)

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
        previous_suffix = self.pending_suffix

        if is_last:
            # Without audio alignment, a matching emitted prefix can be a real
            # repetition. Keep the final decode instead of guessing it is echo.
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

    def apply(self, update: DecoderSuffixUpdate) -> None:
        if update.pending_suffix is None:
            return
        self._append_emitted_text(update.delta)
        self.pending_suffix = update.pending_suffix

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
        minimum_prefix_words: int = MIN_PREFIX_ECHO_WORDS,
    ) -> tuple[str, bool]:
        """Remove an exact replay of supplied context and report a match."""
        decoded_words = decoded_suffix.split()
        prefix_words = decoder_prefix.split()
        if (
            not prefix_words
            or (not trim_short_prefix and len(prefix_words) < minimum_prefix_words)
            or len(decoded_words) < len(prefix_words)
        ):
            return decoded_suffix, False
        if _normalized_word_prefix_len(prefix_words, decoded_words) != len(
            prefix_words
        ):
            return decoded_suffix, False
        return " ".join(decoded_words[len(prefix_words) :]), True

    def _append_emitted_text(self, delta: str) -> str:
        if delta:
            self.emitted_text = join_text(self.emitted_text, delta)
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
        and any(
            is_cjk_char(char)
            or _THAI_CODEPOINT_START <= ord(char) <= _THAI_CODEPOINT_END
            for char in text
        )
    )


def join_text(left: str, right: str) -> str:
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
