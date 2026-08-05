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


class DecoderSuffixUpdate(msgspec.Struct, frozen=True):
    """A decoder-suffix state update that can be committed atomically."""

    delta: str
    # None means the decode made no state transition.
    pending_suffix: Optional[str]


class GeneratedTranscript(msgspec.Struct, frozen=True):
    """Normalized ASR text plus the backend stop reason."""

    text: str
    finish_reason: Optional[str]


@dataclass
class StreamingASRState:
    """Reconcile decoded text into a stable stream of transcript deltas.

    Text state only: no audio buffer, GPU state, or scheduler state lives here.
    Two machines share the emitted-text anchor and the trim helpers:

    - Cumulative machine (below the gate, and text without word boundaries): every decode
      re-transcribes all audio, so ``reconcile_cumulative_transcript()`` and
      ``flush_cumulative_transcript()`` emit only the words that stopped
      changing between decodes (word/char rollback).
    - Decoder-suffix machine (encoder-window mode): every decode continues from
      a text prefix, so ``reconcile_decoder_suffix()`` derives an update
      without mutating state and ``commit_decoder_suffix_update()`` applies it.
      ``prepend_unemitted_cumulative_text()`` performs the one-way handoff.
    """

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    # Prefix the rollback machine treats as stable; may lag emitted text after
    # an implausible CJK jump.
    confirmed_text: str = ""
    # Text actually sent to the client; prompts and dedupe anchor on it.
    emitted_text: str = ""
    # Latest cumulative decode, or emitted text plus the pending suffix.
    latest_text: str = ""
    # Decoded text awaiting cross-decode agreement before it may be emitted.
    pending_suffix: str = ""
    # Decode counter; gates cumulative prompt-prefix injection.
    decode_count: int = 0

    # --- Cumulative machine: reconcile full re-transcriptions ---

    def reconcile_cumulative_transcript(
        self, decoded_text: str, *, is_last: bool
    ) -> str:
        """Reconcile one cumulative decode with already emitted text."""
        if is_last:
            self.latest_text = decoded_text
            return self.flush_cumulative_transcript()
        return self._reconcile_incremental_transcript(decoded_text)

    def _reconcile_incremental_transcript(self, decoded_text: str) -> str:
        if is_cjk_no_whitespace(decoded_text):
            return self._reconcile_incremental_chars(decoded_text)

        old_confirmed = self.confirmed_text
        words = decoded_text.split()
        holdback = self.unfixed_token_num
        if holdback:
            self.confirmed_text = " ".join(words[: max(0, len(words) - holdback)])
        else:
            self.confirmed_text = decoded_text
        self.latest_text = decoded_text
        self.decode_count += 1
        return self._emit_cumulative_word_delta(old_confirmed, self.confirmed_text)

    def _reconcile_incremental_chars(self, decoded_text: str) -> str:
        """Use character rollback when whitespace cannot define stable words."""
        old_confirmed = self.confirmed_text
        holdback = max(0, self.unfixed_token_num)
        if holdback == 0:
            cut = len(decoded_text)
        elif len(decoded_text) > holdback:
            cut = len(decoded_text) - holdback
        else:
            cut = 0
        # Do not split an embedded Latin word at the char holdback boundary.
        while (
            0 < cut < len(decoded_text)
            and _is_word_char(decoded_text[cut - 1])
            and _is_word_char(decoded_text[cut])
        ):
            cut -= 1
        stable_prefix = decoded_text[:cut]
        self.latest_text = decoded_text
        self.decode_count += 1

        common_count = _common_prefix_len(old_confirmed, stable_prefix)
        delta = stable_prefix[common_count:]
        max_delta_chars = max(24, int(self.chunk_size_sec * _MAX_CJK_CHARS_PER_SECOND))
        if len(delta) > max_delta_chars:
            # A cumulative decode can transiently expand or rewrite a repeated
            # CJK passage. Keep the latest text for commit rather than publishing
            # a jump that cannot belong to one audio chunk.
            return ""

        self.confirmed_text = stable_prefix
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        return self._append_emitted_text(delta)

    def flush_cumulative_transcript(self) -> str:
        """Emit the remaining cumulative text when the item ends."""
        if is_cjk_no_whitespace(self.latest_text):
            # confirmed_text can intentionally lag after an implausibly large
            # intermediate jump; flush against what reached the client.
            old_confirmed = self.emitted_text
            self.confirmed_text = self.latest_text
            common_count = _cjk_common_prefix_end(old_confirmed, self.latest_text)
            delta = self.latest_text[common_count:]
            if common_count == 0:
                delta = self._trim_cjk_emitted_overlap(delta)
            return self._append_emitted_text(delta)

        old_confirmed = self.confirmed_text
        self.confirmed_text = self.latest_text
        return self._emit_cumulative_word_delta(old_confirmed, self.latest_text)

    def get_cumulative_prompt_prefix(self) -> str:
        if self.decode_count < self.unfixed_chunk_num or not self.emitted_text:
            return ""
        # Word overlap is unsafe for no-whitespace CJK; keep that path cumulative.
        if is_cjk_no_whitespace(self.emitted_text):
            return ""
        return self.emitted_text

    def _emit_cumulative_word_delta(self, old_text: str, new_text: str) -> str:
        """Emit the word-level tail of new_text not already covered by old_text."""
        old_words = old_text.split()
        new_words = new_text.split()
        common_count = _normalized_word_prefix_len(old_words, new_words)
        delta = " ".join(new_words[common_count:])
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        delta = self._trim_large_prompt_echo(delta)
        return self._append_emitted_text(delta)

    # --- Decoder-suffix machine: reconcile encoder-window continuations ---

    def is_decoder_prefix_compatible(self) -> bool:
        """Whether cumulative text is compatible with word-based suffix state."""
        return not (
            has_no_word_boundaries(self.emitted_text)
            or has_no_word_boundaries(self.latest_text)
        )

    def get_bounded_decoder_prefix(
        self, tokenizer, max_tokens: int, *, include_unconfirmed: bool = False
    ) -> str:
        """Return recent emitted context for a suffix-only decoder request."""
        source_text = self.emitted_text
        if include_unconfirmed:
            source_text = _join_words(
                source_text, self._get_unemitted_cumulative_text() or ""
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

    def prepend_unemitted_cumulative_text(self, decoded_suffix: str) -> str:
        """Preserve the cumulative holdback while adopting a decoder prefix."""
        return _join_words(self._get_unemitted_cumulative_text() or "", decoded_suffix)

    def _get_unemitted_cumulative_text(self) -> Optional[str]:
        """Return cumulative text still held back by rollback."""
        confirmed_words = self.confirmed_text.split()
        latest_words = self.latest_text.split()
        common_count = _normalized_word_prefix_len(confirmed_words, latest_words)
        if common_count < len(confirmed_words):
            return None
        return " ".join(latest_words[common_count:])

    def reconcile_decoder_suffix(
        self, decoded_suffix: str, *, is_last: bool = False, holdback_words: int
    ) -> DecoderSuffixUpdate:
        """Reconcile a decode with the pending suffix without mutating state."""
        if self.emitted_text and (not self.pending_suffix or is_last):
            decoded_suffix, _ = _trim_word_overlap(self.emitted_text, decoded_suffix)
        previous_suffix = self.pending_suffix

        if is_last:
            return DecoderSuffixUpdate(
                delta=decoded_suffix or previous_suffix, pending_suffix=""
            )
        if not decoded_suffix:
            return DecoderSuffixUpdate(delta="", pending_suffix=None)
        if not previous_suffix:
            return DecoderSuffixUpdate(delta="", pending_suffix=decoded_suffix)

        if is_cjk_no_whitespace(previous_suffix) or is_cjk_no_whitespace(
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
        # Keep the acoustic tail out of the decoder prefix. A premature
        # sentence end there can make the next request stop before newly
        # appended audio, while the retained audio can safely recover it.
        emit_count = max(
            0,
            _normalized_word_prefix_len(previous_words, decoded_words) - holdback_words,
        )
        return DecoderSuffixUpdate(
            delta=" ".join(decoded_words[:emit_count]),
            pending_suffix=" ".join(decoded_words[emit_count:]),
        )

    def trim_decoder_prefix_echo(self, decoded_suffix: str, decoder_prefix: str) -> str:
        """Remove an implausibly large echo of the exact requested prefix."""
        decoded_words = decoded_suffix.split()
        prefix_words = decoder_prefix.split()
        max_words_for_chunk = max(24, int(self.chunk_size_sec * 16))
        if len(prefix_words) < max_words_for_chunk or len(decoded_words) < len(
            prefix_words
        ):
            return decoded_suffix
        if _normalized_word_prefix_len(prefix_words, decoded_words) != len(
            prefix_words
        ):
            return decoded_suffix
        return " ".join(decoded_words[len(prefix_words) :])

    def commit_decoder_suffix_update(
        self, update: DecoderSuffixUpdate, *, is_last: bool
    ) -> str:
        """Commit a computed update after the request mode is accepted."""
        self.decode_count += 1
        if update.pending_suffix is None:
            return ""
        delta = self._append_emitted_text(update.delta)
        self._set_pending_suffix("" if is_last else update.pending_suffix)
        return delta

    def flush_pending_decoder_suffix(self) -> str:
        """Emit the pending suffix at item end: no further decode will confirm it."""
        delta = self._append_emitted_text(self.pending_suffix)
        self._set_pending_suffix("")
        return delta

    def _set_pending_suffix(self, decoded_suffix: str) -> None:
        """Record the latest unconfirmed suffix decoded after emitted_text."""
        self.pending_suffix = decoded_suffix
        self.latest_text = _join_words(self.emitted_text, decoded_suffix)
        self.confirmed_text = self.emitted_text

    # --- Shared emit and trim helpers ---

    def _append_emitted_text(self, delta: str) -> str:
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
                if _normalize_overlap_word(delta_words[i]) != _normalize_overlap_word(
                    emitted_words[i]
                ):
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
    return _PUNCT_WS_RE.sub(r"\1", text).strip()


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def is_cjk_char(c: str) -> bool:
    """CJK-context character that takes no inter-word space."""
    cp = ord(c)
    if 0xFFA0 <= cp <= 0xFFDC:  # halfwidth Hangul jamo -- Korean is space-delimited
        return False
    return (
        0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation (，。、《》「」…)
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth/Fullwidth Forms (incl. fullwidth ASCII)
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
    text = text.strip()
    return (
        bool(text)
        and not any(c.isspace() for c in text)
        and any(_is_cjk_script_char(c) for c in text)
    )


def has_no_word_boundaries(text: str) -> bool:
    """Whether word-based suffix reconciliation is unsafe for this text."""
    text = text.strip()
    return (
        bool(text)
        and not any(c.isspace() for c in text)
        and any(
            _is_cjk_script_char(c) or 0x0E00 <= ord(c) <= 0x0E7F for c in text  # Thai
        )
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


def _normalize_overlap_word(word: str) -> str:
    """Normalize a word so recasing and edge-punctuation drift between decodes
    cannot break overlap matching."""
    word = unicodedata.normalize("NFKC", word)
    lo, hi = 0, len(word)
    while lo < hi and unicodedata.category(word[lo])[0] == "P":
        lo += 1
    while hi > lo and unicodedata.category(word[hi - 1])[0] == "P":
        hi -= 1
    return word[lo:hi].lower()


def _normalized_word_prefix_len(left_words: List[str], right_words: List[str]) -> int:
    """Common prefix robust to recasing and edge punctuation drift."""
    count = 0
    for lw, rw in zip(left_words, right_words):
        if _normalize_overlap_word(lw) != _normalize_overlap_word(rw):
            break
        count += 1
    return count


def _trim_word_overlap(emitted_text: str, decoded_text: str) -> "tuple[str, bool]":
    """Trim only a normalized decoded prefix matching the emitted tail."""
    decoded_words = decoded_text.split()
    if not decoded_words:
        return decoded_text, False
    emitted_tail = emitted_text.rsplit(maxsplit=len(decoded_words))[
        -len(decoded_words) :
    ]
    if not emitted_tail:
        return decoded_text, False
    emitted_tail_norm = [_normalize_overlap_word(w) for w in emitted_tail]
    decoded_norm = [_normalize_overlap_word(w) for w in decoded_words]
    max_overlap = min(len(emitted_tail_norm), len(decoded_norm))

    cut = 0
    for length in range(max_overlap, 0, -1):
        if emitted_tail_norm[-length:] == decoded_norm[:length] and any(
            emitted_tail_norm[-length:]
        ):
            cut = length
            break
    if cut == 0:
        return decoded_text, False
    return " ".join(decoded_words[cut:]), True


async def generate_asr_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    prompt: str,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[GeneratedTranscript]:
    """Run one stateless backend request and return text with its stop reason."""
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

    finish_reason = ret.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    return GeneratedTranscript(
        text=normalize_whitespace(adapter.postprocess_text(ret.get("text", ""))),
        finish_reason=finish_reason,
    )


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
    result = await generate_asr_transcript(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        audio_data=audio_data,
        sampling_params=sampling_params,
        prompt=prompt,
        raw_request=raw_request,
        routing_key=routing_key,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    return None if result is None else result.text


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
        prompt=adapter.prompt_template + state.get_cumulative_prompt_prefix(),
        raw_request=raw_request,
        routing_key=routing_key,
    )
    if text is None:
        return ""

    return state.reconcile_cumulative_transcript(text, is_last=is_last)
