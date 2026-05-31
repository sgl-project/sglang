import asyncio
import io
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import regex
import soundfile as sf
from fastapi import Request

from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Collapse whitespace before punctuation so batched-inference token
# boundary jitter (" ," vs ",") doesn't leak into deltas. Covers both
# ASCII punctuation and the CJK / fullwidth equivalents.
_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


@dataclass
class StreamingASRState:
    """State for chunk-based streaming ASR with prefix rollback.

    Parameters are model-specific and should be provided via the
    adapter's ``chunked_streaming_config``.

    Known limitation: rollback uses str.split() which is ineffective
    for CJK languages (no whitespace between words).
    TODO: implement token-level rollback to handle all languages
    correctly.
    """

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    confirmed_text: str = ""
    # Monotonic accumulator. Used as the prompt prefix on cumulative paths and
    # as the dedupe prefix on the slicing path.
    emitted_text: str = ""
    full_transcript: str = ""
    chunk_index: int = 0

    def get_prefix_text(self) -> str:
        if self.chunk_index < self.unfixed_chunk_num or not self.emitted_text:
            return ""
        return self.emitted_text

    def _record_emit(self, delta: str) -> str:
        if delta:
            if self.emitted_text:
                # needs_space avoids a space between adjacent CJK characters;
                # this accumulator feeds the prompt prefix and the dedupe target.
                sep = " " if needs_space(self.emitted_text, delta) else ""
                self.emitted_text = f"{self.emitted_text}{sep}{delta}".strip()
            else:
                self.emitted_text = delta
        return delta

    def update(self, new_transcript: str) -> str:
        old_confirmed = self.confirmed_text
        words = new_transcript.split()
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = " ".join(words[: -self.unfixed_token_num])
        else:
            self.confirmed_text = ""
        self.full_transcript = new_transcript
        self.chunk_index += 1
        # Token-level common prefix, not char-level startswith: startswith
        # sliced mid-word when a confirmed token was extended ("world" ->
        # "worldly" emitted "ly").
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        common_count = 0
        for ow, nw in zip(old_words, new_words):
            if ow != nw:
                break
            common_count += 1
        return self._record_emit(" ".join(new_words[common_count:]))

    def finalize(self) -> str:
        confirmed_words = self.confirmed_text.split()
        all_words = self.full_transcript.split()
        # Use word level common prefix to handle punctuation differences
        # between intermediate chunks and the final full transcription.
        common_count = 0
        for cw, aw in zip(confirmed_words, all_words):
            if cw != aw:
                break
            common_count += 1
        self.confirmed_text = self.full_transcript
        if common_count == 0 and confirmed_words and all_words:
            return self._record_emit(self.full_transcript)
        return self._record_emit(" ".join(all_words[common_count:]))


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


# Two predicates: dedup = genuine CJK script only; spacing also includes the
# fullwidth forms. scx (not Script) so the kana marks ー/・ count; Hangul excluded
# (Korean uses spaces).
_HALFWIDTH_HANGUL = chr(0xFFA0) + "-" + chr(0xFFDC)

# Dedup set: &&\p{P} adds fullwidth punctuation but not fullwidth ASCII/digits.
_CJK_DEDUPE_RE = regex.compile(
    "(?V1)["
    r"\p{Han}\p{scx=Hiragana}\p{scx=Katakana}\p{Bopomofo}"
    r"\p{Block=CJK_Symbols_and_Punctuation}"
    r"[\p{Block=Halfwidth_and_Fullwidth_Forms}&&\p{P}]]"
)

# Spacing set: dedup set + fullwidth ASCII/digits (wide glyphs take no space),
# minus the halfwidth Hangul jamo (U+FFA0-FFDC) the block would re-add.
_CJK_NO_SPACE_RE = regex.compile(
    "(?V1)["
    r"\p{Han}\p{scx=Hiragana}\p{scx=Katakana}\p{Bopomofo}"
    r"\p{Block=CJK_Symbols_and_Punctuation}"
    r"[\p{Block=Halfwidth_and_Fullwidth_Forms}--[" + _HALFWIDTH_HANGUL + "]]]"
)


def _is_cjk_no_space(c: str) -> bool:
    """No-inter-word-space char (CJK script + fullwidth forms); spacing only."""
    return bool(_CJK_NO_SPACE_RE.match(c))


def _is_cjk_dedupe(c: str) -> bool:
    """Genuine CJK script char eligible for char-level dedup (no fullwidth ASCII)."""
    return bool(_CJK_DEDUPE_RE.match(c))


def needs_space(prev: str, cur: str) -> bool:
    """Return whether a boundary space is needed between emitted deltas.

    Avoid spaces around punctuation and between adjacent CJK-context characters.
    Shared by the realtime WS and HTTP SSE chunked streaming paths.
    """
    if not prev or not cur:
        return False
    if prev[-1].isspace() or cur[0].isspace():
        return False
    if cur[0] in _NO_SPACE_BEFORE or prev[-1] in _NO_SPACE_AFTER:
        return False
    if _is_cjk_no_space(prev[-1]) and _is_cjk_no_space(cur[0]):
        return False
    return True


def _dedupe_norm(word: str) -> str:
    """Lowercase + NFKC-fold + strip edge punctuation (Unicode category P), so
    "dinner," == "dinner" and exotic marks (《》«» …) need no hand-listed set.
    Strips P only, not S, so "$5" / "3+4" keep their symbols."""
    word = unicodedata.normalize("NFKC", word)
    lo, hi = 0, len(word)
    while lo < hi and unicodedata.category(word[lo])[0] == "P":
        lo += 1
    while hi > lo and unicodedata.category(word[hi - 1])[0] == "P":
        hi -= 1
    return word[lo:hi].lower()


def _dedupe_by_word(committed_text: str, candidate_out: str) -> str:
    """Drop the longest prefix of ``candidate_out`` matching the suffix of
    ``committed_text`` word-for-word (case- and punctuation-insensitive)."""
    candidate_words = candidate_out.split()
    if not candidate_words:
        return candidate_out
    # Only the last len(candidate_words) committed words can overlap, so rsplit
    # the tail instead of tokenizing the whole (growing) committed transcript.
    committed_tail = committed_text.rsplit(maxsplit=len(candidate_words))[
        -len(candidate_words) :
    ]
    if not committed_tail:
        return candidate_out
    # Normalize the committed tail and candidate prefix once, then compare slices.
    max_overlap = min(len(committed_tail), len(candidate_words))
    committed_tail_norm = [_dedupe_norm(w) for w in committed_tail]
    candidate_norm = [_dedupe_norm(w) for w in candidate_words[:max_overlap]]
    # Longest overlap first; the first match wins.
    for overlap in range(max_overlap, 0, -1):
        if committed_tail_norm[-overlap:] != candidate_norm[:overlap]:
            continue
        # Skip all-punctuation overlaps: lone "@"/"#" both normalize to "" and
        # would match spuriously.
        if not any(candidate_norm[:overlap]):
            continue
        return " ".join(candidate_words[overlap:])
    return candidate_out


def _is_punctuation(c: str) -> bool:
    return unicodedata.category(c)[0] == "P"


def _get_leading_cjk_chars(text: str) -> List[str]:
    """Return the CJK characters at the start of ``text``, stopping at the first
    non-CJK character (leading whitespace is skipped)."""
    chars: List[str] = []
    for char in text.lstrip():
        if not _is_cjk_dedupe(char):
            break
        chars.append(char)
    return chars


def _get_trailing_cjk_chars(text: str) -> List[str]:
    """Return the CJK characters at the end of ``text``, stopping at the first
    non-CJK character (trailing whitespace is skipped)."""
    chars: List[str] = []
    for char in reversed(text.rstrip()):
        if not _is_cjk_dedupe(char):
            break
        chars.append(char)
    chars.reverse()
    return chars


def _dedupe_by_cjk_char(committed_text: str, candidate_out: str) -> str:
    """Drop the CJK characters at the start of ``candidate_out`` when they repeat
    the CJK characters at the end of ``committed_text``. Only those boundary
    characters are compared, so a non-CJK prefix ("today ") is never deleted to
    reach a later match."""
    lead = _get_leading_cjk_chars(candidate_out)
    tail = _get_trailing_cjk_chars(committed_text)
    if not lead or not tail:
        return candidate_out
    for overlap in range(min(len(lead), len(tail)), 0, -1):
        if tail[-overlap:] != lead[:overlap]:
            continue
        # Single-glyph matches collide too often for CJK letters; require >=2,
        # allow 1 only for punctuation.
        if overlap == 1 and not _is_punctuation(lead[0]):
            continue
        return candidate_out.lstrip()[overlap:].lstrip()
    return candidate_out


def dedupe_overlap(committed_text: str, candidate_out: str) -> str:
    """Trim words / CJK characters at the start of ``candidate_out`` that
    re-transcribe ``committed_text``'s tail. Matches by whole word first, then
    falls back to matching the leading/trailing CJK characters."""
    if not committed_text or not candidate_out:
        return candidate_out
    deduped = _dedupe_by_word(committed_text, candidate_out)
    if deduped != candidate_out:
        return deduped
    return _dedupe_by_cjk_char(committed_text, candidate_out)


async def process_asr_chunk(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    prompt: Optional[str] = None,
    dedupe_against: Optional[str] = None,
) -> str:
    """Run inference on one audio chunk. Shared by the HTTP and WS paths.

    ``audio_data`` accepts WAV bytes or pre-decoded float samples.
    ``prompt`` overrides the default ``adapter.prompt_template + state.get_prefix_text()``.
    ``dedupe_against`` triggers ``dedupe_overlap`` on raw model output before ``state`` ingests it.
    """
    if prompt is None:
        prompt = adapter.prompt_template + state.get_prefix_text()

    chunk_request = GenerateReqInput(
        text=prompt,
        audio_data=audio_data,
        sampling_params=sampling_params,
        stream=False,
        modalities=["audio"],
    )
    if routing_key is not None:
        chunk_request.routing_key = routing_key

    try:
        ret = None
        async for ret in tokenizer_manager.generate_request(chunk_request, raw_request):
            break
    except asyncio.CancelledError:
        raise
    except ValueError:
        logger.warning(
            "[streaming_asr] chunk %d failed", state.chunk_index, exc_info=True
        )
        raise

    if ret is None:
        logger.warning("[streaming_asr] empty response for chunk %d", state.chunk_index)
        return ""

    text = normalize_whitespace(adapter.postprocess_text(ret.get("text", "")))
    if dedupe_against is not None:
        text = dedupe_overlap(dedupe_against, text)

    if is_last:
        state.full_transcript = text
        return state.finalize()
    return state.update(text)
