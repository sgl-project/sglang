import asyncio
import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
            self.emitted_text = (
                f"{self.emitted_text} {delta}".strip() if self.emitted_text else delta
            )
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


def _is_cjk(c: str) -> bool:
    """CJK-context glyph that doesn't take inter-word spaces."""
    cp = ord(c)
    return (
        0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth & Fullwidth Forms
    )


def needs_space(prev: str, cur: str) -> bool:
    """Return whether a boundary space is needed between emitted deltas.

    Avoid spaces around punctuation and between adjacent CJK-context glyphs.
    Shared by the realtime WS and HTTP SSE chunked streaming paths.
    """
    if not prev or not cur:
        return False
    if prev[-1].isspace() or cur[0].isspace():
        return False
    if cur[0] in _NO_SPACE_BEFORE or prev[-1] in _NO_SPACE_AFTER:
        return False
    if _is_cjk(prev[-1]) and _is_cjk(cur[0]):
        return False
    return True


# Trailing punctuation stripped during dedupe match. Includes em dash
# (U+2014), hyphen-minus, and CJK fullwidth equivalents.
_DEDUPE_NORM_STRIP = ",.!?;:—-，。！？；：、"


def _dedupe_norm(word: str) -> str:
    """Lowercase + strip trailing punctuation for dedupe matching."""
    return word.strip(_DEDUPE_NORM_STRIP).lower()


def _dedupe_word_level(committed_text: str, candidate_out: str) -> str:
    """Drop the longest prefix of ``candidate_out`` matching the suffix of
    ``committed_text`` word-for-word (case- and punctuation-insensitive)."""
    candidate_words = candidate_out.split()
    if not candidate_words:
        return candidate_out
    committed_words = committed_text.split()
    if not committed_words:
        return candidate_out
    # The overlap is at most as long as the candidate, so only the last
    # `max_overlap` committed words can match. Normalize that committed tail and
    # the candidate prefix once, then compare list slices instead of
    # re-normalizing inside the loop.
    max_overlap = min(len(committed_words), len(candidate_words))
    committed_tail_norm = [_dedupe_norm(w) for w in committed_words[-max_overlap:]]
    candidate_norm = [_dedupe_norm(w) for w in candidate_words[:max_overlap]]
    # Longest overlap first; the first match wins.
    for overlap in range(max_overlap, 0, -1):
        if committed_tail_norm[-overlap:] == candidate_norm[:overlap]:
            return " ".join(candidate_words[overlap:])
    return candidate_out


def _find_kth_cjk_pos(text: str, k: int) -> Optional[int]:
    """Return index after the k-th CJK glyph in text, or None if text
    contains fewer than k CJK glyphs."""
    seen = 0
    for i, c in enumerate(text):
        if c.isspace() or not _is_cjk(c):
            continue
        seen += 1
        if seen == k:
            return i + 1
    return None


def _dedupe_cjk_char_level(committed_text: str, candidate_out: str) -> str:
    """Drop leading CJK glyphs of ``candidate_out`` matching the CJK-tail of
    ``committed_text``. Non-CJK glyphs are skipped during match, preserved
    in trimmed output."""
    candidate_chars = [c for c in candidate_out if not c.isspace() and _is_cjk(c)]
    if not candidate_chars:
        return candidate_out
    # The overlap is at most the candidate's CJK length, so collect only that
    # many CJK glyphs from the end of committed_text (scanning in reverse and
    # stopping early) instead of the whole history.
    max_overlap = len(candidate_chars)
    committed_tail_rev = []
    for c in reversed(committed_text):
        if c.isspace() or not _is_cjk(c):
            continue
        committed_tail_rev.append(c)
        if len(committed_tail_rev) >= max_overlap:
            break
    if not committed_tail_rev:
        return candidate_out
    committed_tail_chars = list(reversed(committed_tail_rev))
    # Longest overlap first; the first match wins.
    for overlap in range(len(committed_tail_chars), 0, -1):
        if committed_tail_chars[-overlap:] != candidate_chars[:overlap]:
            continue
        cut_pos = _find_kth_cjk_pos(candidate_out, overlap)
        if cut_pos is None:
            return ""
        return candidate_out[cut_pos:].lstrip()
    return candidate_out


def dedupe_overlap(committed_text: str, candidate_out: str) -> str:
    """Trim words/CJK glyphs at the start of ``candidate_out`` that
    re-transcribe ``committed_text``'s tail. Word-level first, CJK
    char-level fallback."""
    if not committed_text or not candidate_out:
        return candidate_out
    deduped = _dedupe_word_level(committed_text, candidate_out)
    if deduped != candidate_out:
        return deduped
    if any(_is_cjk(c) for c in committed_text) or any(
        _is_cjk(c) for c in candidate_out
    ):
        return _dedupe_cjk_char_level(committed_text, candidate_out)
    return candidate_out


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
