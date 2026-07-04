import asyncio
import io
import logging
import re
import unicodedata
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


# Treat unmatched sliced overlap as unsafe only when the overlap has clear speech.
# Keep this higher than the near-silence floor so quiet/noise-only overlap does
# not block new text forever.
_OVERLAP_VOICE_RMS = 0.02


def _overlap_has_voice(
    samples: Any, sample_rate: Optional[int], overlap_seconds: float
) -> bool:
    if not isinstance(samples, np.ndarray) or not sample_rate or overlap_seconds <= 0:
        return False
    n = int(overlap_seconds * sample_rate)
    if n <= 0:
        return False
    region = samples[:n]
    if region.size == 0:
        return False
    rms = float(np.sqrt(np.mean(region.astype(np.float32) ** 2)))
    return rms >= _OVERLAP_VOICE_RMS


def _apply_sliced_dedupe(
    committed_text: str,
    text: str,
    samples: Any,
    sample_rate: Optional[int],
    overlap_seconds: float,
) -> "tuple[str, bool]":
    """Return deduped text and whether the sliced overlap was verified."""
    if not committed_text or not text:
        return text, True
    deduped, matched = _dedupe_by_word(committed_text, text)
    if matched:
        return deduped, True
    if _overlap_has_voice(samples, sample_rate, overlap_seconds):
        return deduped, False
    return deduped, True


_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


# Text reconciliation only: no audio buffer, GPU state, or scheduler state lives
# here. Cumulative requests emit a stable prefix with word/char rollback. Sliced
# realtime requests first trim exact overlap text; if the voiced overlap cannot
# be verified, the caller can defer the slice without mutating transcript state.
@dataclass
class StreamingASRState:
    """Chunked ASR transcript state with word or char rollback."""

    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    confirmed_text: str = ""
    # Already emitted text; cumulative prompt and sliced dedupe both depend on it.
    emitted_text: str = ""
    full_transcript: str = ""
    chunk_index: int = 0

    def get_prefix_text(self) -> str:
        if self.chunk_index < self.unfixed_chunk_num or not self.emitted_text:
            return ""
        # Word overlap is unsafe for no-whitespace CJK; keep that path cumulative.
        if _is_cjk_no_whitespace(self.emitted_text):
            return ""
        return self.emitted_text

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
        # Ignore trailing sentence enders when aligning: a sliced boundary can
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

    def _trim_large_cumulative_prompt_echo(self, delta: str) -> str:
        """Drop obvious full-prefix echoes from cumulative chunked ASR."""
        if not delta or not self.emitted_text:
            return delta

        delta_words = delta.split()
        emitted_words = self.emitted_text.split()
        # ~16 words/sec outpaces real speech, so a delta this long that also
        # prefix-matches emitted text is a prompt echo, not new audio content.
        max_words_for_chunk = max(24, int(self.chunk_size_sec * 16))
        if len(delta_words) <= max_words_for_chunk or not emitted_words:
            return delta

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

    def update(self, new_transcript: str, *, cumulative: bool = True) -> str:
        if _is_cjk_no_whitespace(new_transcript):
            return self._update_chars(new_transcript, cumulative=cumulative)

        old_confirmed = self.confirmed_text
        words = new_transcript.split()
        holdback = self.unfixed_token_num if cumulative else 0
        if holdback > 0:
            if len(words) > holdback:
                self.confirmed_text = " ".join(words[:-holdback])
            else:
                self.confirmed_text = ""
        else:
            self.confirmed_text = new_transcript
        self.full_transcript = new_transcript
        self.chunk_index += 1
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        if cumulative:
            common_count = _norm_common_prefix_len(old_words, new_words)
        else:
            common_count = 0
        delta = " ".join(new_words[common_count:])
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        if cumulative:
            delta = self._trim_large_cumulative_prompt_echo(delta)
        return self._record_emit(delta)

    def _update_chars(self, new_transcript: str, *, cumulative: bool = True) -> str:
        """Use character rollback when whitespace cannot define stable words."""
        old_confirmed = self.confirmed_text
        # Sliced windows are final on arrival (like the word path): no holdback,
        # and no prefix diff against the unrelated previous window's text.
        holdback = max(0, self.unfixed_token_num) if cumulative else 0
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
        self.confirmed_text = new_transcript[:cut]
        self.full_transcript = new_transcript
        self.chunk_index += 1

        common_count = (
            _common_prefix_len(old_confirmed, self.confirmed_text) if cumulative else 0
        )
        delta = self.confirmed_text[common_count:]
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        return self._record_emit(delta)

    def finalize(self, *, cumulative: bool = True) -> str:
        if _is_cjk_no_whitespace(self.full_transcript):
            old_confirmed = self.confirmed_text
            self.confirmed_text = self.full_transcript
            common_count = (
                _common_prefix_len(old_confirmed, self.full_transcript)
                if cumulative
                else 0
            )
            delta = self.full_transcript[common_count:]
            if common_count == 0:
                delta = self._trim_cjk_emitted_overlap(delta)
            return self._record_emit(delta)

        confirmed_words = self.confirmed_text.split()
        all_words = self.full_transcript.split()
        if cumulative:
            common_count = _norm_common_prefix_len(confirmed_words, all_words)
        else:
            common_count = 0
        self.confirmed_text = self.full_transcript
        delta = " ".join(all_words[common_count:])
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
        if cumulative:
            delta = self._trim_large_cumulative_prompt_echo(delta)
        return self._record_emit(delta)


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


def _is_word_char(c: str) -> bool:
    return c.isalnum() and not is_cjk_char(c)


def _is_cjk_no_whitespace(text: str) -> bool:
    return (
        bool(text)
        and not any(c.isspace() for c in text)
        and any(is_cjk_char(c) for c in text)
    )


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for lc, rc in zip(left, right):
        if lc != rc:
            break
        count += 1
    return count


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


def _dedupe_norm(word: str) -> str:
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
    sample_rate: Optional[int] = None,
    overlap_seconds: float = 0.0,
    defer_if_unverified: bool = False,
    verified_out: Optional[Dict[str, Any]] = None,
) -> str:
    """Run one ASR request for HTTP chunking or realtime WS."""
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
        text, overlap_verified = _apply_sliced_dedupe(
            dedupe_against, text, audio_data, sample_rate, overlap_seconds
        )
        if verified_out is not None:
            verified_out["verified"] = overlap_verified
        if defer_if_unverified and not overlap_verified:
            return ""

    if is_last:
        state.full_transcript = text
        return state.finalize(cumulative=dedupe_against is None)
    return state.update(text, cumulative=dedupe_against is None)
