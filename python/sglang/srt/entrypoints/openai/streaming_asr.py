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


# Collapse whitespace before punctuation so batched-inference token
# boundary jitter (" ," vs ",") doesn't leak into deltas. Covers both
# ASCII punctuation and the CJK / fullwidth equivalents.
_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


@dataclass
class StreamingASRState:
    """State for chunk-based streaming ASR with prefix rollback.

    Chunked ASR outputs can revise the newest text when more audio arrives.
    Keep only a stable prefix as confirmed, emit deltas from that prefix, and
    hold back the latest tokens/chars so later chunks can correct them.
    Parameters are model-specific and should be provided via the
    adapter's ``chunked_streaming_config``.

    CJK-style no-whitespace text uses character rollback so partial deltas can
    advance without whitespace-confirmed words.
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
        # Sliced overlap dedupe relies on whitespace-delimited words; CJK
        # char-rollback streams stay cumulative instead of using word overlap.
        if _is_cjk_no_whitespace(self.emitted_text):
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

    def _trim_cjk_emitted_overlap(self, delta: str) -> str:
        """Drop a leading all-CJK run of ``delta`` that duplicates the tail of
        ``emitted_text``.

        The whitespace-word path (``str.split``) and the CJK char path store
        ``confirmed_text`` in incompatible encodings (space-delimited words vs a
        spaceless char run). When a stream flips between them -- code-switching
        CJK<->Latin, or a slicing item disarming back to the cumulative path --
        the reconciler yields ``common_count == 0`` and re-emits an already-sent
        CJK prefix. Only ever trims an all-CJK overlap ending on a CJK/space
        boundary, so Latin word extension ("world"->"worldly") and legitimate
        repeats (reconciled with ``common_count > 0``) are never touched.
        """
        if not delta or not self.emitted_text:
            return delta
        max_k = min(len(delta), len(self.emitted_text))
        for k in range(max_k, 0, -1):
            if self.emitted_text[-k:] != delta[:k]:
                continue
            # Require the overlap to be all-CJK. That alone guarantees the cut is
            # a valid boundary (CJK chars are self-delimiting, so delta[k] -- even
            # a glued Latin token like "abc" in "你好abc" -- starts a fresh unit),
            # while a Latin overlap ("cat"/"cats") is rejected and never trimmed.
            if all(_is_cjk(c) for c in delta[:k]):
                return delta[k:].lstrip()
        return delta

    def update(self, new_transcript: str, *, cumulative: bool = True) -> str:
        if _is_cjk_no_whitespace(new_transcript):
            return self._update_chars(new_transcript)

        old_confirmed = self.confirmed_text
        words = new_transcript.split()
        if len(words) > self.unfixed_token_num:
            self.confirmed_text = " ".join(words[: -self.unfixed_token_num])
        else:
            self.confirmed_text = ""
        self.full_transcript = new_transcript
        self.chunk_index += 1
        # Word-level common prefix, not char-level startswith: startswith
        # sliced mid-word when a confirmed word was extended ("world" ->
        # "worldly" emitted "ly").
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        if cumulative:
            # Normalized compare so recasing/repunctuation of an already-emitted
            # word (He->he, But->but, .->,) does not reset the common prefix.
            # Emit from the first genuinely-different word: this can re-send an
            # early word the model revised, but never DROPS a new one (all new
            # words are at the tail, at index >= common_count).
            common_count = _norm_common_prefix_len(old_words, new_words)
        else:
            # Sliced path: each slice is a disjoint audio window and
            # dedupe_overlap already removed the emitted overlap, so a word
            # prefix match against the previous slice's confirmed_text is
            # spurious -- it would drop held-back words that merely share a
            # leading function word ("the"/"a"/"I"). Emit all of the new content.
            common_count = 0
        delta = " ".join(new_words[common_count:])
        if common_count == 0:
            # Full re-emit: guard against re-sending an already-emitted CJK
            # prefix when the char<->word encoding flipped.
            delta = self._trim_cjk_emitted_overlap(delta)
        return self._record_emit(delta)

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
        # Never split a Latin/alnum run embedded in CJK text. Back the cut up to
        # the nearest boundary between two word characters.
        while (
            0 < cut < len(new_transcript)
            and _is_word_char(new_transcript[cut - 1])
            and _is_word_char(new_transcript[cut])
        ):
            cut -= 1
        self.confirmed_text = new_transcript[:cut]
        self.full_transcript = new_transcript
        self.chunk_index += 1

        # Character common prefix: emit from the first differing char so a
        # rewritten early character never drops the new characters after it.
        # (No case folding -- CJK text has no case.)
        common_count = _common_prefix_len(old_confirmed, self.confirmed_text)
        delta = self.confirmed_text[common_count:]
        if common_count == 0:
            # Full re-emit (e.g. a sliced->cumulative flip re-transcribed the
            # tail window): drop an already-emitted CJK prefix.
            delta = self._trim_cjk_emitted_overlap(delta)
        return self._record_emit(delta)

    def finalize(self, *, cumulative: bool = True) -> str:
        if _is_cjk_no_whitespace(self.full_transcript):
            old_confirmed = self.confirmed_text
            self.confirmed_text = self.full_transcript
            # Emit from the first differing char; never drop the finalized tail.
            common_count = _common_prefix_len(old_confirmed, self.full_transcript)
            delta = self.full_transcript[common_count:]
            if common_count == 0:
                delta = self._trim_cjk_emitted_overlap(delta)
            return self._record_emit(delta)

        confirmed_words = self.confirmed_text.split()
        all_words = self.full_transcript.split()
        # Word-level common prefix (normalized on the cumulative path) absorbs
        # casing/punctuation differences between the last chunk and the final
        # transcription without dropping any newly-finalized word.
        if cumulative:
            common_count = _norm_common_prefix_len(confirmed_words, all_words)
        else:
            # Sliced path: dedupe_overlap is the only intended dedup (see update).
            common_count = 0
        self.confirmed_text = self.full_transcript
        delta = " ".join(all_words[common_count:])
        if common_count == 0:
            delta = self._trim_cjk_emitted_overlap(delta)
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


def _is_cjk(c: str) -> bool:
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
    """A Latin/alphanumeric character that forms a contiguous word (not CJK,
    which is self-delimiting). Used to avoid splitting an embedded word at the
    char-rollback holdback boundary."""
    return c.isalnum() and not _is_cjk(c)


def _is_cjk_no_whitespace(text: str) -> bool:
    return (
        bool(text)
        and not any(c.isspace() for c in text)
        and any(_is_cjk(c) for c in text)
    )


def _common_prefix_len(left: str, right: str) -> int:
    count = 0
    for lc, rc in zip(left, right):
        if lc != rc:
            break
        count += 1
    return count


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
    if _is_cjk(prev[-1]) and _is_cjk(cur[0]):
        return False
    return True


def _dedupe_norm(word: str) -> str:
    """Normalize one whitespace-delimited token for overlap matching.

    NFKC handles fullwidth/compatibility forms, lowercasing handles casing
    drift, and edge-punctuation stripping lets "word," match "word".
    """
    word = unicodedata.normalize("NFKC", word)
    lo, hi = 0, len(word)
    while lo < hi and unicodedata.category(word[lo])[0] == "P":
        lo += 1
    while hi > lo and unicodedata.category(word[hi - 1])[0] == "P":
        hi -= 1
    return word[lo:hi].lower()


def _norm_common_prefix_len(left_words: List[str], right_words: List[str]) -> int:
    """Word-level common-prefix length using normalized-token comparison.

    Normalization (casefold + edge-punctuation strip, via ``_dedupe_norm``)
    means recasing or repunctuation of an already-emitted word (``He``->``he``,
    ``.``->``,``) does not reset the prefix. Cumulative snapshots therefore stay
    append-only and never drop a genuinely-new word from the delta.
    """
    count = 0
    for lw, rw in zip(left_words, right_words):
        if _dedupe_norm(lw) != _dedupe_norm(rw):
            break
        count += 1
    return count


def _dedupe_by_word(committed_text: str, candidate_out: str) -> str:
    """Remove text repeated because a sliced audio request includes overlap.

    The model may re-transcribe the committed tail from the left-overlap audio.
    Compare the committed suffix with the candidate prefix, then drop the
    longest word-level match from the candidate.
    """
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
        # would otherwise match.
        if not any(candidate_norm[:overlap]):
            continue
        return " ".join(candidate_words[overlap:])
    return candidate_out


def dedupe_overlap(committed_text: str, candidate_out: str) -> str:
    """Word-level text dedupe for sliced ASR overlap.

    Sliced realtime requests intentionally resend a little old audio for
    continuity. That overlap can make the model repeat already-emitted words,
    so trim the repeated prefix before updating streaming state.
    Trims words at the start of ``candidate_out`` that re-transcribe
    ``committed_text``'s tail. It matches whitespace-delimited words only;
    repeated-speech edge cases need timestamp/token alignment for exact handling.

    CJK has no inter-word spaces, so those streams stay on the cumulative path."""
    if not committed_text or not candidate_out:
        return candidate_out
    return _dedupe_by_word(committed_text, candidate_out)


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
    Realtime sliced calls pass both: the bare prompt avoids text-prefix injection,
    and the dedupe target removes text repeated from the acoustic overlap.
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
        return state.finalize(cumulative=dedupe_against is None)
    return state.update(text, cumulative=dedupe_against is None)
