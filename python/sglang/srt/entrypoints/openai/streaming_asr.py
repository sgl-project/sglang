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


# Overlap region RMS above which we treat the re-sent audio as containing real
# speech. Used only to decide dedupe *safety*, never to skip/drop audio.
_OVERLAP_VOICE_RMS = 0.02


def _overlap_has_voice(
    samples: Any, sample_rate: Optional[int], overlap_seconds: float
) -> bool:
    """Whether the leading overlap region of a slice carries voiced speech.

    Used to disambiguate a ``cut == 0`` exact-prefix result: if the overlap audio
    was silent there was nothing to repeat, so a non-matching candidate is
    genuinely new (safe to emit); if it was voiced, a non-match means the model
    reworded the overlap and emitting verbatim would duplicate (unsafe). Biased
    toward "no voice" (safe/emit) when the audio payload can't be inspected."""
    if (
        not isinstance(samples, np.ndarray)
        or not sample_rate
        or overlap_seconds <= 0
    ):
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
    """Conservative sliced-overlap dedupe.

    Returns ``(deduped_text, overlap_verified)``. Trims only a verbatim
    (normalized) copy of the committed tail from the start of the candidate --
    never guesses, and never deletes an unmatched leading word.

    ``overlap_verified`` reports whether the trim is provably safe:
      * a verbatim overlap prefix was matched and trimmed -> safe;
      * no match, but the overlap audio was silent        -> safe (new content);
      * no match over a *voiced* overlap                  -> UNSAFE: the model
        reworded the re-sent audio, so the caller defers instead of emitting a
        duplicate guess.
    """
    if not committed_text or not text:
        return text, True
    deduped, matched = _dedupe_by_word(committed_text, text)
    if matched:
        return deduped, True
    if _overlap_has_voice(samples, sample_rate, overlap_seconds):
        return deduped, False
    return deduped, True


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

    def _trim_large_cumulative_prompt_echo(self, delta: str) -> str:
        """Drop obvious full-prefix echoes from cumulative chunked ASR.

        HTTP SSE chunked ASR prompts the next cumulative audio snapshot with
        already emitted text. On repetitive audio, Qwen3-ASR can copy that
        prompt text back before the new words. A short chunk should not add
        dozens of words at once, so this only trims unusually large deltas that
        start with a long normalized copy of the emitted prefix.
        """
        if not delta or not self.emitted_text:
            return delta

        delta_words = delta.split()
        emitted_words = self.emitted_text.split()
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
            return self._update_chars(new_transcript)

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
            # spurious. No token-count holdback here: the sliced window is
            # already acoustically bounded, and holding a word back risks it
            # being deferred past the next (time-based) overlap and dropped.
            common_count = 0
        delta = " ".join(new_words[common_count:])
        if common_count == 0:
            # Full re-emit: guard against re-sending an already-emitted CJK
            # prefix when the char<->word encoding flipped.
            delta = self._trim_cjk_emitted_overlap(delta)
        if cumulative:
            delta = self._trim_large_cumulative_prompt_echo(delta)
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


def _dedupe_by_word(committed_text: str, candidate_out: str) -> "tuple[str, bool]":
    """Trim a verbatim overlap prefix repeated by a sliced audio request.

    Returns ``(trimmed_text, matched)`` where ``matched`` is True iff a non-empty
    normalized overlap prefix was found and trimmed. ``matched == False`` means
    the candidate did not verifiably start on the committed tail (the model
    reworded it, or there was no overlap): nothing is deleted, and the caller
    decides whether emitting verbatim is safe.

    The model re-transcribes the committed tail from the left-overlap audio, so
    the candidate can begin with a copy of the committed tail. Trim only the
    longest *normalized prefix* of the candidate that exactly matches a suffix of
    the committed tail: the candidate must START on the overlap, so genuine new
    speech -- which always follows the overlap in a sliced request -- is never
    deleted, and unmatched leading words are never dropped.

    This is deliberately conservative. Re-worded overlap ("ladled" -> "laid") is
    left as-is rather than guessed at: a reworded copy can't be told from genuinely
    repeated speech by text alone, so ``matched`` stays False and the caller keeps
    the audio and defers instead of guessing. When nothing overlaps, the candidate
    is returned verbatim (never re-tokenized or rewritten).
    """
    candidate_words = candidate_out.split()
    if not candidate_words:
        return candidate_out, False
    # Only the last len(candidate_words) committed words can overlap, so rsplit
    # the tail instead of tokenizing the whole (growing) committed transcript.
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
        # No overlap trimmed -- return the model output untouched.
        return candidate_out, False
    return " ".join(candidate_words[cut:]), True


def dedupe_overlap(committed_text: str, candidate_out: str) -> str:
    """Word-level text dedupe for sliced ASR overlap.

    Sliced realtime requests intentionally resend a little old audio for
    continuity. That overlap can make the model repeat already-emitted words,
    so trim the repeated prefix before updating streaming state.
    Trims words at the start of ``candidate_out`` that re-transcribe
    ``committed_text``'s tail. It matches whitespace-delimited words only and only
    a verbatim (normalized) prefix; when the overlap can't be matched it is left
    untouched rather than guessed at.

    CJK has no inter-word spaces, so those streams stay on the cumulative path."""
    if not committed_text or not candidate_out:
        return candidate_out
    return _dedupe_by_word(committed_text, candidate_out)[0]


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
    """Run inference on one audio chunk. Shared by the HTTP and WS paths.

    ``audio_data`` accepts WAV bytes or pre-decoded float samples.
    ``prompt`` overrides the default ``adapter.prompt_template + state.get_prefix_text()``.
    ``dedupe_against`` triggers the sliced overlap dedupe on raw model output
    before ``state`` ingests it. Realtime sliced calls pass both: the bare prompt
    avoids text-prefix injection, and the dedupe target removes text repeated from
    the acoustic overlap.

    ``defer_if_unverified``: when the overlap trim can't be proven safe, return ""
    WITHOUT ingesting into ``state`` so a later slice or the commit re-covers the
    audio instead of emitting a duplicate guess. ``verified_out``, if given,
    receives the verdict under key ``"verified"``.
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
        text, overlap_verified = _apply_sliced_dedupe(
            dedupe_against, text, audio_data, sample_rate, overlap_seconds
        )
        if verified_out is not None:
            verified_out["verified"] = overlap_verified
        if defer_if_unverified and not overlap_verified:
            # Unsafe boundary: do not ingest into streaming state. The caller
            # leaves the slice anchor/PCM intact so a later slice or the commit
            # re-covers this audio instead of emitting a duplicate guess.
            return ""

    if is_last:
        state.full_transcript = text
        return state.finalize(cumulative=dedupe_against is None)
    return state.update(text, cumulative=dedupe_against is None)
