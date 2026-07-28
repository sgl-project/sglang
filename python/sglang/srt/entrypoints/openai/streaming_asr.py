import asyncio
import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    # Maximum chunks of audio sent per inference; 0 = unbounded (each
    # inference re-encodes the whole utterance, so per-chunk cost grows
    # linearly and total cost quadratically with utterance length). When
    # the buffered audio exceeds this, the window rolls: audio whose text
    # is already emitted is dropped and the transcript continues from the
    # ``emitted_text`` prompt prefix.
    window_chunk_num: int = 0
    confirmed_text: str = ""
    # Monotonic accumulator; used as prompt prefix so the model sees a
    # natural continuation point, not the rolled-back ``confirmed_text``.
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
        if self.confirmed_text.startswith(old_confirmed):
            return self._record_emit(self.confirmed_text[len(old_confirmed) :].strip())
        # Model revised earlier text, use word level common prefix to avoid
        # re-emitting already-sent content and cutting mid-word.
        old_words = old_confirmed.split()
        new_words = self.confirmed_text.split()
        common_count = 0
        for ow, nw in zip(old_words, new_words):
            if ow != nw:
                break
            common_count += 1
        return self._record_emit(" ".join(new_words[common_count:]))

    def start_new_window(self) -> None:
        """Reset the continuation frame after an audio-window roll.

        ``emitted_text`` (the prompt prefix) carries all confirmed text
        across the roll; ``confirmed_text``/``full_transcript`` are
        relative to the model's continuation output, which restarts when
        the audio the model sees changes.
        """
        self.confirmed_text = ""
        self.full_transcript = ""

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


def compute_window_drop(
    buffered: int,
    inferred: int,
    chunk_size: int,
    state: StreamingASRState,
) -> int:
    """How much audio to drop from the buffer head when the window rolls.

    Unit-agnostic (callers pass bytes or samples; ``chunk_size`` is one
    chunk in the same unit). Only audio that has (a) been through at
    least one inference pass and (b) sits outside the unfixed rollback
    zone is safe to drop — its text is already in ``emitted_text`` and
    conditions the next inference as the prompt prefix. Never-inferred
    audio (e.g. one giant append before the first inference) is never
    dropped, so no content is lost regardless of append cadence.

    Returns 0 when windowing is disabled or the buffer fits the window.
    """
    if not state.window_chunk_num:
        return 0
    if buffered <= state.window_chunk_num * chunk_size:
        return 0
    # Keep the unfixed zone plus the newest (not yet inferred) chunk.
    keep = (state.unfixed_chunk_num + 1) * chunk_size
    safe = inferred - state.unfixed_chunk_num * chunk_size
    return max(0, min(buffered - keep, safe))


def decode_audio_mono(audio_data: bytes):
    """Decode an audio file to (float32 mono ndarray, sample_rate)."""
    if not audio_data:
        raise ValueError("audio_data is empty")
    audio_file = io.BytesIO(audio_data)
    try:
        data, sample_rate = sf.read(audio_file, dtype="float32")
    except sf.LibsndfileError as e:
        raise ValueError(f"failed to decode audio: {e}") from e
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data, sample_rate


def encode_wav(data, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


def split_audio_chunks(audio_data: bytes, chunk_size_sec: float) -> List[bytes]:
    if chunk_size_sec <= 0:
        raise ValueError(f"chunk_size_sec must be positive, got {chunk_size_sec}")
    data, sample_rate = decode_audio_mono(audio_data)
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    total_samples = len(data)
    chunks = []
    for end in range(
        chunk_size_samples, total_samples + chunk_size_samples, chunk_size_samples
    ):
        end = min(end, total_samples)
        chunks.append(encode_wav(data[:end], sample_rate))
    return chunks


def normalize_whitespace(text: str) -> str:
    return _PUNCT_WS_RE.sub(r"\1", text)


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def _is_cjk(c: str) -> bool:
    """Whether char is a CJK-context glyph that doesn't take inter-word
    spaces — ideographs, Japanese kana, CJK punctuation, fullwidth forms.
    Excludes Hangul / Devanagari / Arabic etc., which are non-ASCII but
    space-separated and need the normal boundary space."""
    cp = ord(c)
    return (
        0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation (，。、《》「」…)
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth & Fullwidth Forms (fullwidth ASCII)
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


async def process_asr_chunk(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    state: StreamingASRState,
    audio_data: bytes,
    sampling_params: Dict[str, Any],
    is_last: bool,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
) -> str:
    """Run inference on one audio chunk. Shared by the HTTP and WebSocket paths."""
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

    if is_last:
        state.full_transcript = text
        return state.finalize()
    return state.update(text)
