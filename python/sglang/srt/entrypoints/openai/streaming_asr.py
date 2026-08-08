"""Shared cumulative streaming ASR state and backend request helpers."""

import asyncio
import io
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import msgspec
import numpy as np
import soundfile as sf
from fastapi import Request

from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.mm_utils import hash_feature
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.multimodal.audio_encoder_windowing import (
        AudioEncoderWindowConfig,
    )

# Cumulative decodes can jitter only in whitespace before punctuation. Remove
# that formatting noise before comparing successive transcript prefixes.
_PUNCT_WS_RE = re.compile(r"\s+([,.;:!?，。！？；：、])")


class GeneratedTranscript(msgspec.Struct, frozen=True):
    """Normalized ASR text plus the backend stop reason."""

    text: str
    finish_reason: Optional[str]


def hash_audio_content(audio_data: Union[bytes, np.ndarray]) -> str:
    """Return the per-audio hex identity accepted by GenerateReqInput."""
    return f"{hash_feature(audio_data):016x}"


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
    except sf.LibsndfileError as error:
        raise ValueError(f"failed to decode audio: {error}") from error
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    total_samples = len(data)
    chunks = []
    for end in range(
        chunk_size_samples,
        total_samples + chunk_size_samples,
        chunk_size_samples,
    ):
        end = min(end, total_samples)
        buffer = io.BytesIO()
        sf.write(buffer, data[:end], sample_rate, format="WAV")
        chunks.append(buffer.getvalue())
    return chunks


def normalize_whitespace(text: str) -> str:
    return _PUNCT_WS_RE.sub(r"\1", text)


_NO_SPACE_BEFORE = frozenset(".,!?;:%)]}，。！？；：、）】》」』")
_NO_SPACE_AFTER = frozenset("([{（【《「『")


def is_cjk_char(char: str) -> bool:
    """Whether a character belongs to a CJK context that takes no added space.

    This includes CJK punctuation, Japanese kana, ideographs, and fullwidth
    forms while excluding non-ASCII scripts that remain whitespace-delimited.
    """
    cp = ord(char)
    return (
        0x3000 <= cp <= 0x303F
        or 0x3040 <= cp <= 0x309F
        or 0x30A0 <= cp <= 0x30FF
        or 0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xFF00 <= cp <= 0xFFEF
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
    if is_cjk_char(prev[-1]) and is_cjk_char(cur[0]):
        return False
    return True


async def generate_asr_transcript(
    tokenizer_manager: TokenizerManager,
    adapter: TranscriptionAdapter,
    audio_data: Union[bytes, np.ndarray],
    sampling_params: Dict[str, Any],
    prompt: str,
    raw_request: Optional[Request] = None,
    routing_key: Optional[str] = None,
    audio_encoder_window_config: Optional["AudioEncoderWindowConfig"] = None,
    mm_hashes: Optional[List[str]] = None,
) -> Optional[GeneratedTranscript]:
    """Run one stateless backend request and return text with its stop reason."""
    chunk_request = GenerateReqInput(
        text=prompt,
        audio_data=audio_data,
        sampling_params=sampling_params,
        stream=False,
        modalities=["audio"],
        routing_key=routing_key,
        mm_hashes=mm_hashes,
    )

    try:
        ret = None
        async for ret in tokenizer_manager.generate_request(
            chunk_request,
            raw_request,
            audio_encoder_window_config=audio_encoder_window_config,
        ):
            break
    except asyncio.CancelledError:
        raise
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
    result = await generate_asr_transcript(
        tokenizer_manager=tokenizer_manager,
        adapter=adapter,
        audio_data=audio_data,
        sampling_params=sampling_params,
        prompt=adapter.prompt_template + state.get_prefix_text(),
        raw_request=raw_request,
        routing_key=routing_key,
        mm_hashes=[hash_audio_content(audio_data)],
    )
    if result is None:
        return ""
    text = result.text

    if is_last:
        state.full_transcript = text
        return state.finalize()
    return state.update(text)
