"""Segment-streaming ASR: fixed-duration, disjoint audio segments.

Chunked streaming (streaming_asr.py) re-transcribes a sliding window of
audio on every chunk, because bidirectional audio towers can't extend KV
over one growing clip. Segment streaming takes the other trade: audio is
cut into fixed-duration, non-overlapping segments and each segment is
transcribed independently; the per-segment texts are concatenated. Because
the segments are disjoint, each second of audio is encoded exactly once
(no re-encoded window), at the cost of accuracy at segment boundaries.

Mirrors vLLM's qwen3_asr_realtime, which yields each segment as a
standalone transcription prompt (bare template + that segment's audio).
An earlier design threaded the segments through one growing engine
streaming session; that re-fed each turn's transcript back as context,
and the model re-emitted the previous transcript instead of transcribing
the new segment, so segments 2..N all repeated segment 1. Independent
per-segment requests avoid that (and share no meaningful KV across
disjoint audio anyway).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sglang.srt.entrypoints.openai.streaming_asr import normalize_whitespace
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class SegmentStreamingASR:
    """One utterance transcribed as a sequence of independent segments.

    Call ``transcribe_segment(wav)`` once per fixed-duration segment; each
    call is a standalone transcription request that returns that segment's
    text and appends it to ``transcript``. ``close()`` is a no-op kept for
    API symmetry with the streaming lifecycle.
    """

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        sampling_params: Dict[str, Any],
        routing_key: Optional[str] = None,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.sampling_params = sampling_params
        self.routing_key = routing_key
        self.transcript = ""

    async def transcribe_segment(self, wav_data: bytes) -> str:
        """Transcribe one audio segment as a standalone request and return
        its text. The prompt is the bare per-turn template plus this
        segment's audio only — no prior segment's audio or transcript is
        carried, so the model transcribes exactly this segment."""
        req = GenerateReqInput(
            text=self.adapter.prompt_template,
            audio_data=wav_data,
            sampling_params=self.sampling_params,
            stream=False,
            modalities=["audio"],
        )
        if self.routing_key is not None:
            req.routing_key = self.routing_key

        ret = None
        async for ret in self.tokenizer_manager.generate_request(req):
            break
        if ret is None:
            logger.warning("[segment_asr] empty response")
            return ""

        text = normalize_whitespace(self.adapter.postprocess_text(ret.get("text", "")))
        if text:
            self.transcript = (
                f"{self.transcript} {text}".strip() if self.transcript else text
            )
        return text

    async def close(self) -> None:
        """No session to release: each segment is an independent request."""
        return
