from __future__ import annotations

import re

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*\[(?P<speaker>S\d{2,})\]"
    r"(?P<text>.*?)"
    r"\s*\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\s*\[S\d{2,}\]|$))",
    re.DOTALL,
)


@register_transcription_adapter("MossTranscribeDiarize")
class MossTranscribeDiarizeAdapter(TranscriptionAdapter):
    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        return {
            "temperature": request.temperature,
            "max_new_tokens": 2048,
        }

    def postprocess_text(self, text: str) -> str:
        return _SPECIAL_TOKEN_RE.sub("", text).strip()

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        return TranscriptionVerboseResponse(
            language=request.language or "auto",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=self._parse_segments(text),
            usage=usage,
        )

    @staticmethod
    def _parse_segments(text: str) -> list[TranscriptionSegment]:
        segments = []
        for segment_id, match in enumerate(_SEGMENT_RE.finditer(text)):
            speaker = match.group("speaker")
            body = match.group("text").strip()
            segment_text = f"[{speaker}]{body}" if body else f"[{speaker}]"
            segments.append(
                TranscriptionSegment(
                    id=segment_id,
                    start=round(float(match.group("start")), 2),
                    end=round(float(match.group("end")), 2),
                    text=segment_text,
                )
            )
        return segments
