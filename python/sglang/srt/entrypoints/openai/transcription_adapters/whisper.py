from __future__ import annotations

from typing import List

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


@register_transcription_adapter("Whisper")
class WhisperAdapter(TranscriptionAdapter):
    TIMESTAMP_BASE_TOKEN_ID = 50365  # <|0.00|>
    TIMESTAMP_BASE_OFFSET = 0.02  # each token step = 0.02 s

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        params: dict = {
            "temperature": request.temperature,
            "max_new_tokens": 448,  # Whisper default max tokens
            "language": request.language,
        }
        if request.timestamp_granularities:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        output_ids = ret.get("output_ids", [])
        parsed_text, segments = self._parse_segments(output_ids, tokenizer)
        return TranscriptionVerboseResponse(
            language=request.language or "en",
            duration=round(request.audio_duration_s, 2),
            text=parsed_text or text,
            segments=segments,
            usage=usage,
        )

    @staticmethod
    def _parse_segments(
        output_ids: List[int], tokenizer
    ) -> tuple[str, List[TranscriptionSegment]]:
        """Parse Whisper timestamp tokens from *output_ids* into segments.

        The decoder prompt ends with ``<|0.00|>``, so the first segment starts
        at t=0.  The model then outputs::

            text_tokens <|end_ts|> [<|start_ts|> text_tokens <|end_ts|> ...]

        Each timestamp token marks the end of the current segment; its value
        also becomes the start of the next segment.
        """
        eos_token_id = getattr(tokenizer, "eos_token_id", 50257)
        ts_base = WhisperAdapter.TIMESTAMP_BASE_TOKEN_ID
        ts_step = WhisperAdapter.TIMESTAMP_BASE_OFFSET

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        current_text_tokens: list[int] = []
        current_start = 0.0  # First segment starts at 0.0 (from prompt <|0.00|>)
        seg_id = 0

        for token_id in output_ids:
            if token_id >= ts_base:
                timestamp = (token_id - ts_base) * ts_step

                if current_text_tokens:
                    seg_text = tokenizer.decode(
                        current_text_tokens, skip_special_tokens=True
                    ).strip()
                    if seg_text:
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                start=round(current_start, 2),
                                end=round(timestamp, 2),
                                text=seg_text,
                            )
                        )
                        full_text_parts.append(seg_text)
                        seg_id += 1
                    current_text_tokens = []

                current_start = timestamp

            elif token_id == eos_token_id:
                continue
            else:
                current_text_tokens.append(token_id)

        if current_text_tokens:
            seg_text = tokenizer.decode(
                current_text_tokens, skip_special_tokens=True
            ).strip()
            if seg_text:
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        start=round(current_start, 2),
                        end=round(current_start, 2),
                        text=seg_text,
                    )
                )
                full_text_parts.append(seg_text)

        return " ".join(full_text_parts), segments
