from __future__ import annotations

import math
from typing import Optional

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)

# Arbitrary headroom over natural speech (~3-6 tokens/s on this tokenizer);
# the model ends the request itself once the audio is exhausted.
_MAX_NEW_TOKENS_PER_SECOND = 25
_MIN_MAX_NEW_TOKENS = 256


@register_transcription_adapter("Nemotron3_5Asr")
class Nemotron3_5AsrAdapter(TranscriptionAdapter):
    """Adapter for NVIDIA Nemotron 3.5 ASR (RNN-T).

    Decoding is greedy RNN-T inside the model, so temperature has no effect.
    ``language`` selects the model's language prompt; without it the model
    detects the language itself.
    """

    @property
    def max_audio_clip_s(self) -> Optional[float]:
        # The encoder attends over the whole clip at once, so its memory grows
        # quadratically with length; 30 s matches the Whisper adapter.
        return 30.0

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        duration_s = request.audio_duration_s or 0.0
        return {
            "max_new_tokens": max(
                _MIN_MAX_NEW_TOKENS,
                math.ceil(duration_s * _MAX_NEW_TOKENS_PER_SECOND),
            ),
            # Popped by the multimodal processor to pick the language prompt.
            "language": request.language,
        }

    def postprocess_text(self, text: str) -> str:
        # Long-audio chunks are joined with no separator, and decoding drops a
        # clip's leading word boundary; one leading space keeps words apart.
        text = text.strip()
        return f" {text}" if text else text

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        return TranscriptionVerboseResponse(
            language=request.language,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
