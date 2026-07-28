from __future__ import annotations

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("GraniteSpeech")
class GraniteSpeechAdapter(TranscriptionAdapter):
    """Transcription adapter for IBM Granite Speech.

    Granite Speech is a decoder-only speech LM: the audio encoder features are
    merged into the ``<|audio|>`` placeholder positions and the model free-form
    generates the transcript. It has no Whisper-style forced language/task
    prefix, so language auto-detection and fused prefix parsing are disabled and
    the default (no-op) prompt/response handling applies.
    """

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        return {
            "temperature": request.temperature,
            "max_new_tokens": 256,
        }

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        return TranscriptionVerboseResponse(
            language=request.language or "en",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
