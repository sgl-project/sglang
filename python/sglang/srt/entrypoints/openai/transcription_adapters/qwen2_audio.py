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


@register_transcription_adapter("Qwen2Audio")
class Qwen2AudioAdapter(TranscriptionAdapter):
    """Adapter for Qwen2-Audio.

    Qwen2-Audio is a decoder-only audio LM: the multimodal processor
    (``Qwen2AudioMultimodalProcessor``) inserts the audio placeholder and the
    model free-form generates the transcript. There is no Whisper-style forced
    language/task prefix, so language auto-detection is disabled and the
    default (no-op) prompt/response handling applies.
    """

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        return {
            "temperature": request.temperature,
            "max_new_tokens": 448,
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
            language=request.language or "auto",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
