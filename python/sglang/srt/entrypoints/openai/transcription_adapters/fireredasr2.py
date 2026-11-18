from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("FireRedASR2")
class FireRedASR2Adapter(TranscriptionAdapter):
    @property
    def max_audio_clip_s(self) -> float:
        return 30.0

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        return {
            "temperature": request.temperature,
            "max_new_tokens": 512,
            "repetition_penalty": 1.05,
            "stop_token_ids": [151645, 151643],
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
            language=None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
