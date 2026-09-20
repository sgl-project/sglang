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


@register_transcription_adapter("MiMoV2ASR")
class MiMoV2ASRAdapter(TranscriptionAdapter):
    """Adapter for MiMo-V2-ASR.

    The multimodal processor (``MiMoV2ASRProcessor``) prepends the audio
    placeholder ``<|sosp|><|empty|>...<|eosp|>`` when ``input_text`` lacks
    one, so the request text can stay empty and the adapter only has to
    supply sampling params and the verbose-response shape.
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
        # MiMo-V2-ASR does not emit timestamp tokens; segments stay empty
        # until a forced-aligner path is added.
        return TranscriptionVerboseResponse(
            language=request.language or "auto",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
