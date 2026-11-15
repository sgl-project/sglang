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
from sglang.srt.multimodal.processors.qwen3_asr import DEFAULT_ASR_PROMPT


@register_transcription_adapter("Qwen3ASR")
class Qwen3ASRAdapter(TranscriptionAdapter):
    ASR_TEXT_TAG = "<asr_text>"

    @property
    def supports_chunked_streaming(self) -> bool:
        return True

    @property
    def chunked_streaming_config(self) -> dict:
        # Qwen3-ASR paper (arXiv:2601.21337), Table 8 uses 4 unfixed chunks.
        # We use 2 here for lower latency; tune based on quality needs.
        # TODO: allow users to override these via API request parameters.
        return {
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
        }

    @property
    def prompt_template(self) -> str:
        return DEFAULT_ASR_PROMPT

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        temperature = request.temperature
        if temperature == 0.0:
            temperature = 0.01  # Qwen3-ASR recommended near-greedy temperature
        return {
            "temperature": temperature,
            "max_new_tokens": 256,  # Qwen3-ASR default
        }

    def postprocess_text(self, text: str) -> str:
        # Qwen3-ASR outputs "language <lang><asr_text>transcription" format;
        # strip the prefix to return clean transcription text.
        if self.ASR_TEXT_TAG in text:
            return text.split(self.ASR_TEXT_TAG, 1)[-1]
        return text

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        # TODO: Qwen3-ASR needs ForcedAligner to produce timestamps
        return TranscriptionVerboseResponse(
            language=request.language or "auto",
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
