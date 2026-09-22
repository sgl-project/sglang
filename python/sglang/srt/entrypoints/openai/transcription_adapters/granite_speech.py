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

# Per-second output-token budget for clips long enough to exceed the floor.
# Speech runs ~2-3 words/s and tokenizers emit >1 token/word, so 15 tok/s
# leaves generous headroom over the natural transcript length while still
# bounding runaway generation.
_MAX_NEW_TOKENS_PER_SECOND = 15
# Floor for the generation-length cap; sized so a ~30s clip always fits.
_DEFAULT_MAX_NEW_TOKENS = 448


@register_transcription_adapter("GraniteSpeech")
class GraniteSpeechAdapter(TranscriptionAdapter):
    """Transcription adapter for IBM Granite Speech.

    Granite Speech is a decoder-only speech LM: the audio encoder features are
    merged into the ``<|audio|>`` placeholder positions and the model free-form
    generates the transcript. It has no Whisper-style forced language/task
    prefix, so language auto-detection and fused prefix parsing are disabled and
    the default (no-op) prompt/response handling applies. Granite is
    English-only, so the verbose response reports ``"en"`` when the request
    omits a language.
    """

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        # ``/v1/audio/transcriptions`` has no request-side length field, so the
        # adapter is the only control. Short clips use the floor; longer clips
        # scale with duration so the transcript is not silently truncated.
        duration_s = request.audio_duration_s or 0.0
        return {
            "temperature": request.temperature,
            "max_new_tokens": max(
                _DEFAULT_MAX_NEW_TOKENS,
                int(duration_s * _MAX_NEW_TOKENS_PER_SECOND),
            ),
        }

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        # ``language`` left unset. It is multilingual and infers the language from the audio.
        return TranscriptionVerboseResponse(
            language=None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
