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

# Registered under a key that can never match a real HF `architectures` entry
# (resolve_adapter() does substring matching) -- this adapter is only ever
# reached via _DEFAULT_ADAPTER_KEY, never via the substring scan itself.
_DEFAULT_ADAPTER_REGISTRY_KEY = "__generic_default__"


@register_transcription_adapter(_DEFAULT_ADAPTER_REGISTRY_KEY)
class GenericAdapter(TranscriptionAdapter):
    """Fallback adapter for ASR models with no dedicated adapter registered.

    Assumes the common decoder-only-audio-LM shape (the multimodal processor
    inserts an audio placeholder and the model free-form generates the
    transcript) rather than Whisper's forced language/task prefix.
    GraniteSpeechAdapter and Qwen2AudioAdapter already use this exact
    no-op-language pattern for their own decoder-only models; this is the same
    shape, used as the true default.
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
        # ``language`` left unset -- unknown for an arbitrary unmatched model.
        return TranscriptionVerboseResponse(
            language=None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
