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


@register_transcription_adapter("GlmAsr")
class GlmAsrAdapter(TranscriptionAdapter):
    """Adapter for GLM-ASR.

    GLM-ASR is a decoder-only speech LM: the multimodal processor
    (``GlmAsrProcessor``) inserts the ``<|begin_of_audio|>...<|end_of_audio|>``
    placeholder and the model free-form generates the transcript. There is no
    Whisper-style forced language/task prefix, so language auto-detection is
    disabled and the default (no-op) prompt/response handling applies.
    """

    # Assistant framing GLM-ASR was trained to emit around the transcript.
    # Mirrors HF ``GlmAsrProcessor.decode(strip_prefix=True)`` so the raw
    # transcript is returned
    _ASSISTANT_PREFIXES = (
        "The spoken content of the audio is",
        "The transcription of the audio is",
        "The content of the input audio is",
    )

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

    def postprocess_text(self, text: str) -> str:
        # Strip the assistant prefix and surrounding quotes GLM-ASR wraps the
        # transcript in (mirrors HF's ``strip_prefix=True``).
        stripped = text.strip()
        for prefix in self._ASSISTANT_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break
        if stripped.endswith("."):
            stripped = stripped[:-1].strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0]
            in {
                "'",
                '"',
            }
        ):
            stripped = stripped[1:-1].strip()
        return stripped

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
