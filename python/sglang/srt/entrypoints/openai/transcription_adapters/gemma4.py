from __future__ import annotations

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

# Per-second output-token budget for clips long enough to exceed the floor.
# Speech runs ~2-3 words/s and tokenizers emit >1 token/word, so 15 tok/s
# leaves generous headroom over the natural transcript length while still
# bounding runaway generation.
_MAX_NEW_TOKENS_PER_SECOND = 15
# Floor for the generation-length cap; sized so a ~30s clip always fits.
_DEFAULT_MAX_NEW_TOKENS = 448

# ``<|channel>`` in the tokenizer shared by every Gemma 4 model.
_CHANNEL_OPEN_TOKEN_ID = 100


@register_transcription_adapter("Gemma4")
class Gemma4Adapter(TranscriptionAdapter):
    """Transcription adapter for the audio-capable Gemma 4 models (E2B, E4B, 12B).

    Gemma 4 is a decoder-only multimodal LM: ``Gemma4SGLangProcessor`` builds the
    model card's ASR chat prompt for text-less requests and the model free-form
    generates the transcript, so there is no Whisper-style language/task prefix.

    With thinking disabled the chat template already ends in an empty
    ``<|channel>thought\\n<channel|>`` block, yet under greedy decoding the model
    sometimes opens more of them, and on some clips repeats the empty block until
    ``max_new_tokens`` instead of transcribing (the HF reference does the same).
    A transcription never needs a thought channel, so ``<|channel>`` is banned.
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
            "logit_bias": {str(_CHANNEL_OPEN_TOKEN_ID): -100.0},
        }

    @property
    def max_audio_clip_s(self) -> Optional[float]:
        # Gemma 4 audio input supports at most 30 seconds (model card).
        return 30.0

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        # ``language`` left unset: the model infers the language from the audio.
        return TranscriptionVerboseResponse(
            language=None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )
