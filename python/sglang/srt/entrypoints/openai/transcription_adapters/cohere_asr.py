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

# The reference transcribe() caps generation at 256 tokens per clip, sized for
# its 35s chunk; keep that budget.
_MAX_NEW_TOKENS = 256

# Scripts the reference stitches without a separator (NO_SPACE_LANGS).
_NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})


@register_transcription_adapter("CohereAsr")
class CohereAsrAdapter(TranscriptionAdapter):
    """Transcription adapter for CohereLabs cohere-transcribe.

    The model performs no language detection -- the decoder prompt carries an
    explicit source/target language token -- so autodetect stays off and
    ``language`` is forwarded for the processor to build that prompt from.
    """

    @property
    def max_audio_clip_s(self) -> Optional[float]:
        # The Conformer was trained on clips up to 35s (config max_audio_clip_s);
        # the reference splits anything longer at low-energy points and stitches
        # the transcripts, which is what the server-side chunker does here.
        return 35.0

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        return {
            "temperature": request.temperature,
            "max_new_tokens": _MAX_NEW_TOKENS,
            "language": request.language,
        }

    def chunk_separator(self, request: TranscriptionRequest) -> str:
        # Transcripts come back with no leading space, so a seam would otherwise
        # fuse the words either side of it.
        language = (request.language or "").lower().strip()
        return "" if language in _NO_SPACE_LANGUAGES else " "

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
