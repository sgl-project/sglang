from __future__ import annotations

import logging
import re
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)
from sglang.srt.multimodal.processors.whisper import ISO639_1_SUPPORTED_LANGS

logger = logging.getLogger(__name__)

# Regex that matches the forced decoder prefix output:
#   <|lang|><|transcribe|><|notimestamps|>  followed by any transcription text.
# Built once, reused for every auto-detect request.
_LANG_ALT = "|".join(re.escape(c) for c in ISO639_1_SUPPORTED_LANGS)
WHISPER_AUTODETECT_REGEX = (
    r"<\|(" + _LANG_ALT + r")\|>" + r"<\|transcribe\|>" + r"<\|notimestamps\|>" + r"[\s\S]*"
)

# Pattern to extract the language code from the fused output.
_LANG_PREFIX_RE = re.compile(r"^<\|([a-z]{2})\|>")


@register_transcription_adapter("Whisper")
class WhisperAdapter(TranscriptionAdapter):
    TIMESTAMP_BASE_TOKEN_ID = 50365  # <|0.00|>
    TIMESTAMP_BASE_OFFSET = 0.02  # each token step = 0.02 s

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        params: dict = {
            "temperature": request.temperature,
            "max_new_tokens": 448,  # Whisper default max tokens
            "language": request.language,
        }
        if request.timestamp_granularities:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params

    # -- language detection ------------------------------------------------

    @property
    def supports_language_detection(self) -> bool:
        return True

    def build_fused_autodetect_params(self, request: TranscriptionRequest) -> dict:
        """Build sampling params for a single fused detect+transcribe request.

        Uses SGLang's native structured generation (``regex``) to constrain
        the first 3 decode tokens to ``<|lang|><|transcribe|><|notimestamps|>``
        while allowing free transcription afterwards.  This runs language
        detection and transcription in a single encoder pass with no extra
        HTTP round-trip.
        """
        return {
            "temperature": request.temperature,
            # 3 forced prefix tokens + 448 transcription tokens
            "max_new_tokens": 451,
            "regex": WHISPER_AUTODETECT_REGEX,
            "skip_special_tokens": False,
            "_detect_language": True,
        }

    @staticmethod
    def parse_fused_output(text: str) -> tuple[str, str]:
        """Parse fused output ``<|en|><|transcribe|><|notimestamps|> Hello...``

        Returns (language_code, transcription_text).
        """
        m = _LANG_PREFIX_RE.match(text)
        lang = m.group(1) if m else "en"
        # Strip the 3-token forced prefix
        prefix_end = text.find("<|notimestamps|>")
        if prefix_end >= 0:
            transcription = text[prefix_end + len("<|notimestamps|>") :]
        else:
            transcription = text
        return lang, transcription.strip()

    # -- Standalone detection (for external callers) -------------------------

    def build_language_detection_params(self, tokenizer) -> dict:
        """Build sampling params for a language-detection-only request.

        Uses SGLang's native structured generation (``regex``) to constrain
        the single output token to a valid Whisper language token.
        Can be sent via the ``/generate`` endpoint independently of
        transcription.
        """
        lang_regex = r"<\|(" + _LANG_ALT + r")\|>"
        return {
            "max_new_tokens": 1,
            "temperature": 0,
            "regex": lang_regex,
            "skip_special_tokens": False,
            "_detect_language": True,
        }

    @staticmethod
    def parse_language_detection_output(
        output_ids: List[int], tokenizer
    ) -> Optional[str]:
        """Decode the predicted token and extract the ISO 639-1 language code."""
        if not output_ids:
            return None
        decoded = tokenizer.decode([output_ids[0]], skip_special_tokens=False)
        # Whisper language tokens have the form <|xx|>
        if decoded.startswith("<|") and decoded.endswith("|>"):
            lang_code = decoded[2:-2]
            if lang_code in ISO639_1_SUPPORTED_LANGS:
                return lang_code
        return None

    # -- end language detection --------------------------------------------

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        output_ids = ret.get("output_ids", [])
        parsed_text, segments = self._parse_segments(output_ids, tokenizer)
        return TranscriptionVerboseResponse(
            language=request.language or "en",
            duration=round(request.audio_duration_s, 2),
            text=parsed_text or text,
            segments=segments,
            usage=usage,
        )

    @staticmethod
    def _parse_segments(
        output_ids: List[int], tokenizer
    ) -> tuple[str, List[TranscriptionSegment]]:
        """Parse Whisper timestamp tokens from *output_ids* into segments.

        The decoder prompt ends with ``<|0.00|>``, so the first segment starts
        at t=0.  The model then outputs::

            text_tokens <|end_ts|> [<|start_ts|> text_tokens <|end_ts|> ...]

        Each timestamp token marks the end of the current segment; its value
        also becomes the start of the next segment.
        """
        eos_token_id = getattr(tokenizer, "eos_token_id", 50257)
        ts_base = WhisperAdapter.TIMESTAMP_BASE_TOKEN_ID
        ts_step = WhisperAdapter.TIMESTAMP_BASE_OFFSET

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        current_text_tokens: list[int] = []
        current_start = 0.0  # First segment starts at 0.0 (from prompt <|0.00|>)
        seg_id = 0

        for token_id in output_ids:
            if token_id >= ts_base:
                timestamp = (token_id - ts_base) * ts_step

                if current_text_tokens:
                    seg_text = tokenizer.decode(
                        current_text_tokens, skip_special_tokens=True
                    ).strip()
                    if seg_text:
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                start=round(current_start, 2),
                                end=round(timestamp, 2),
                                text=seg_text,
                            )
                        )
                        full_text_parts.append(seg_text)
                        seg_id += 1
                    current_text_tokens = []

                current_start = timestamp

            elif token_id == eos_token_id:
                continue
            else:
                current_text_tokens.append(token_id)

        if current_text_tokens:
            seg_text = tokenizer.decode(
                current_text_tokens, skip_special_tokens=True
            ).strip()
            if seg_text:
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        start=round(current_start, 2),
                        end=round(current_start, 2),
                        text=seg_text,
                    )
                )
                full_text_parts.append(seg_text)

        return " ".join(full_text_parts), segments
