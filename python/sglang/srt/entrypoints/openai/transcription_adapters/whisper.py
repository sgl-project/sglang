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
    r"<\|("
    + _LANG_ALT
    + r")\|>"
    + r"<\|transcribe\|>"
    + r"<\|notimestamps\|>"
    + r"[\s\S]*"
)

# Pattern to extract the language code from the fused output.
# Scoped to ISO639_1_SUPPORTED_LANGS so a bypassed/drifted FSM can't sneak
# through a random 2-letter code.
_LANG_PREFIX_RE = re.compile(r"^<\|(" + _LANG_ALT + r")\|>")
_FUSED_SENTINEL = "<|notimestamps|>"


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
            # Whisper's max_target_positions is 448 — 3 forced prefix tokens
            # + up to 445 transcription tokens stays under the position-embed cap.
            "max_new_tokens": 448,
            "regex": WHISPER_AUTODETECT_REGEX,
            "skip_special_tokens": False,
            "_detect_language": True,
        }

    @staticmethod
    def parse_fused_output(text: str) -> tuple[Optional[str], str]:
        """Parse fused output ``<|en|><|transcribe|><|notimestamps|> Hello...``

        Returns ``(language_code, transcription_text)`` on success, or
        ``(None, text)`` on parse failure. Callers must treat ``None`` as a
        parse error and not write it back to ``request.language``.

        Failure modes surface as ``None`` rather than a silent ``"en"``
        default so an FSM abort, truncation, or regex drift is visible in
        logs instead of being reported as a correct English detection.
        """
        m = _LANG_PREFIX_RE.match(text)
        if not m:
            logger.warning(
                "Whisper fused auto-detect parse failed: missing language prefix in %r",
                text[:64],
            )
            return None, text
        sentinel_idx = text.find(_FUSED_SENTINEL)
        if sentinel_idx < 0:
            logger.warning(
                "Whisper fused auto-detect parse failed: missing %s sentinel in %r",
                _FUSED_SENTINEL,
                text[:64],
            )
            return None, text
        transcription = text[sentinel_idx + len(_FUSED_SENTINEL) :]
        return m.group(1), transcription.strip()

    @staticmethod
    def fused_prefix_end(text: str) -> int:
        """Char offset in *text* of the first user-visible transcription char.

        Returns ``-1`` when the boundary isn't locatable yet — either the
        ``<|notimestamps|>`` sentinel hasn't arrived, or the sentinel is in
        but only trailing whitespace follows it. Deferring in the
        whitespace-only case matches ``parse_fused_output``'s ``.strip()``
        behavior and prevents the first streamed delta from leaking the
        leading space Whisper emits between the prefix and the first word.

        Used by the streaming handler to re-anchor ``stream_buffer`` so
        deltas sent to the client never include the forced special tokens.
        """
        idx = text.find(_FUSED_SENTINEL)
        if idx < 0:
            return -1
        end = idx + len(_FUSED_SENTINEL)
        while end < len(text) and text[end].isspace():
            end += 1
        if end >= len(text):
            # Sentinel seen but no non-whitespace transcription char yet —
            # defer so streaming doesn't emit a leading space.
            return -1
        return end

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
