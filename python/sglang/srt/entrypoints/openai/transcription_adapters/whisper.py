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

logger = logging.getLogger(__name__)

# The complete set of Whisper language tokens as they appear in the tokenizer
# vocab (<|xx|> / <|xxx|>). Intentionally defined separately from
# ``processors.whisper.ISO639_1_SUPPORTED_LANGS`` (a narrower set used by the
# input-validation path ``normalize_language_to_code``) — for the FSM regex
# we want to cover every language the model was actually trained on so we
# don't silently force a wrong nearest-match code on audio in languages the
# model *can* detect but the input dict doesn't list (yue/Cantonese,
# jw/Javanese, haw/Hawaiian, ba/Bashkir, su/Sundanese, ...).
#
# Source: the ``LANGUAGES`` dict in
# ``transformers.models.whisper.tokenization_whisper``. Includes ``yue``
# (Cantonese), added in Whisper v3; harmless on older models where the
# ``<|yue|>`` token isn't in the vocab — xgrammar simply leaves that
# regex branch with no admissible tokens.
WHISPER_LANG_TOKEN_CODES: frozenset[str] = frozenset(
    {
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "yue",
        "zh",
    }
)

# Two forced-prefix variants, picked at request build time based on whether
# the client asked for timestamp_granularities:
#   * notimestamps variant: <|lang|><|transcribe|><|notimestamps|> text...
#     — drops segment/word timing, used when the client doesn't request it.
#   * timestamps variant:   <|lang|><|transcribe|><|0.00|> text <|X.XX|> ...
#     — <|0.00|> anchors the first segment at t=0, and the model naturally
#     emits further timestamp tokens between segments. _parse_segments
#     reconstructs segments from output_ids afterwards.
# sorted() gives a deterministic regex string so the warmup-compiled FSM is
# reused across server restarts.
_LANG_ALT = "|".join(re.escape(c) for c in sorted(WHISPER_LANG_TOKEN_CODES))
_LANG_PREFIX = r"<\|(" + _LANG_ALT + r")\|>"
WHISPER_AUTODETECT_REGEX = (
    _LANG_PREFIX + r"<\|transcribe\|>" + r"<\|notimestamps\|>" + r"[\s\S]*"
)
WHISPER_AUTODETECT_TS_REGEX = (
    _LANG_PREFIX + r"<\|transcribe\|>" + r"<\|0\.00\|>" + r"[\s\S]*"
)

# Pattern to extract the language code from the fused output. Scoped to
# WHISPER_LANG_TOKEN_CODES so a bypassed/drifted FSM can't sneak through a
# random 2-or-3-letter code that isn't actually a Whisper language token.
_LANG_PREFIX_RE = re.compile(r"^" + _LANG_PREFIX)

# Sentinel strings that mark the boundary between the forced prefix and the
# user-visible transcription. Ordered so we match the notimestamps variant
# first when both could theoretically appear (they never do in practice:
# the FSM emits exactly one of the two at position 2).
_FUSED_SENTINELS = ("<|notimestamps|>", "<|0.00|>")

# Matches any Whisper special token (``<|...|>``). Used to scrub trailing
# ``<|endoftext|>`` and embedded ``<|X.XX|>`` timestamp tokens from the
# user-visible text in fused-autodetect responses, where
# ``skip_special_tokens=False`` is needed to preserve the language prefix
# for parsing but would otherwise leak other special tokens downstream.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


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
        the first 3 decode tokens. Picks the regex variant based on whether
        the client requested ``timestamp_granularities``:

          * no timestamps:  ``<|lang|><|transcribe|><|notimestamps|>text``
          * with timestamps: ``<|lang|><|transcribe|><|0.00|>text<|X.XX|>...``
            — ``<|0.00|>`` anchors segment 0 at t=0; the model naturally
            emits further timestamp tokens between segments and
            ``_parse_segments`` reconstructs them from ``output_ids``.

        Either way, detection and transcription run in a single encoder
        pass with no extra HTTP round-trip.
        """
        use_ts = bool(request.timestamp_granularities)
        params: dict = {
            "temperature": request.temperature,
            # Fused auto-detect decoder prompt is just <|startoftranscript|>
            # (1 token, see processors/whisper.py). Whisper's
            # max_target_positions is 448, so max_new_tokens caps at 447:
            # 1 prompt + 3 forced prefix + up to 444 free transcription = 448.
            "max_new_tokens": 447,
            "regex": (
                WHISPER_AUTODETECT_TS_REGEX if use_ts else WHISPER_AUTODETECT_REGEX
            ),
            "skip_special_tokens": False,
            "_detect_language": True,
        }
        if use_ts:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params

    @staticmethod
    def _find_fused_sentinel(text: str) -> tuple[int, int]:
        """Locate the prefix sentinel and return ``(start_index, length)``.

        Tries each variant in ``_FUSED_SENTINELS``. The FSM emits exactly
        one of them at position 2, and the timestamps-variant ``<|0.00|>``
        only appears as the forced prefix (later segment-boundary timestamp
        tokens use non-zero values), so the first ``find`` hit is safe.
        Returns ``(-1, 0)`` when no sentinel is found.
        """
        for sentinel in _FUSED_SENTINELS:
            idx = text.find(sentinel)
            if idx >= 0:
                return idx, len(sentinel)
        return -1, 0

    @staticmethod
    def parse_fused_output(text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse fused output into ``(language_code, user_visible_text)``.

        Handles both prefix variants:
          * ``<|en|><|transcribe|><|notimestamps|> Hello...``
          * ``<|en|><|transcribe|><|0.00|> Hello<|5.00|>...``

        Return cases:

        * ``(None, None)`` — the forced prefix isn't locatable yet (language
          tag missing, or sentinel not in). Streaming callers should keep
          buffering; non-streaming / end-of-stream callers should treat
          this as a parse failure and fall back to a best-effort scrub of
          the raw text.
        * ``(lang, visible)`` — prefix fully parsed. ``visible`` is the
          transcription with the forced prefix removed, any embedded
          special tokens (``<|X.XX|>``, ``<|endoftext|>``) scrubbed, and
          surrounding whitespace trimmed. It grows monotonically across
          streaming chunks because Whisper's special tokens detokenize
          atomically, so callers can compute deltas against it directly.
        """
        m = _LANG_PREFIX_RE.match(text)
        if not m:
            return None, None
        sentinel_idx, sentinel_len = WhisperAdapter._find_fused_sentinel(text)
        if sentinel_idx < 0:
            return None, None
        transcription = text[sentinel_idx + sentinel_len :]
        # Scrub any remaining special tokens. skip_special_tokens=False is
        # set on fused requests so the language prefix survives for
        # parsing, but that also preserves trailing <|endoftext|> and, in
        # the timestamps variant, embedded <|X.XX|> segment tokens. Those
        # are unwanted in the user-visible text (verbose_json gets its
        # segments from _parse_segments over output_ids instead).
        transcription = _SPECIAL_TOKEN_RE.sub("", transcription)
        return m.group(1), transcription.strip()

    @staticmethod
    def strip_special_tokens(text: str) -> str:
        """Remove any ``<|...|>`` special-token strings from *text*.

        Used as the best-effort scrub on FSM abort / parse failure when
        the full ``parse_fused_output`` path can't locate the prefix.
        """
        return _SPECIAL_TOKEN_RE.sub("", text)

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
            # Pass None through when fused auto-detect failed to parse a
            # language — the client should see detection-failed, not a silent
            # English default. For explicit-language requests request.language
            # is already set by the caller.
            language=request.language,
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
