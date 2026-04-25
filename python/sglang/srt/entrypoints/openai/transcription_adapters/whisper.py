from __future__ import annotations

import logging
import re
from typing import List, Optional

from transformers.models.whisper.tokenization_whisper import LANGUAGES

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

# Sampling-params key the adapter plants and the multimodal processor pops
# to flip the decoder prompt from the explicit 4-token forced sequence to
# the bare ``<|startoftranscript|>`` (so the FSM regex drives token 1-3
# instead). Centralized so adapter / processor / warmup all reference the
# same string.
FUSED_AUTODETECT_FLAG = "_detect_language"

# The complete set of Whisper language tokens as they appear in the tokenizer
# vocab (<|xx|> / <|xxx|>). Sourced from the upstream ``LANGUAGES`` dict in
# ``transformers.models.whisper.tokenization_whisper`` so newly-added tokens
# (e.g. ``yue`` in Whisper v3) automatically propagate.
#
# Intentionally wider than ``processors.whisper.ISO639_1_SUPPORTED_LANGS``
# (the narrower input-validation set used by ``normalize_language_to_code``)
# — for the FSM regex we want every language the model was trained on so we
# don't silently force a wrong nearest-match code on audio in languages the
# model *can* detect but the input dict doesn't list (yue/Cantonese,
# jw/Javanese, haw/Hawaiian, ba/Bashkir, su/Sundanese, ...). Codes whose
# ``<|xxx|>`` token isn't in an older checkpoint's vocab are harmless —
# xgrammar simply leaves that regex branch with no admissible tokens.
WHISPER_LANG_TOKEN_CODES: frozenset[str] = frozenset(LANGUAGES.keys())

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

# Forced-prefix patterns, one per FSM variant. Each is anchored at start
# and rejects anything missing ``<|transcribe|>`` so a bypassed FSM or a
# mid-stream snapshot can't slip through as a valid detection. The two
# patterns differ in what the third forced token is decoded *as*:
#
# * ``_FUSED_PREFIX_RE_NOTS`` — non-timestamps variant. The third token
#   is ``<|notimestamps|>`` (id 50364), which detokenizes to its literal
#   string. Mid-stream snapshots stuck at ``<|en|><|transcribe|>`` (the
#   third token hasn't fired yet) correctly miss this regex, so the
#   streaming handler can detect FSM-abort and surface an error.
#
# * ``_FUSED_PREFIX_RE_TS`` — timestamps variant. The third token is
#   ``<|0.00|>`` (id 50365), which Whisper's tokenizer decodes to the
#   *empty string* even with ``skip_special_tokens=False`` (only
#   ``<|notimestamps|>`` survives detokenization; every ``<|X.XX|>``
#   maps to ``""``). So the regex must accept just
#   ``<|en|><|transcribe|>`` and rely on the FSM having already
#   constrained ``output_ids[2] == 50365``. ``_parse_segments`` reads
#   the timestamps from ``output_ids`` directly, so segment timing is
#   unaffected.
_FUSED_PREFIX_RE_NOTS = re.compile(
    r"^" + _LANG_PREFIX + r"<\|transcribe\|><\|notimestamps\|>"
)
_FUSED_PREFIX_RE_TS = re.compile(r"^" + _LANG_PREFIX + r"<\|transcribe\|>")

# Fixed Whisper control tokens (see transformers.models.whisper vocab).
# <|startoftranscript|> / <|startofprev|> / <|startoflm|> only appear at
# the decoder prompt and never in generated output, but they are cheap to
# include and harmless if they ever leak.
_WHISPER_CONTROL_TOKENS = frozenset(
    {
        "endoftext",
        "startoftranscript",
        "startofprev",
        "startoflm",
        "translate",
        "transcribe",
        "notimestamps",
        "nospeech",
    }
)

# Scrubs only actual Whisper special-token literals: language codes
# (WHISPER_LANG_TOKEN_CODES), control tokens (_WHISPER_CONTROL_TOKENS),
# and timestamp tokens (<|X.XX|>, where X.XX matches the
# ``{ts_base + k * 0.02}`` schema the model emits). A broad
# ``<\|[^|]+\|>`` would eat legitimate spoken content on audio that
# pronounces angle-bracket / pipe sequences (e.g. someone reading
# ``<|endoftext|>`` out loud). Used to scrub trailing ``<|endoftext|>``
# and embedded ``<|X.XX|>`` timestamp tokens from the user-visible text
# in fused-autodetect responses, where ``skip_special_tokens=False`` is
# needed to preserve the language prefix for parsing but would otherwise
# leak other special tokens downstream.
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:"
    + "|".join(sorted(WHISPER_LANG_TOKEN_CODES | _WHISPER_CONTROL_TOKENS))
    + r"|\d+\.\d{2}"
    + r")\|>"
)


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
        ts_variant = bool(request.timestamp_granularities)
        params: dict = {
            "temperature": request.temperature,
            # Fused auto-detect decoder prompt is just <|startoftranscript|>
            # (1 token, see processors/whisper.py). Whisper's
            # max_target_positions is 448, so max_new_tokens caps at 447:
            # 1 prompt + 3 forced prefix + up to 444 free transcription = 448.
            "max_new_tokens": 447,
            "regex": (
                WHISPER_AUTODETECT_TS_REGEX if ts_variant else WHISPER_AUTODETECT_REGEX
            ),
            "skip_special_tokens": False,
            # parse_fused_output matches a zero-space forced prefix
            # (``<|en|><|transcribe|><|notimestamps|>`` glued together).
            # Fast Whisper tokenizers decode adjacent added tokens with no
            # space, but slow ones insert a space between them. Force
            # spaces_between_special_tokens=False so the parse regex is
            # correct regardless of tokenizer variant.
            "spaces_between_special_tokens": False,
            FUSED_AUTODETECT_FLAG: True,
        }
        if ts_variant:
            params["timestamp_granularities"] = request.timestamp_granularities
        return params

    @staticmethod
    def parse_fused_output(
        text: str, *, ts_variant: bool = False
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse fused output into ``(language_code, user_visible_text)``.

        Matches the forced prefix the FSM emits. ``ts_variant`` selects
        which shape to expect — the caller knows from
        ``request.timestamp_granularities`` which regex was sent to the
        FSM and so which decoded shape to look for:

          * ``ts_variant=False`` — ``<|en|><|transcribe|><|notimestamps|> Hello...``
          * ``ts_variant=True``  — ``<|en|><|transcribe|> Hello...`` (``<|0.00|>``
            is in ``output_ids`` but Whisper detokenizes it to the empty string).

        Return cases:

        * ``(None, None)`` — the prefix isn't fully in yet (mid-stream
          snapshot before ``<|transcribe|>`` lands, or before
          ``<|notimestamps|>`` lands in the no-ts variant) or the prefix
          is malformed. Streaming callers keep buffering; non-streaming /
          end-of-stream callers treat this as a parse failure and fall
          back to a best-effort scrub of the raw text.
        * ``(lang, visible)`` — prefix fully parsed. ``visible`` is the
          transcription with the forced prefix removed, any embedded
          special tokens (``<|X.XX|>``, ``<|endoftext|>``) scrubbed, and
          surrounding whitespace trimmed. It grows monotonically across
          streaming chunks because Whisper's special tokens detokenize
          atomically, so callers can compute deltas against it directly.
        """
        pattern = _FUSED_PREFIX_RE_TS if ts_variant else _FUSED_PREFIX_RE_NOTS
        m = pattern.match(text)
        if not m:
            return None, None
        transcription = text[m.end() :]
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
