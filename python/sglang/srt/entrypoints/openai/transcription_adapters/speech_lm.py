"""Data-driven transcription adapters for decoder-only speech LMs.

A speech LM transcribes when its chat turn carries the audio and an ASR
instruction. The server renders that turn from ``build_prompt()`` with the
model's own chat template and sends it as the request text, so the multimodal
processor sees an ordinary audio prompt, the same as on ``/v1/chat/completions``.
Supporting a model therefore needs no processor code, only data: the
instruction, where the audio goes, the generation budget and any output
cleanup. Unmatched architectures get ``SpeechLMAdapter`` itself.
"""

from __future__ import annotations

from typing import Mapping, Optional

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    _DEFAULT_ADAPTER_KEY,
    TranscriptionAdapter,
    TranscriptionPrompt,
    register_transcription_adapter,
)


def _language_code(language: Optional[str]) -> Optional[str]:
    """ISO 639-1 code for a language code or English name; ``None`` if unknown."""
    # Lazy import: the Whisper processor imports this package.
    from sglang.srt.multimodal.processors.whisper import normalize_language_to_code

    try:
        return normalize_language_to_code(language)
    except ValueError:
        return None


def _language_name(code: Optional[str]) -> str:
    from sglang.srt.multimodal.processors.whisper import ISO639_1_SUPPORTED_LANGS

    return ISO639_1_SUPPORTED_LANGS.get(code, "") if code else ""


@register_transcription_adapter(_DEFAULT_ADAPTER_KEY)
class SpeechLMAdapter(TranscriptionAdapter):
    """Transcription for a decoder-only speech LM, configured by class attributes.

    ``instruction`` may use ``{language_code}`` / ``{language_name}``; it is
    formatted only when the request names a language this server knows.
    ``instruction_by_language`` overrides it per ISO 639-1 code, and its ``None``
    entry is used when the request has no (or an unknown) language.
    """

    instruction: str = (
        "Transcribe the speech into written text. Output only the exact "
        "transcription, with no preamble, commentary, quotation marks, or "
        "explanation."
    )
    instruction_by_language: Mapping[Optional[str], str] = {}
    audio_first: bool = True
    system_prompt: Optional[str] = None
    content_format: str = "text"

    # Speech runs ~2-3 words/s and tokenizers emit >1 token/word, so 15 tok/s
    # leaves headroom over the transcript while still bounding runaway
    # generation; the floor fits a ~30s clip.
    max_new_tokens_floor: int = 448
    max_new_tokens_per_second: int = 15
    extra_sampling_params: Mapping = {}
    clip_limit_s: Optional[float] = None
    # Echo the request language in verbose_json. Only for models whose prompt
    # carries it; others neither honor nor detect it.
    reports_language: bool = False

    def build_prompt(self, request: TranscriptionRequest) -> TranscriptionPrompt:
        code = _language_code(request.language)
        name = _language_name(code)
        if code in self.instruction_by_language:
            instruction = self.instruction_by_language[code]
        else:
            instruction = self.instruction
            # A code with no English name (e.g. "yue") can't fill
            # {language_name}; treat it like an unknown language.
            if code is None or ("{language_name}" in instruction and not name):
                code = None
                instruction = self.instruction_by_language.get(None, instruction)
        if code is not None:
            instruction = instruction.format(language_code=code, language_name=name)
        return TranscriptionPrompt(
            instruction=instruction,
            audio_first=self.audio_first,
            system=self.system_prompt,
            content_format=self.content_format,
        )

    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        # ``/v1/audio/transcriptions`` has no request-side length field, so the
        # adapter is the only control. Short clips use the floor; longer clips
        # scale with duration so the transcript is not silently truncated.
        duration_s = request.audio_duration_s or 0.0
        return {
            "temperature": request.temperature,
            "max_new_tokens": max(
                self.max_new_tokens_floor,
                int(duration_s * self.max_new_tokens_per_second),
            ),
            **self.extra_sampling_params,
        }

    @property
    def max_audio_clip_s(self) -> Optional[float]:
        return self.clip_limit_s

    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        return TranscriptionVerboseResponse(
            language=request.language if self.reports_language else None,
            duration=round(request.audio_duration_s, 2),
            text=text,
            segments=[],
            usage=usage,
        )


@register_transcription_adapter("Qwen2Audio")
class Qwen2AudioAdapter(SpeechLMAdapter):
    # The strict wording keeps Qwen2-Audio from prefixing the transcript with
    # "The content of this audio is:".
    instruction = (
        "Transcribe the audio. Output only the exact transcription, "
        "with no preamble, prefix, commentary, or quotation marks."
    )
    # Its chat template emits the audio span for a content part with an
    # ``audio`` key.
    content_format = "parts"


@register_transcription_adapter("GlmAsr")
class GlmAsrAdapter(SpeechLMAdapter):
    instruction = "Transcribe the audio."
    content_format = "parts"

    # Assistant framing GLM-ASR was trained to emit around the transcript.
    # Mirrors HF ``GlmAsrProcessor.decode(strip_prefix=True)``.
    _ASSISTANT_PREFIXES = (
        "The spoken content of the audio is",
        "The transcription of the audio is",
        "The content of the input audio is",
    )

    def postprocess_text(self, text: str) -> str:
        stripped = text.strip()
        for prefix in self._ASSISTANT_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :].strip()
                break
        if stripped.endswith("."):
            stripped = stripped[:-1].strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
            stripped = stripped[1:-1].strip()
        return stripped


@register_transcription_adapter("GraniteSpeech")
class GraniteSpeechAdapter(SpeechLMAdapter):
    # Served through the Transformers backend; the default instruction and
    # string turn are what it was validated with. The strict wording matters:
    # "can you transcribe..." makes it wrap the transcript in prose.
    pass


@register_transcription_adapter("Gemma4")
class Gemma4Adapter(SpeechLMAdapter):
    # Google's ASR instruction (Gemma 4 model card, "Audio"), placed before the
    # audio as the card does; audio first raised LibriSpeech test-other WER from
    # ~14 to 23-37.
    instruction = (
        "Transcribe the following speech segment in its original language. "
        "Follow these specific instructions for formatting the answer:\n"
        "* Only output the transcription, with no newlines.\n"
        "* When transcribing numbers, write the digits, i.e. write 1.7 and not "
        "one point seven, and write 3 instead of three."
    )
    audio_first = False
    content_format = "parts"
    # With thinking off the template already ends in an empty thought block, yet
    # under greedy decoding the model reopens one on 2-5% of clips and repeats it
    # until max_new_tokens (the HF reference does too). Token 100 is
    # ``<|channel>`` in every Gemma 4 tokenizer.
    extra_sampling_params = {"logit_bias": {"100": -100.0}}
    # Gemma 4 takes at most 30 s of audio; longer clips are split.
    clip_limit_s = 30.0


@register_transcription_adapter("Qwen3OmniMoe")
class Qwen3OmniAdapter(SpeechLMAdapter):
    # Qwen3-Omni model card, "Setting for Evaluation": audio first, then this
    # instruction, no system prompt.
    instruction = "Transcribe the {language_name} audio into text."
    instruction_by_language = {
        "zh": "请将这段中文语音转换为纯文本。",
        None: "Transcribe the audio into text.",
    }
    content_format = "parts"
    reports_language = True
