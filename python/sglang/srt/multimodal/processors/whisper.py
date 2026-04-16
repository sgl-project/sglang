import logging
from typing import Any, Dict, Optional

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio

logger = logging.getLogger(__name__)

# ISO 639-1 supported languages for Whisper
# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages
# Maps ISO 639-1 code -> Full language name
ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}

# Reverse mapping: Full language name (lowercase) -> ISO 639-1 code
LANG_NAME_TO_CODE = {
    name.lower(): code for code, name in ISO639_1_SUPPORTED_LANGS.items()
}


def normalize_language_to_code(language: Optional[str]) -> Optional[str]:
    """Convert a language input (full name or code) to ISO 639-1 code.

    Args:
        language: Language as full name (e.g., 'English', 'Spanish') or
                  ISO 639-1 code (e.g., 'en', 'es')

    Returns:
        ISO 639-1 code or None if input is None
    """
    if language is None:
        return None

    language_lower = language.lower().strip()

    # Check if it's already a valid ISO code
    if language_lower in ISO639_1_SUPPORTED_LANGS:
        return language_lower

    # Check if it's a full language name
    if language_lower in LANG_NAME_TO_CODE:
        return LANG_NAME_TO_CODE[language_lower]

    # Not recognized
    raise ValueError(
        f"Language '{language}' not recognized. "
        f"Use full name (e.g., 'English') or ISO 639-1 code (e.g., 'en')."
    )


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # Cache tokenizer for language token lookup
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _pop_sampling_param(self, request_obj, key: str):
        sampling_params = getattr(request_obj, "sampling_params", None) or {}
        return sampling_params.pop(key, None)

    def _get_language_token_id(self, language: Optional[str]) -> int:
        # Default to English if not specified
        if language is None:
            language = "en"  # Default to English
        language_token = f"<|{language}|>"
        return self._tokenizer.convert_tokens_to_ids(language_token)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None

        if len(audio_data) != 1:
            raise ValueError(
                f"Whisper expects exactly 1 audio input, got {len(audio_data)}"
            )

        # Check if this is a fused auto-detect request (decoder prompt = [SOT] only,
        # structured generation handles the rest via regex constraint).
        detect_language = self._pop_sampling_param(request_obj, "_detect_language")

        audios = [load_audio(audio) for audio in audio_data]

        # Whisper expects input features padded to max_length (3000 frames = 30 seconds)
        # This is the standard context length for Whisper
        fe_kwargs: dict[str, Any] = dict(
            sampling_rate=16000,
            padding="max_length",  # Pad to 3000 frames
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            fe_kwargs["device"] = "cuda"
        input_features = self._processor.feature_extractor(
            audios[0],
            **fe_kwargs,
        )["input_features"][0]

        # Whisper is a pure speech-to-text model; text prompts are ignored.
        # The full decoder sequence is:
        #   <|startoftranscript|> <|lang|> <|transcribe|> [<|notimestamps|> | <|0.00|>]
        #
        # When language is known, we build this prefix explicitly below.
        # When auto-detecting (_detect_language=True), we feed only <|startoftranscript|>
        # and let SGLang's structured generation (regex) constrain the model to produce
        # <|lang|><|transcribe|><|notimestamps|> as the first 3 decode tokens — this is
        # equivalent to HuggingFace's forced_decoder_ids but uses SGLang's native API.

        decoder_start_token_id = getattr(
            self.hf_config, "decoder_start_token_id", 50258
        )

        if detect_language:
            input_ids = [decoder_start_token_id]
        else:
            language = normalize_language_to_code(
                self._pop_sampling_param(request_obj, "language")
            )
            language_token_id = self._get_language_token_id(language)
            timestamp_granularities = self._pop_sampling_param(
                request_obj, "timestamp_granularities"
            )

            transcribe_token_id = self._tokenizer.convert_tokens_to_ids(
                "<|transcribe|>"
            )

            # Use <|0.00|> to enable timestamp generation, or <|notimestamps|> to disable
            if timestamp_granularities:
                timestamp_token_id = self._tokenizer.convert_tokens_to_ids("<|0.00|>")
            else:
                timestamp_token_id = self._tokenizer.convert_tokens_to_ids(
                    "<|notimestamps|>"
                )

            input_ids = [
                decoder_start_token_id,
                language_token_id,
                transcribe_token_id,
                timestamp_token_id,
            ]

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=[
                MultimodalDataItem(
                    feature=input_features,
                    modality=Modality.AUDIO,
                )
            ],
        )
