from typing import Any, Dict, Optional

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.cohere_asr import CohereAsrForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio

_DEFAULT_LANGUAGE = "en"


def build_cohere_asr_prompt(language: str, punctuation: bool = True) -> str:
    """Decoder prompt prefix, matching the reference ``build_prompt``.

    Slot order is fixed by the checkpoint's ``prompt_format``: emotion, source
    language, target language, punctuation, inverse text normalization,
    timestamps, diarization.
    """
    pnc_token = "<|pnc|>" if punctuation else "<|nopnc|>"
    return (
        "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
        f"<|{language}|><|{language}|>{pnc_token}<|noitn|>"
        "<|notimestamp|><|nodiarize|>"
    )


class CohereAsrMultimodalProcessor(BaseMultimodalProcessor):
    models = [CohereAsrForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._tokenizer = self._processor.tokenizer
        self._supported_languages = set(hf_config.supported_languages)

    def _resolve_language(self, request_obj) -> str:
        # The adapter routes `language` through sampling_params; pop it so it
        # does not leak into SamplingParams(**kwargs) downstream.
        sampling_params = request_obj.sampling_params or {}
        language = sampling_params.pop("language", None) or _DEFAULT_LANGUAGE
        language = language.lower().strip()
        if language not in self._supported_languages:
            raise ValueError(
                f"Language '{language}' is not supported by this model. "
                f"Supported: {', '.join(sorted(self._supported_languages))}."
            )
        return language

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
                f"cohere-transcribe expects exactly 1 audio input, got {len(audio_data)}"
            )

        language = self._resolve_language(request_obj)
        audio = load_audio(audio_data[0])

        features = self._processor.feature_extractor(
            audio, sampling_rate=self.hf_config.sample_rate, return_tensors="pt"
        )
        # The mel frontend emits one stft frame more than the valid frame count
        # and zero-fills the tail. Trimming to `length` is what lets the encoder
        # run unmasked: convolution zero-padding reproduces that dropped frame
        # exactly, so the kept frames are bit-identical to the masked reference.
        valid_frames = int(features["length"][0])
        input_features = features["input_features"][0][:, :valid_frames]

        # Pure speech-to-text: any request text is ignored, the decoder is
        # seeded with the task prompt instead.
        input_ids = self._tokenizer(
            build_cohere_asr_prompt(language), add_special_tokens=False
        )["input_ids"]

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=[
                MultimodalDataItem(
                    feature=input_features,
                    modality=Modality.AUDIO,
                )
            ],
        )
