from typing import Optional

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.nemotron3_5_asr import Nemotron3_5AsrForRNNT
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio

AUTO_LANGUAGE = "auto"


def resolve_prompt_language(
    language: Optional[str], prompt_dictionary: dict[str, int]
) -> str:
    """Map an OpenAI-style language (ISO 639-1 code or locale) to a prompt key.

    Bare codes the checkpoint lacks (e.g. "th") fall back to their only
    locale ("th-TH"); None selects automatic language detection.
    """
    if language is None:
        return AUTO_LANGUAGE
    if language in prompt_dictionary:
        return language
    locales = sorted(k for k in prompt_dictionary if k.startswith(f"{language}-"))
    if locales:
        return locales[0]
    raise ValueError(
        f"Language {language!r} is not supported by this model. Supported: "
        f"{sorted(prompt_dictionary)}."
    )


class Nemotron3_5AsrProcessor(BaseMultimodalProcessor):
    models = [Nemotron3_5AsrForRNNT]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.blank_token_id = hf_config.blank_token_id

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        if not audio_data:
            return None
        if len(audio_data) != 1:
            raise ValueError(
                f"Nemotron 3.5 ASR expects exactly 1 audio input, got {len(audio_data)}"
            )

        sampling_params = request_obj.sampling_params or {}
        language = resolve_prompt_language(
            sampling_params.pop("language", None), self._processor.prompt_dictionary
        )
        audio = load_audio(audio_data[0])
        inputs = self._processor(audio, sampling_rate=16000, language=language)

        return MultimodalProcessorOutput(
            # The prompt is a single blank; RNN-T has no text prefix.
            input_ids=[self.blank_token_id],
            mm_items=[
                MultimodalDataItem(
                    feature=inputs["input_features"][0],
                    modality=Modality.AUDIO,
                    model_specific_data={
                        "prompt_ids": inputs["prompt_ids"],
                        "num_lookahead_tokens": inputs["num_lookahead_tokens"],
                    },
                )
            ],
        )
