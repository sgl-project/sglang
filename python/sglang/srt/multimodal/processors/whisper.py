from typing import Any, Dict, Optional

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        audios = [load_audio(audio) for audio in audio_data]
        assert len(audios) == 1

        if isinstance(input_text, list) and isinstance(input_text[0], int):
            input_ids = input_text
        else:
            input_ids = self._processor.tokenizer(input_text)["input_ids"]

        input_features = self._processor.feature_extractor(
            audios[0],
            pad_to_multiple_of=320,
            sampling_rate=16000,
            padding="longest",
            return_tensors="pt",
        )["input_features"][0]

        output = {}
        output["input_ids"] = input_ids
        output["mm_items"] = [
            MultimodalDataItem(
                feature=input_features,
                modality=Modality.AUDIO,
            )
        ]

        return output
