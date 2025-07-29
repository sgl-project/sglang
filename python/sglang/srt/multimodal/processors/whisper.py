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

        processor_output = self.process_mm_data(
            input_text=input_text,
            audio=audios,
            return_attention_mask=True,
            pad_to_multiple_of=320,
            sampling_rate=16000,
        )
        input_ids = processor_output["labels"][0]
        input_ids_mask = input_ids != self._processor.tokenizer.pad_token_id
        output = {}
        output["data_hashes"] = [hash(audio_data) for audio_data in audio_data]
        output["input_ids"] = input_ids[input_ids_mask].tolist()
        output["mm_items"] = [
            MultimodalDataItem(
                feature=processor_output["input_features"][0],
                modality=Modality.AUDIO,
            )
        ]

        return output
