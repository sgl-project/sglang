from typing import Any, Dict, Optional

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        audios = [load_audio(audio)[0] for audio in audio_data]

        processor_output = self.process_mm_data(input_text=input_text, audio=audios)
        output = {}
        output["data_hashes"] = [hash(str(audio_data)) for audio_data in audio_data]
        output["input_ids"] = processor_output["labels"].tolist()
        output["mm_items"] = [
            MultimodalDataItem(
                audio_features=processor_output["input_features"],
                modality=Modality.AUDIO,
            )
        ]

        return output
