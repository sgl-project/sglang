from typing import List, Union

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.clip import CLIPModel
from sglang.srt.utils import load_image


class ClipImageProcessor(BaseMultimodalProcessor):
    models = [CLIPModel]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        if not image_data:
            return None

        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        if not isinstance(image_data, list):
            image_data = [image_data]

        if len(image_data) > 0:
            images = [load_image(image)[0] for image in image_data]
        else:
            images = load_image(image_data[0])[0]

        image_inputs = self.process_mm_data(input_text=input_text, images=images)
        image_inputs["data_hashes"] = [hash(str(image_data))]
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]
        image_inputs["mm_items"] = [
            MultimodalDataItem(
                pixel_values=image_inputs["pixel_values"], modality=Modality.IMAGE
            )
        ]

        return image_inputs
