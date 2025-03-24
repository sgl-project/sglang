from typing import List, Union

from transformers import BaseImageProcessorFast

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.utils import load_image


class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    def _process_single_image_task(self, images, input_text):
        args = {}
        processor = self._processor
        if isinstance(processor, BaseImageProcessorFast):
            args["device"] = "cuda"
        # input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'
        return processor(images, input_text, return_tensors="pt", **args)

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

        image_inputs = self._process_single_image_task(images, input_text)
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]
        image_inputs["items"] = [
            MultimodalDataItem(
                pixel_values=image_inputs["pixel_values"],
                aspect_ratio_id=image_inputs["aspect_ratio_ids"],
                aspect_ratio_mask=image_inputs["aspect_ratio_mask"],
                modality="image",
            )
        ]

        return image_inputs
