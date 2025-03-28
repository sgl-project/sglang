import asyncio
from typing import List, Union

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    get_global_processor,
)
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.utils import load_image


class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_single_image_task(images, input_text):
        # input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'
        return get_global_processor()(images, input_text, return_tensors="pt")

    async def _process_single_image(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                MllamaImageProcessor._process_single_image_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(images, input_text, return_tensors="pt")

        return image_inputs

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

        image_inputs = await self._process_single_image(images, input_text)
        image_inputs["data_hashes"] = [hash(str(image_data))]
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]

        return image_inputs
