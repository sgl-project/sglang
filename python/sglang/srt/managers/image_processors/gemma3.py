import asyncio
from typing import List, Union

from transformers.utils import logging

from sglang.srt.managers.image_processor import (
    BaseImageProcessor as SGLangBaseImageProcessor,
)
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3_fast.py
# will be removed in the future
logger = logging.get_logger(__name__)


class Gemma3SGLangImageProcessor(SGLangBaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<start_of_image>"
        self.IM_START_TOKEN_ID = hf_config.boi_token_index
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index

    @staticmethod
    def _process_images_task(images, input_text, _hf_config):
        if isinstance(images, list) and len(images) == 0:
            images = None
        processor = get_global_processor()
        result = processor.__call__(
            text=[input_text],
            images=images,
            padding=True,
            return_tensors="pt",
            # if RGBA, this needs to be set
            # images_kwargs={
            #     "input_data_format": ChannelDimension.FIRST
            # }
        )

        pixel_values = getattr(result, "pixel_values", None)

        return {
            "input_ids": result.input_ids,
            "pixel_values": pixel_values,
        }

    async def _process_images(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Gemma3SGLangImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token=image_token,
            max_req_input_len=max_req_input_len,
            discard_alpha_channel=True,
        )

        ret = await self._process_images(
            input_text=base_output.input_text, images=base_output.all_frames
        )

        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "image_hashes": base_output.image_hashes,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }


ImageProcessorMapping = {
    Gemma3ForConditionalGeneration: Gemma3SGLangImageProcessor,
}
