import asyncio
import re
from typing import Dict, List, Union

from PIL import Image

from sglang.srt.models.dots_ocr import DotsOCRForCausalLM
from sglang.srt.models.dots_vlm import DotsVLMForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.qwen_vl import resize_image_async


class DotsVLMImageProcessor(BaseMultimodalProcessor):
    models = [DotsVLMForCausalLM, DotsOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # The single, pre-expanded image token.
        self.IMAGE_TOKEN = "<|img|><|imgpad|><|endofimg|>"
        # The regex that matches expanded image tokens.
        self.IMAGE_TOKEN_REGEX = re.compile(r"<\|img\|>(?:<\|imgpad\|>)+<\|endofimg\|>")

        assert len(_processor.tokenizer.encode("<|img|>")) == 1
        self.im_start_id = _processor.tokenizer.encode("<|img|>")[0]
        self.im_end_id = _processor.tokenizer.encode("<|endofimg|>")[0]
        self.image_token_id = _processor.tokenizer.encode("<|imgpad|>")[0]
        self.IM_TOKEN_ID = self.image_token_id
        self.IM_START_ID = self.im_start_id
        self.IM_END_ID = self.im_end_id

        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        self.IMAGE_FACTOR = patch_size * merge_size
        self.MIN_PIXELS = _processor.image_processor.min_pixels
        self.MAX_PIXELS = _processor.image_processor.max_pixels
        self.MAX_RATIO = 200
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.image_token_id,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if isinstance(image_data, str):
            image_data = [image_data]

        if (
            isinstance(image_data, list)
            and image_data
            and isinstance(image_data[0], list)
        ):
            image_data = sum(image_data, [])

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [
                resize_image_async(
                    image,
                    min_pixels=self.MIN_PIXELS,
                    max_pixels=self.MAX_PIXELS,
                    size_factor=self.IMAGE_FACTOR,
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)
        combined_mm_item, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        if combined_mm_item is None:
            return None

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": combined_mm_item,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "im_token_id": self.image_token_id,
        }
