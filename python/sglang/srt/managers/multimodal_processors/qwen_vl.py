import asyncio
import math
import time
from typing import List, Union

import torch
from PIL import Image

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
    get_global_processor,
)
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    @staticmethod
    def _process_images_task(images, input_text, _hf_config):
        if isinstance(images, list) and len(images) == 0:
            images = None
        result = get_global_processor().__call__(
            text=[input_text], images=images, padding=True, return_tensors="pt"
        )

        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "image_grid_thw": getattr(result, "image_grid_thw", None),
            "second_per_grid_ts": getattr(result, "second_per_grid_ts", None),
            "video_grid_thws": getattr(result, "video_grid_thws", None),
        }

    async def _process_single_image(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Qwen2_5VLImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        start = time.time()
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_mm_data(
            input_ids=input_ids,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=image_token),
            max_req_input_len=max_req_input_len,
        )

        def smart_resize(
            height: int,
            width: int,
            factor: int = self.IMAGE_FACTOR,
            min_pixels: int = self.MIN_PIXELS,
            max_pixels: int = self.MAX_PIXELS,
        ) -> tuple[int, int]:
            """
            Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > self.MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
                )
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
            return h_bar, w_bar

        def resize_image(image, size_factor: int = self.IMAGE_FACTOR) -> Image.Image:
            width, height = image.size
            min_pixels = self.MIN_PIXELS
            max_pixels = self.MAX_PIXELS
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            return image

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        images = [resize_image(image) for image in base_output.images]

        ret = await self._process_single_image(
            images=images, input_text=base_output.input_text
        )

        image_grid_thws = torch.concat([ret["image_grid_thw"]])
        video_grid_thws = None
        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "data_hashes": base_output.mm_data_hashes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": image_grid_thws,
            "video_grid_thws": video_grid_thws,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "second_per_grid_ts": ret["second_per_grid_ts"],
        }
