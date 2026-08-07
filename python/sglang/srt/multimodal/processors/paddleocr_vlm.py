# Adapted from:
# https://github.com/PaddlePaddle/PaddleX/blob/d28ed814fe769e312990b35f1a4657ed51f6226d/paddlex/inference/genai/models/paddleocr_vl_09b
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
from typing import List, Union

import torch
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.paddleocr_vl import PaddleOCRVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_image(image, min_pixels, max_pixels, factor) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image


async def resize_image_async(image, min_pixels, max_pixels, factor):
    return resize_image(image, min_pixels, max_pixels, factor)


class PaddleOCRVLImageProcessor(BaseMultimodalProcessor):
    models = [PaddleOCRVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        image_processor_config = _processor.image_processor
        self.MIN_PIXELS = image_processor_config.min_pixels
        self.MAX_PIXELS = image_processor_config.max_pixels
        self.IMAGE_FACTOR = (
            image_processor_config.patch_size * image_processor_config.merge_size
        )

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>",
            image_token_id=hf_config.image_token_id,
            video_token_id=hf_config.video_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [
                resize_image_async(
                    image, self.MIN_PIXELS, self.MAX_PIXELS, self.IMAGE_FACTOR
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.hf_config.model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.mm_tokens.image_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
