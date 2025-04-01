from __future__ import annotations

import asyncio
import math
from typing import List, Union

import torch
from PIL import Image

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen2_5_omni import Qwen2_5OmniModel
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration

QWEN_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""
QWEN_AUDIO_SYSTEM_PROMPT = """You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."""


# Compatible with Qwen2VL, Qwen2_5VL and Qwen2_5_o
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5OmniModel,
    ]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        if self.arch == Qwen2_5OmniModel.__name__:
            self.image_token_id = hf_config.thinker_config.image_token_index
            self.image_start_id = hf_config.thinker_config.vision_start_token_id
            self.image_end_id = hf_config.thinker_config.vision_end_token_id

            self.audio_token_id = hf_config.thinker_config.audio_token_index
            self.audio_start_id = hf_config.thinker_config.audio_start_token_id
            self.audio_end_id = hf_config.thinker_config.audio_end_token_id

            self.video_token_id = hf_config.thinker_config.video_token_index
        else:
            self.image_token_id = hf_config.image_token_id
            self.image_start_id = hf_config.vision_start_token_id
            self.image_end_id = hf_config.vision_end_token_id
            self.video_token_id = hf_config.video_token_id
        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        prompt,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):

        print(f"{request_obj=}")

        if isinstance(image_data, str):
            image_data = [image_data]
        audio_data = request_obj.audio_data

        base_output = self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            audio_data=audio_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token_id,
                audio_token=getattr(self, "audio_token_id", None),
                video_token=getattr(self, "video_token_id", None),
            ),
            max_req_input_len=max_req_input_len,
        )
        print(f"11 {base_output=}")

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

        async def resize_image_async(image):
            return resize_image(image)

        resized_images = base_output.images
        if base_output.images:
            resize_tasks = [resize_image_async(image) for image in resized_images]
            resized_images = await asyncio.gather(*resize_tasks)
        print(f"{base_output=}")
        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=resized_images,
            audios=base_output.audios,
        )

        print(f"{res=}")

        items = []

        if "pixel_values" in res and res["pixel_values"] is not None:
            image_grid_thws = torch.concat([res["image_grid_thw"]])
            item = MultimodalDataItem(
                pixel_values=res["pixel_values"],
                image_grid_thws=image_grid_thws,
                modality=Modality.IMAGE,
            )
            items += [item]

        if "input_features" in res and res["input_features"] is not None:
            item = MultimodalDataItem(
                audio_feature=res["input_features"],
                feature_attention_mask=res["feature_attention_mask"],
                attention_mask=res["attention_mask"],
                modality=Modality.AUDIO,
            )
            items += [item]
        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "mm_items": items,
            "im_start_id": self.image_start_id,
            "im_end_id": self.image_end_id,
            "im_token_id": self.image_token_id,
            "audio_start_id": getattr(self, "audio_start_id", None),
            "audio_end_id": getattr(self, "audio_end_id", None),
            "video_token_id": self.video_token_id,
        }
