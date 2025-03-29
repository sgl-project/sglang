import asyncio
import math
from typing import List, Union

from PIL import Image

from sglang.srt.managers.image_processor import BaseImageProcessor
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration


# Compatible with Qwen2VL and Qwen2_5VL
class Qwen2_5VLImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        self.VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
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
    def _process_images_task(images, input_text, _hf_config, videos=None):
        if isinstance(images, list) and len(images) == 0:
            images = None
        if isinstance(videos, list) and len(videos) == 0:
            videos = None
        result = get_global_processor().__call__(
            text=[input_text],
            images=images,
            padding=True,
            return_tensors="pt",
            videos=videos,
        )

        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "pixel_values_videos": getattr(result, "pixel_values_videos", None),
            "image_grid_thw": getattr(result, "image_grid_thw", None),
            "second_per_grid_ts": getattr(result, "second_per_grid_ts", None),
            "video_grid_thw": getattr(result, "video_grid_thw", None),
        }

    async def _process_images(self, images, input_text, videos=None) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Qwen2_5VLImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
                videos,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        video_data: List[Union[str, bytes]],
        *args,
        **kwargs,
    ):
        if not image_data and not video_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]
        if isinstance(video_data, str):
            video_data = [video_data]

        image_token = self.IMAGE_TOKEN
        video_token = self.VIDEO_TOKEN
        base_output = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token=image_token,
            max_req_input_len=max_req_input_len,
            video_token=video_token,
            video_data=video_data,
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

        images = [resize_image(image) for image in base_output.all_frames]

        print(
            "videos to process_images",
            len(base_output.videos),
            type(base_output.videos),
        )
        ret = await self._process_images(
            images,
            base_output.input_text,
            videos=base_output.videos,
        )
        print('video grid thws', ret["video_grid_thw"])
        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "pixel_values_videos": ret["pixel_values_videos"],
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": ret["image_grid_thw"],
            "video_grid_thws": ret["video_grid_thw"],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "second_per_grid_ts": ret["second_per_grid_ts"],
            "media_order": base_output.media_order,
        }


ImageProcessorMapping = {
    Qwen2VLForConditionalGeneration: Qwen2_5VLImageProcessor,
    Qwen2_5_VLForConditionalGeneration: Qwen2_5VLImageProcessor,
}
