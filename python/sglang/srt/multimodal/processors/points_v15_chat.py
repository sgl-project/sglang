# Copy from qwen_vl.py, adapted for points-v15-chat

import asyncio
from typing import List, Union

from PIL import Image

from sglang.srt.models.points_v15_chat import POINTSV15ChatModel
from sglang.srt.multimodal.processors.qwen_vl import (
    Qwen2_5VLImageProcessor,
    resize_image_async,
)


class POINTSV15ChatProcessor(Qwen2_5VLImageProcessor):
    models = [POINTSV15ChatModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # Compatible with POINTSV15Chat
        hf_config.vision_start_token_id = None
        hf_config.vision_end_token_id = None
        hf_config.video_token_id = None

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

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
            resize_tasks = [resize_image_async(image) for image in base_output.images]
            base_output.images = await asyncio.gather(*resize_tasks)

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
        }
