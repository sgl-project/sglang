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
from sglang.srt.models.kimi_vl import KimiVLForConditionalGeneration


# Compatible with KimiVLForConditionalGeneration
class KimiVLImageProcessor(SGLangBaseProcessor):
    models = [KimiVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<|image_pad|>"
        self.im_token_id = 163605

        self.im_start = "<|media_start|>"
        self.im_start_id = 163602

        self.im_end = "<|media_end|>"
        self.im_end_id = 163604

        self.im_content = "<|media_content|>"
        self.im_content_id = 163603


    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        prompt,
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
        base_output = self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=image_token),
            max_req_input_len=max_req_input_len,
        )

        ret = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
        )

        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=ret["pixel_values"],
                    image_grid_thws=ret["image_grid_hws"],
                    modality=Modality.IMAGE,
                )
            ],
            "im_token_id": self.im_token_id,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "im_content_id": self.im_content_id,
        }
