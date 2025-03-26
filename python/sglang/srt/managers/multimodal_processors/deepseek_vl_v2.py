# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import asyncio

import torch

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
    get_global_processor,
)
from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM


class DeepseekVL2ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekVL2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<image>"

    @staticmethod
    def _process_images_task(image, input_text, max_req_input_len):
        processor = get_global_processor()
        res = processor.__call__(
            conversations=input_text, images=image, max_req_input_len=max_req_input_len
        )

        image_token_id = processor.image_token_id

        res["im_token_id"] = image_token_id
        return res

    async def _process_images(self, image_data, input_text, max_req_input_len):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                DeepseekVL2ImageProcessor._process_images_task,
                image_data,
                input_text,
                max_req_input_len,
            )
        else:
            image_inputs = self._process_images_task(
                image_data, input_text, max_req_input_len
            )

        return image_inputs

    async def _process_images(self, image_data, input_text, max_req_input_len):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                DeepseekVL2ImageProcessor._process_images_task,
                image_data,
                input_text,
                max_req_input_len,
            )
        else:
            image_inputs = self._process_images_task(
                image_data, input_text, max_req_input_len
            )
        return image_inputs

    async def process_mm_data_async(
        self, image_data, input_ids, request_obj, max_req_input_len, *args, **kwargs
    ):
        if not image_data:
            return None

        if not isinstance(image_data, list):
            image_data = [image_data]

        images, image_sizes = [], []

        image_token = self.IMAGE_TOKEN
        base_output = self.load_mm_data(
            input_ids,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=image_token),
            max_req_input_len=max_req_input_len,
        )
        res = await self._process_images(
            base_output.images, base_output.input_text, max_req_input_len
        )
        images_seq_mask = res["images_seq_mask"]
        images_spatial_crop = res["images_spatial_crop"]
        batched_images_spatial_crop = []
        batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        return {
            "input_ids": res["input_ids"].tolist(),
            "pixel_values": res["images"],
            "im_token_id": res["im_token_id"],
            "data_hashes": base_output.mm_data_hashes,
            "image_sizes": image_sizes,
            "images_emb_mask": images_seq_mask,
            "image_spatial_crop": batched_images_spatial_crop,
            "modalities": request_obj.modalities or ["image"],
        }
