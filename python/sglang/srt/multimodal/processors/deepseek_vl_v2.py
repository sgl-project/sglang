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
from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.deepseek_vl2 import DeepseekVL2ForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class DeepseekVL2ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekVL2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<image>"

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs
    ):
        base_output = self.load_mm_data(
            input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=self.IMAGE_TOKEN),
            max_req_input_len=max_req_input_len,
        )
        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            max_req_input_len=max_req_input_len,
            conversations=base_output.input_text,
        )
        images_seq_mask = res["images_seq_mask"]
        images_spatial_crop = res["images_spatial_crop"]
        batched_images_spatial_crop = []
        batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        items = []
        input_ids = res["input_ids"]
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids, mm_token_id=self._processor.image_token_id
        )
        item = MultimodalDataItem(
            pixel_values=res["images"],
            offsets=image_offsets,
            modality=Modality.IMAGE,
            image_emb_mask=images_seq_mask,
            image_spatial_crop=batched_images_spatial_crop,
        )
        items += [item]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self._processor.image_token_id,
        }
