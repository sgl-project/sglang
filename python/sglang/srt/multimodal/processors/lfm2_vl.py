# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multimodal processor for LFM2-VL models with SigLip2 NaFlex support."""

from typing import List, Union

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.lfm2_vl import Lfm2VlForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)


class Lfm2VlImageProcessor(SGLangBaseProcessor):
    """Multimodal processor for LFM2-VL vision-language models.

    Uses the base class load_mm_data + process_and_combine_mm_data flow.
    The HF processor handles NaFlex variable-resolution tiling internally.
    """

    models = [Lfm2VlForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN_ID = hf_config.image_token_id
        self.IMAGE_TOKEN = "<image>"

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=hf_config.image_token_id,
        ).build(_processor)

        # Register NaFlex-specific HF processor outputs so
        # collect_mm_items_from_processor_output picks them up
        self.ATTR_NAME_TO_MODALITY["pixel_attention_mask"] = Modality.IMAGE
        self.ATTR_NAME_TO_MODALITY["spatial_shapes"] = Modality.IMAGE

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text: str,
        request_obj,
        **kwargs,
    ):
        if not image_data:
            input_ids = self._tokenizer(
                input_text, return_tensors="pt", add_special_tokens=False
            ).input_ids
            return {
                "input_ids": input_ids.squeeze(0).tolist(),
                "mm_items": [],
                "im_token_id": self.IMAGE_TOKEN_ID,
            }

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.IMAGE_TOKEN_ID,
        )
