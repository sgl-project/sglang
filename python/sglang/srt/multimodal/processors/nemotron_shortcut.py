# Copyright 2026 SGLang Team
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
"""Multimodal processor for Nemotron Shortcut: the checkpoint's processor
expands each image into rows of image tokens for the Pixtral tower."""

from typing import Any, List, Union

from sglang.srt.managers.req_types import MultimodalProcessorOutput
from sglang.srt.models.nemotron_shortcut import Nemotron4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor


class NemotronShortcutMMProcessor(PixtralProcessor):
    models = [Nemotron4ForConditionalGeneration]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # Pixtral's __init__ registers a <pad> special token; this tokenizer
        # needs no new tokens, so set up the base class directly.
        BaseMultimodalProcessor.__init__(
            self, hf_config, server_args, _processor, *args, **kwargs
        )
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.image_token = _processor.image_token
        self._merged_patch = _processor.patch_size * _processor.spatial_merge_size
        self._longest_edge = _processor.image_processor.size.longest_edge
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        mm_data = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
            return_text=True,
            # The image processor composites transparency onto white.
            discard_alpha_channel=False,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            mm_data, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.IM_TOKEN_ID,
        )

    def _get_image_nrows(self, images: List[Any]) -> List[int]:
        """Rows of image tokens per image: one contiguous run per row."""
        rows = []
        for image in images:
            # The checkpoint's image processor shrinks (never grows) the image
            # until its longest side fits, then pads each side to a whole patch.
            width, height = image.size
            ratio = max(height / self._longest_edge, width / self._longest_edge, 1.0)
            rows.append(-(-round(height / ratio) // self._merged_patch))
        return rows
