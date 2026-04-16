# Copyright 2025 SGLang Team
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
# ==============================================================================

"""
Multimodal processor for lightonai/LightOnOCR-2-1B.

Key difference from Pixtral: LightOnOCR does NOT use image break/end tokens.
The parent PixtralProcessor inserts row-break and image-end tokens between
image patch rows. This processor removes them after the parent processing
to produce a single contiguous range of image tokens per image.
"""

from typing import List, Union

from sglang.srt.models.lightonocr import LightOnOCRForConditionalGeneration
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor


class LightOnOCRProcessor(PixtralProcessor):
    """Processor for LightOnOCR model."""

    models = [LightOnOCRForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # LightOnOCR uses image_token_id instead of image_token_index
        if not hasattr(hf_config, "image_token_index"):
            hf_config.image_token_index = getattr(hf_config, "image_token_id", 151655)

        # Propagate spatial_merge_size from root config to vision_config
        spatial_merge_size = getattr(hf_config, "spatial_merge_size", 2)
        if hasattr(hf_config, "vision_config"):
            vc = hf_config.vision_config
            if not hasattr(vc, "spatial_merge_size") or vc.spatial_merge_size is None:
                vc.spatial_merge_size = spatial_merge_size

        if hasattr(_processor, "patch_size"):
            _processor.spatial_merge_size = spatial_merge_size

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Identify break/end token IDs for removal
        self._break_token_ids = set()
        for attr in ("image_break_token_id", "image_break_id"):
            tid = getattr(_processor, attr, None)
            if tid is not None:
                self._break_token_ids.add(tid)
        for attr in ("image_end_token_id", "image_end_id"):
            tid = getattr(_processor, attr, None)
            if tid is not None:
                self._break_token_ids.add(tid)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        result = await super().process_mm_data_async(
            image_data=image_data,
            input_text=input_text,
            request_obj=request_obj,
            *args,
            **kwargs,
        )

        if not result or not self._break_token_ids:
            return result

        # Remove break/end tokens and fix multimodal item offsets
        input_ids = result.get("input_ids", [])
        mm_items = result.get("mm_items", [])

        new_input_ids = []
        old_to_new = {}
        for old_idx, token_id in enumerate(input_ids):
            if token_id not in self._break_token_ids:
                old_to_new[old_idx] = len(new_input_ids)
                new_input_ids.append(token_id)

        if len(new_input_ids) == len(input_ids):
            return result

        # Remap multimodal item offsets to account for removed tokens
        for mm_item in mm_items:
            if not mm_item.offsets:
                continue
            new_indices = sorted(
                old_to_new[idx]
                for start, end in mm_item.offsets
                for idx in range(start, end + 1)
                if idx in old_to_new
            )
            if new_indices:
                mm_item.offsets = [(new_indices[0], new_indices[-1])]

        result["input_ids"] = new_input_ids
        return result
