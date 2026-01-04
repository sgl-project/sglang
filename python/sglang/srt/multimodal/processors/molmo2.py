# Copyright 2023-2024 SGLang Team
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
"""Multimodal processor for Molmo2."""

from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.molmo2 import Molmo2ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

# Special tokens for Molmo2
IMAGE_PROMPT = "<|image|>"
VIDEO_PROMPT = "<|video|>"


class Molmo2MultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for Molmo2 models."""

    models = [Molmo2ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.hf_config = hf_config
        self.image_patch_id = getattr(hf_config, "image_patch_id", None)
        self.image_start_token_id = getattr(hf_config, "image_start_token_id", None)
        self.image_end_token_id = getattr(hf_config, "image_end_token_id", None)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PROMPT,
            video_token=VIDEO_PROMPT,
            image_token_id=self.image_patch_id,
            video_token_id=self.image_patch_id,  # Videos also use image_patch tokens
        ).build(_processor)

    def _split_offsets_by_item(
        self,
        all_offsets: List[Tuple[int, int]],
        expected_counts: List[int],
    ) -> List[List[Tuple[int, int]]]:
        """Split offsets into groups based on expected token counts per item."""
        result = []
        offset_idx = 0

        for count in expected_counts:
            item_offsets = []
            tokens_collected = 0

            while tokens_collected < count and offset_idx < len(all_offsets):
                start, end = all_offsets[offset_idx]
                tokens_collected += end - start + 1
                item_offsets.append((start, end))
                offset_idx += 1

            result.append(item_offsets)

        return result

    def _process_media_items(
        self,
        pixel_values: torch.Tensor,
        token_pooling: Optional[torch.Tensor],
        grids: Optional[torch.Tensor],
        num_items_info: Optional[torch.Tensor],
        all_offsets: List[Tuple[int, int]],
        modality: Modality,
        is_video: bool = False,
    ) -> List[MultimodalDataItem]:
        """Process image or video items into MultimodalDataItem list."""
        if grids is not None:
            expected_counts = []
            for grid in grids:
                if is_video:
                    num_frames, h, w = grid.tolist()
                    expected_counts.append(num_frames * h * w)
                else:
                    resized_h, resized_w, height, width = grid.tolist()
                    expected_counts.append(resized_h * resized_w + height * width)
            offsets_per_item = self._split_offsets_by_item(all_offsets, expected_counts)
        else:
            offsets_per_item = [all_offsets]

        num_items = len(grids) if grids is not None else 1
        items = []
        pooling_offset = 0
        pixel_offset = 0

        for i in range(num_items):
            # Determine number of crops/frames for this item
            if num_items_info is not None and not is_video:
                num_pixels = num_items_info[i].item()
            elif is_video and grids is not None:
                num_pixels = grids[i][0].item()
            else:
                num_pixels = pixel_values.shape[0] // num_items

            item_pixels = pixel_values[pixel_offset : pixel_offset + num_pixels]
            pixel_offset += num_pixels

            # Extract pooling slice
            pooling = None
            if token_pooling is not None and grids is not None:
                grid = grids[i]
                if is_video:
                    num_pooled = grid[0].item() * grid[1].item() * grid[2].item()
                else:
                    resized_h, resized_w, height, width = grid.tolist()
                    num_pooled = resized_h * resized_w + height * width
                pooling = token_pooling[pooling_offset : pooling_offset + num_pooled]
                pooling_offset += num_pooled
            elif token_pooling is not None:
                pooling = token_pooling

            item = MultimodalDataItem(modality=modality)
            item.feature = item_pixels
            item.offsets = offsets_per_item[i] if i < len(offsets_per_item) else []

            pooling_key = "video_token_pooling" if is_video else "image_token_pooling"
            grid_key = "video_grid" if is_video else "image_grid"
            item.model_specific_data = {pooling_key: pooling}
            if grids is not None:
                item.model_specific_data[grid_key] = grids[i]

            items.append(item)

        return items

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Image.Image]]],
        audio_data,
        input_text,
        request_obj,
        max_req_input_len: Optional[int] = None,
        **kwargs,
    ):
        """Process multimodal data for Molmo2."""
        video_data = getattr(request_obj, "video_data", None)

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )

        if not base_output.images and not base_output.videos:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return {"input_ids": input_ids.tolist(), "mm_items": []}

        processor_kwargs = {}
        if base_output.images:
            processor_kwargs["images"] = base_output.images
        if base_output.videos:
            processor_kwargs["videos"] = base_output.videos

        result = self._processor(
            text=base_output.input_text,
            return_tensors="pt",
            **processor_kwargs,
        )

        input_ids = result["input_ids"].flatten()
        assert self.image_patch_id is not None
        all_offsets = self.get_mm_items_offset(input_ids, self.image_patch_id)
        mm_items = []

        if "pixel_values" in result:
            mm_items.extend(
                self._process_media_items(
                    pixel_values=result["pixel_values"],
                    token_pooling=result.get("image_token_pooling"),
                    grids=result.get("image_grids"),
                    num_items_info=result.get("image_num_crops"),
                    all_offsets=all_offsets,
                    modality=Modality.IMAGE,
                    is_video=False,
                ),
            )

        if "pixel_values_videos" in result:
            mm_items.extend(
                self._process_media_items(
                    pixel_values=result["pixel_values_videos"],
                    token_pooling=result.get("video_token_pooling"),
                    grids=result.get("video_grids"),
                    num_items_info=None,
                    all_offsets=all_offsets,
                    modality=Modality.VIDEO,
                    is_video=True,
                ),
            )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.image_patch_id,
            "im_start_id": self.image_start_token_id,
            "im_end_id": self.image_end_token_id,
        }
