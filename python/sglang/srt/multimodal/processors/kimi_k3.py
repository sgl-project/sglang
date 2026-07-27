# SPDX-License-Identifier: Apache-2.0
"""Multimodal processor registration for Kimi-K3."""

import re
from typing import Dict, List, Union

import torch

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin


class KimiK3ImageProcessor(KimiGridMMDataMixin, BaseMultimodalProcessor):
    models = [KimiK3ForConditionalGeneration]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(
            hf_config, server_args, _processor, *args, **kwargs
        )
        image_token = getattr(
            hf_config, "image_placeholder", "<|kimi_image_placeholder|>"
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(
                rf"(?:{re.escape(image_token)})+"
            ),
        ).build(_processor)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        if videos or audios:
            raise ValueError("Kimi-K3 currently supports image inputs only")
        images = images or []
        medias = [{"type": "image", "image": image} for image in images]
        result = self._processor(
            medias=medias,
            text=input_text,
            return_tensors="pt",
        )

        grid_thws = result.pop("grid_thws")
        image_token_counts = [
            self._num_image_tokens_from_grid(grid) for grid in grid_thws
        ]
        input_ids = result["input_ids"].flatten().tolist()
        expanded_input_ids = self._expand_input_ids(
            input_ids,
            image_token_counts,
            self.mm_tokens.image_token_id,
        )
        result["input_ids"] = torch.tensor(
            [expanded_input_ids], dtype=torch.long
        )
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        # Use the standard SGLang key so bundled images are split by patch
        # ranges and each item retains its own [t, h, w] metadata.
        result["image_grid_thw"] = grid_thws
        if not self.server_args.keep_mm_feature_on_device:
            result["pixel_values"] = result["pixel_values"].cpu()
            result["image_grid_thw"] = result["image_grid_thw"].cpu()
        return result

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        grid_thws = kwargs.get(
            "img_grid_thw", kwargs.get("grid_thws", None)
        )
        return self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=grid_thws,
        )
