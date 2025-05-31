import re
from typing import Any, Dict, List, Optional, Union

import torch

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
        self.IMAGE_TOKEN = "<|media_pad|>"
        self.IMAGE_TOKEN_REGEX = re.compile(r"(?:<\|media_pad\|>)+")
        self.im_token_id = _processor.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN, image_token_regex=self.IMAGE_TOKEN_REGEX
            ),
            max_req_input_len=max_req_input_len,
        )

        images_are_preprocessed = self.mm_inputs_are_preprocessed(base_output.images)
        if not images_are_preprocessed:
            ret = self.process_mm_data(
                input_text=base_output.input_text,
                images=base_output.images,
            )
            input_ids = ret["input_ids"].flatten()
            image_grid_thws = ret["image_grid_hws"]
            pixel_values = ret["pixel_values"]
            precomputed_features = None
        else:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

            image_grid_thws = self._extract_processor_features(
                base_output.images, "image_grid_thws"
            )
            precomputed_features = self._extract_processor_features(
                base_output.images, "precomputed_features"
            )
            pixel_values = self._extract_processor_features(
                base_output.images, "pixel_values"
            )

        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=self.im_token_id,
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=pixel_values,
                    image_grid_thws=image_grid_thws,
                    precomputed_features=precomputed_features,
                    modality=Modality.IMAGE,
                    image_offsets=image_offsets,
                )
            ],
            "im_token_id": self.im_token_id,
        }
