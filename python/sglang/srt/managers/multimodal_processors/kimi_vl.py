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
        self.IM_TOKEN_ID = _processor.tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

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

        combined_mm_item, input_ids = self.process_and_combine_mm_data(base_output)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": [combined_mm_item] if combined_mm_item is not None else [],
            "im_token_id": self.IM_TOKEN_ID,
        }
