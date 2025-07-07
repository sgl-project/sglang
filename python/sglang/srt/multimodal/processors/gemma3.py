import re
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/image_processing_gemma3_fast.py
# will be removed in the future


class Gemma3SGLangImageProcessor(SGLangBaseProcessor):
    models = [Gemma3ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        # The single, pre-expanded image token.
        self.IMAGE_TOKEN = "<start_of_image>"
        # The regex that matches expanded image tokens.
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<start_of_image>(?:(?:<image_soft_token>)*<end_of_image>)?"
        )
        self.IM_START_TOKEN_ID = hf_config.boi_token_index
        self.IM_END_TOKEN_ID = hf_config.eoi_token_index
        self.IM_TOKEN_ID = hf_config.image_token_index

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        print(f"{image_data=}")
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.IMAGE_TOKEN, image_token_regex=self.IMAGE_TOKEN_REGEX
            ),
            max_req_input_len=max_req_input_len,
            discard_alpha_channel=True,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_output)
        print(f"{base_output=}")
        print(f"{mm_items=}")
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }
