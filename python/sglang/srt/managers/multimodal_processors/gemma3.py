import re
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration

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

        image_token = self.IMAGE_TOKEN
        image_token_regex = self.IMAGE_TOKEN_REGEX
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=image_token, image_token_regex=image_token_regex
            ),
            max_req_input_len=max_req_input_len,
            discard_alpha_channel=True,
        )

        combined_mm_item = self.get_combined_mm_item(base_output)

        if combined_mm_item is None:
            ret = self.process_mm_data(
                input_text=base_output.input_text,
                images=base_output.images,
            )
            combined_mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                pixel_values=ret["pixel_values"],
            )
            input_ids = ret["input_ids"].flatten()
        else:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        combined_mm_item.image_offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=self.hf_config.image_token_index,
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": [combined_mm_item],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }
