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

        images_are_preprocessed = self.mm_inputs_are_preprocessed(base_output.images)
        ret = self.process_mm_data(
            input_text=base_output.input_text,
            images=None if images_are_preprocessed else base_output.images,
        )

        items = []
        input_ids = ret["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=self.hf_config.image_token_index,
        )
        for i, image in enumerate(base_output.images):
            if images_are_preprocessed:
                pixel_values = image.pixel_values
                precomputed_features = image.precomputed_features
            else:
                pixel_values = ret["pixel_values"][i]
                precomputed_features = None

            item = MultimodalDataItem(
                pixel_values=pixel_values,
                precomputed_features=precomputed_features,
                modality=Modality.IMAGE,
                image_offsets=image_offsets[i],
            )
            items += [item]

        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
        }
