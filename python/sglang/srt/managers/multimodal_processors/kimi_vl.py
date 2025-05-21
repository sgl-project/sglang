import re
import torch
from typing import List, Union, Dict


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
                image_token_pattern=self.IMAGE_TOKEN_REGEX
            ),
            max_req_input_len=max_req_input_len,
        )
        images_are_preprocessed = self.mm_inputs_are_preprocessed(base_output.images)

        ret = self.process_mm_data(
            input_text=base_output.input_text,
            images=None if images_are_preprocessed else base_output.images,
        )

        if base_output.images:
            if images_are_preprocessed:
                image_grid_hws = torch.concat(
                    [
                        torch.as_tensor(item.image_grid_thws)
                        for item in base_output.images
                    ]
                )
                precomputed_features = torch.concat(values) if (values := [
                    item.precomputed_features
                    for item in base_output.images
                    if item.precomputed_features is not None
                ]) else None
                pixel_values = None
            else:
                image_grid_hws = ret["image_grid_hws"]
                pixel_values = ret["pixel_values"]
                precomputed_features = None

        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=pixel_values,
                    image_grid_thws=image_grid_hws,
                    precomputed_features=precomputed_features,
                    modality=Modality.IMAGE,
                )
            ],
            "im_token_id": self.im_token_id,
        }
