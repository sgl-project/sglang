import re
from typing import Dict, List, Union

from sglang.srt.models.kimi_k2_vl import K2VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


# Compatible with KimiVLForConditionalGeneration
class KimiK2_5VLImageProcessor(SGLangBaseProcessor):
    models = [K2VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            # TODO: could we convert in MultimodalSpecialTokens?
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        print(f"32 image_data: {image_data}")
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        print(f"39 base_output: {base_output}")
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        print(f"43 mm_items: {mm_items}")
        print(f"44 input_ids: {input_ids}")
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
        }
