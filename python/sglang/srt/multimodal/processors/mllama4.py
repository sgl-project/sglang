from typing import List, Union

from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Mllama4ImageProcessor(BaseMultimodalProcessor):
    models = [Llama4ForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IM_TOKEN = self._processor.image_token
        self.IM_TOKEN_ID = self._processor.image_token_id

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        max_req_input_len,
        *args,
        **kwargs
    ):
        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=self.IM_TOKEN),
            max_req_input_len=max_req_input_len,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_out)

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.IM_TOKEN_ID,
        }
