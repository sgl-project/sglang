from typing import List, Union

from sglang.srt.models.clip import CLIPModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class ClipImageProcessor(BaseMultimodalProcessor):
    models = [CLIPModel]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self._processor.tokenizer.image_token
            ),
            image_data=image_data,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_output)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
        }
