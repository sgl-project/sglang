from typing import List, Union

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.deepseek_janus_pro import MultiModalityCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class JanusProImageProcessor(BaseMultimodalProcessor):
    models = [MultiModalityCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        processor = self._processor

        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=processor.image_token
            ),
            max_req_input_len=max_req_input_len,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(base_out)

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_start_id": processor.image_start_id,
            "im_end_id": processor.image_end_id,
            "im_token_id": processor.image_id,
        }
