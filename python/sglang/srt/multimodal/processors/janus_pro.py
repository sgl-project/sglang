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

        images = base_out.images
        res = self.process_mm_data(
            input_text=base_out.input_text,
            prompt=base_out.input_text,
            images=images,
        )

        input_ids = res["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids, mm_token_id=processor.image_id
        )
        return {
            "mm_items": [
                MultimodalDataItem(
                    pixel_values=res["pixel_values"],
                    image_emb_mask=res["images_emb_mask"],
                    offsets=image_offsets,
                    modality=Modality.IMAGE,
                )
            ],
            "input_ids": input_ids.tolist(),
            "im_start_id": processor.image_start_id,
            "im_end_id": processor.image_end_id,
            "im_token_id": processor.image_id,
        }
