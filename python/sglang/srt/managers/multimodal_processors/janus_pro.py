from typing import List, Union

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.deepseek_janus_pro import MultiModalityCausalLM


class JanusProImageProcessor(BaseMultimodalProcessor):
    models = [MultiModalityCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    def process_images_task(self, images, input_text):
        processor = self._processor
        result = processor.__call__(
            prompt=input_text, images=images, return_tensors="pt", device="cuda"
        )
        return {
            "input_ids": result["input_ids"],
            "pixel_values": result["pixel_values"],
            "images_emb_mask": result["images_emb_mask"],
            "im_start_id": processor.image_start_id,
            "im_end_id": processor.image_end_id,
            "im_token_id": processor.image_id,
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        if not image_data:
            return None

        if not isinstance(image_data, list):
            image_data = [image_data]

        base_out = self.load_mm_data(
            prompt=input_ids,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token="<image_placeholder>"
            ),
            max_req_input_len=max_req_input_len,
        )
        images = base_out.images
        res = self.process_images_task(images=images, input_text=base_out.input_text)
        return {
            "items": [
                MultimodalDataItem(
                    pixel_values=res["pixel_values"],
                    image_emb_mask=res["images_emb_mask"],
                    modality="image",
                )
            ],
            "input_ids": res["input_ids"].flatten().tolist(),
            "im_start_id": res["im_start_id"],
            "im_end_id": res["im_end_id"],
            "im_token_id": res["im_token_id"],
        }
