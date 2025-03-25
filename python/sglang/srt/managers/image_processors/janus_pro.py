import asyncio
from typing import List, Union

from sglang.srt.managers.image_processors.base_image_processor import (
    BaseImageProcessor as SGLangBaseImageProcessor,
)
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.deepseek_janus_pro import MultiModalityCausalLM


class JanusProProcessor(SGLangBaseImageProcessor):
    models = [MultiModalityCausalLM]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)

    @staticmethod
    def _process_images_task(images, input_text):
        processor = get_global_processor()
        result = processor.__call__(
            prompt=input_text, images=images, return_tensors="pt"
        )
        return {
            "input_ids": result["input_ids"],
            "pixel_values": result["pixel_values"],
            "images_emb_mask": result["images_emb_mask"],
            "im_start_id": processor.image_start_id,
            "im_end_id": processor.image_end_id,
            "im_token_id": processor.image_id,
        }

    async def _process_images(self, images, input_text):
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            image_inputs = await loop.run_in_executor(
                self.executor,
                JanusProProcessor._process_images_task,
                images,
                input_text,
            )
        else:
            image_inputs = self._processor(
                images=images, text=input_text, return_tensors="pt"
            )

        return image_inputs

    async def process_images_async(
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

        base_out = self.load_images(
            input_ids=input_ids,
            image_data=image_data,
            image_token="<image_placeholder>",
            max_req_input_len=max_req_input_len,
        )
        images = base_out.all_frames
        res = await self._process_images(images=images, input_text=base_out.input_text)

        return {
            "input_ids": res["input_ids"].flatten().tolist(),
            "pixel_values": res["pixel_values"],
            "images_emb_mask": res["images_emb_mask"],
            "image_hashes": base_out.image_hashes,
            "im_start_id": res["im_start_id"],
            "im_end_id": res["im_end_id"],
            "im_token_id": res["im_token_id"],
        }
