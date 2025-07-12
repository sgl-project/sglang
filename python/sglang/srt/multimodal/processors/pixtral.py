import asyncio
import math
from typing import List, Union

from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens,
)

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.pixtral import PixtralVisionModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class PixtralProcessor(BaseMultimodalProcessor):
    models = [PixtralVisionModel]

    PAD_TOKEN = "<pad>"
    IMG_BREAK_TOKEN_ID = 12
    IMG_END_TOKEN_ID = 13

    def get_patch_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_width = max_height = self.image_size
        patch_width = patch_height = self.patch_size

        ratio = max(image_width / max_width, image_height / max_height)

        if ratio > 1:
            image_width = int(math.floor(image_width / ratio))
            image_height = int(math.floor(image_height / ratio))

        nrows, ncols = _get_pixtral_hf_num_image_tokens(
            (image_height, image_width),
            (patch_height, patch_width),
        )

        return ncols, nrows

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.image_token_id = getattr(
            hf_config, "image_token_index", PixtralVisionModel.DEFAULT_IMAGE_TOKEN_ID
        )
        # Instantiate the patcher logic helper using the class defined above

        self.vision_config = hf_config.vision_config
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        self.multimodal_tokens = MultimodalSpecialTokens(
            image_token=_processor.image_token
        )
        _processor.tokenizer.add_special_tokens(
            {
                "pad_token": getattr(hf_config, "pad_token", self.PAD_TOKEN),
            }
        )

    async def _resize(self, image):
        num_w_tokens, num_h_tokens = self.get_patch_grid_size(
            image_width=image.size[0],
            image_height=image.size[1],
        )
        new_size = (num_w_tokens * self.patch_size, num_h_tokens * self.patch_size)
        return image.resize(new_size)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        mm_data = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.multimodal_tokens,
            max_req_input_len=kwargs.get("max_req_input_len", 4096),
            image_data=image_data,
            return_text=True,
        )

        if mm_data.images:
            resize_tasks = [self._resize(image) for image in mm_data.images]
            mm_data.images = await asyncio.gather(*resize_tasks)

        processor_output = self.process_mm_data(
            input_text=mm_data.input_text,
            images=mm_data.images,
        )

        if "pixel_values" in processor_output:
            input_ids = processor_output["input_ids"].view(-1)
            image_offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=self.image_token_id,
            )
            mm_items = [
                MultimodalDataItem(
                    pixel_values=processor_output["pixel_values"],
                    image_sizes=processor_output["image_sizes"],
                    modality=Modality.IMAGE,
                    offsets=image_offsets,
                )
            ]

            input_ids = input_ids.tolist()
            processor_output.update(
                input_ids=input_ids,
                mm_items=mm_items,
                # there's no im_start_id for pixtral, only im_token and im_end_token
                im_end_id=self.IMG_END_TOKEN_ID,
                im_token_id=self.image_token_id,
            )
        return processor_output
