# Adapted from https://huggingface.co/stepfun-ai/GOT-OCR2_0

from typing import List, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.got_ocr2 import GOTQwenForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class GOTOCR2Processor(BaseMultimodalProcessor):
    """GOT-OCR2_0 ships no HF image processor, so reproduce its
    `GOTImageEvalProcessor`: bicubic resize to a fixed 1024x1024, then CLIP
    normalization. Every image costs exactly `image_token_len` tokens."""

    models = [GOTQwenForCausalLM]
    gpu_image_decode = False

    IMAGE_SIZE = 1024
    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    IMAGE_PLACEHOLDER_TOKEN = "<image>"
    IMG_START = "<img>"
    IMG_END = "</img>"
    IMG_PAD = "<imgpad>"

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)

        self.image_token_len = hf_config.image_token_len
        self.img_start_token_id = hf_config.im_start_token
        self.img_end_token_id = hf_config.im_end_token
        self.img_pad_token_id = hf_config.im_patch_token

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_PLACEHOLDER_TOKEN,
            image_token_id=self.img_pad_token_id,
        ).build(_image_processor)

    def _to_pixel_values(self, image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize(
            (self.IMAGE_SIZE, self.IMAGE_SIZE), Image.BICUBIC
        )
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.IMAGE_MEAN).view(-1, 1, 1)
        std = torch.tensor(self.IMAGE_STD).view(-1, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        pixel_values = [self._to_pixel_values(img) for img in base_output.images]

        image_tokens = (
            self.IMG_START + self.IMG_PAD * self.image_token_len + self.IMG_END
        )
        input_text_updated = base_output.input_text
        for _ in pixel_values:
            input_text_updated = input_text_updated.replace(
                self.IMAGE_PLACEHOLDER_TOKEN, image_tokens, 1
            )

        input_ids = self._tokenizer.encode(input_text_updated)
        input_ids_tensor = torch.tensor(input_ids)

        items = []
        if pixel_values:
            offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor,
                mm_token_id=self.img_pad_token_id,
            )
            for i, feature in enumerate(pixel_values):
                items.append(
                    MultimodalDataItem(
                        feature=feature,
                        modality=Modality.IMAGE,
                        offsets=[offsets[i]],
                    )
                )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=items,
            im_start_id=self.img_start_token_id,
            im_end_id=self.img_end_token_id,
            im_token_id=self.img_pad_token_id,
        )
