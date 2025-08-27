from typing import List, Union

<<<<<<< HEAD
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_image
=======
from sglang.srt.models.mllama import MllamaForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
>>>>>>> origin/main


class MllamaImageProcessor(BaseMultimodalProcessor):
    models = [MllamaForConditionalGeneration]

<<<<<<< HEAD
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
=======
    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self._processor.image_token,
            image_token_id=self._processor.image_token_id,
        ).build(_processor)
>>>>>>> origin/main

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
<<<<<<< HEAD
        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        images = [load_image(image)[0] for image in image_data]
        image_inputs = self.process_mm_data(input_text=input_text, images=images)
        image_inputs["input_ids"] = image_inputs["input_ids"].tolist()[0]
        image_inputs["mm_items"] = [
            MultimodalDataItem(
                pixel_values=image_inputs["pixel_values"],
                aspect_ratio_id=image_inputs["aspect_ratio_ids"],
                aspect_ratio_mask=image_inputs["aspect_ratio_mask"],
                modality=Modality.IMAGE,
            )
        ]

        return image_inputs
=======
        base_out = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_out, self.mm_tokens
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "im_token_id": self.mm_tokens.image_token_id,
        }
>>>>>>> origin/main
