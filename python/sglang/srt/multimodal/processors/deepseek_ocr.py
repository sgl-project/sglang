from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class DeepseekOCRProcessor(BaseMultimodalProcessor):
    models = [DeepseekOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        _processor.image_size = 640
        _processor.ocr2_mode = (
            str(
                getattr(getattr(hf_config, "vision_config", None), "model_name", "")
            ).lower()
            == "deepencoderv2"
            or getattr(getattr(hf_config, "projector_config", None), "input_dim", None)
            == 896
        )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    def process_mm_data(self, input_text, images=None, **kwargs):
        # tokenize_with_images (deepseekvl2) requires PIL Images, not tensors.
        # Convert any GPU-decoded tensors (CHW uint8) to PIL before processing.
        if images:
            converted = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    from torchvision.transforms.functional import to_pil_image

                    img = to_pil_image(img.cpu())
                converted.append(img)
            images = converted
        return super().process_mm_data(input_text, images=images, **kwargs)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
