import logging
from typing import Any, Dict, Optional

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nemotron_parse import NemotronParseForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import load_image

logger = logging.getLogger(__name__)


class NemotronParseProcessor(BaseMultimodalProcessor):
    models = [NemotronParseForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)

        self.target_height, self.target_width = getattr(
            hf_config, "image_size", [2048, 1648]
        )

    def _resize_with_aspect_ratio(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        aspect_ratio = width / height

        new_height = height
        new_width = width

        if new_height > self.target_height:
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)

        if new_width > self.target_width:
            new_width = self.target_width
            new_height = int(new_width / aspect_ratio)

        if (new_width, new_height) != (width, height):
            image = TF.resize(
                image,
                [new_height, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        return image

    def _pad_to_target(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        if width == self.target_width and height == self.target_height:
            return image

        pad_right = self.target_width - width
        pad_bottom = self.target_height - height
        return TF.pad(
            image,
            [0, 0, pad_right, pad_bottom],
            fill=1.0,
        )

    def _image_to_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return TF.normalize(
            image,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not image_data:
            return None

        if len(image_data) != 1:
            raise ValueError(
                f"Nemotron Parse expects exactly 1 image input, got {len(image_data)}"
            )

        pil_image, _ = load_image(image_data[0])
        pil_image = pil_image.convert("RGB")
        pixel_values = TF.convert_image_dtype(
            TF.pil_to_tensor(pil_image),
            torch.float32,
        )

        pixel_values = self._resize_with_aspect_ratio(pixel_values)
        pixel_values = self._pad_to_target(pixel_values)

        pixel_values = self._image_to_tensor(pixel_values)

        eos_token_id = getattr(self.hf_config, "eos_token_id", 2)
        bos_token_id = getattr(self.hf_config, "bos_token_id", 0)

        predict_bbox_id = self._tokenizer.convert_tokens_to_ids("<predict_bbox>")
        predict_classes_id = self._tokenizer.convert_tokens_to_ids("<predict_classes>")
        output_markdown_id = self._tokenizer.convert_tokens_to_ids("<output_markdown>")

        input_ids = [
            eos_token_id,
            bos_token_id,
            predict_bbox_id,
            predict_classes_id,
            output_markdown_id,
        ]

        return {
            "input_ids": input_ids,
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_values,
                    modality=Modality.IMAGE,
                )
            ],
        }
