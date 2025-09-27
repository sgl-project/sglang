import asyncio
from typing import List, Optional, Union

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.openvla import OpenVLAForActionPrediction, PrismaticProcessor
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_audio, load_image, load_video, logger


class OpenVLAImageProcessor(BaseMultimodalProcessor):
    models = [OpenVLAForActionPrediction]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(image_token="<image>").build(
            _processor
        )

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        if not image_data or len(image_data) != 1:
            raise Exception("OpenVLA can only take in a single image")
        image = image_data[0]
        image, image_size = load_image(image)
        image = image.resize((224, 224))
        pixel_value = np.array(image)
        return {
            "origin_input_ids": [None],
            "mm_items": [
                MultimodalDataItem(
                    feature=pixel_value,
                    model_specific_data={
                        "image_sizes": [(224, 224)],
                    },
                    modality=Modality.IMAGE,
                )
            ],
        }
