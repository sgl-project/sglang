from typing import List, Optional, Union

from python.sglang.srt.layers.openvla import PrismaticProcessor
from python.sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.openvla import OpenVLAForActionPrediction
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_audio, load_image, load_video, logger
import asyncio
import torch
import numpy as np

class OpenVLAImageProcessor(BaseMultimodalProcessor):
    models = [OpenVLAForActionPrediction]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.mm_tokens = MultimodalSpecialTokens(image_token="<image>").build(
            _processor
        )

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        if not image_data or len(image_data)!=1:
            raise Exception("OpenVLA can only take in a single image")
        image = image_data[0]
        image, image_size = load_image(image)
        image = image.resize((224, 224))
        # pixel_value = self._processor.process_image(image).to(torch.bfloat16).to(0)
        pixel_value = np.array(image)
        return {
            "origin_input_ids": [
                None
            ],
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
