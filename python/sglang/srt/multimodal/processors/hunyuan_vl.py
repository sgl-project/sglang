import math
import os
import re
import time
from typing import List, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import XDRotaryEmbedding
from sglang.srt.models.hunyuan_vl import HunYuanVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.utils import logger

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 16384 * 32 * 32
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )

VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 32 * 32 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 32 * 32
VIDEO_MAX_PIXELS = 768 * 32 * 32
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# Compatible with HunYuan VL
class HunYuanImageProcessor(SGLangBaseProcessor):
    models = [HunYuanVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.image_token_id = hf_config.image_token_id
        self.im_start_token_id = hf_config.image_start_token_id
        self.im_end_token_id = hf_config.image_end_token_id

        self.image_config = server_args.mm_process_config.get("image", {})
        self.video_config = server_args.mm_process_config.get("video", {})

        self.vision_config = hf_config.vision_config
        self.spatial_merge_size = self.vision_config.spatial_merge_size
        
        self.rope_scaling = hf_config.rope_scaling
        if self.rope_scaling is not None and self.rope_scaling.get("xdrope_section", None) is not None:
            self.xd_num = len(self.rope_scaling["xdrope_section"])

        self.mm_tokens = MultimodalSpecialTokens(
            # image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token="<｜hy_place▁holder▁no▁102｜>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            # image_token_regex=re.compile(
                # r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
                # r"<｜hy_place▁holder▁no▁102｜>"  
            # ),
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.im_start_token_id,
            "im_end_id": self.im_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "xdrope_positions": ret['position_ids'].squeeze(0),
        }
