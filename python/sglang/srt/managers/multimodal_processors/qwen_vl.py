from __future__ import annotations

import asyncio
import math
import re
from typing import Dict, List, Union

import torch
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.multimodal_processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen2_5_omni import Qwen2_5OmniModel
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration


# Compatible with Qwen2VL, Qwen2_5VL and Qwen2_5_o
class Qwen2_5VLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5OmniModel,
    ]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        # The single, pre-expanded image token.
        if self.arch == Qwen2_5OmniModel.__name__:
            self.image_token_id = hf_config.thinker_config.image_token_index
            self.image_start_id = hf_config.thinker_config.vision_start_token_id
            self.image_end_id = hf_config.thinker_config.vision_end_token_id
            self.audio_token_id = hf_config.thinker_config.audio_token_index
            self.audio_start_id = hf_config.thinker_config.audio_start_token_id
            self.audio_end_id = hf_config.thinker_config.audio_end_token_id
            self.video_token_id = hf_config.thinker_config.video_token_index
            # TODO: precomputed features might not need pre-processing anymore, try removing this
            self.IMAGE_TOKEN_REGEX = re.compile(
                r"<\|vision_bos\|>(?:<\|IMAGE\|>)+<\|vision_eos\|>"
            )
            self.image_token = "<|vision_bos|><|IMAGE|><|vision_eo|>"
        else:
            self.image_token_id = hf_config.image_token_id
            self.image_start_id = hf_config.vision_start_token_id
            self.image_end_id = hf_config.vision_end_token_id
            self.video_token_id = hf_config.video_token_id
            # The regex that matches expanded image tokens.
            self.IMAGE_TOKEN_REGEX = re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            )
            self.image_token = "<|vision_start|><|image_pad|><|vision_end|>"

        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if isinstance(image_data, str):
            image_data = [image_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token,
                image_token_regex=self.IMAGE_TOKEN_REGEX,
                audio_token=getattr(self, "audio_token_id", None),
                video_token=getattr(self, "video_token_id", None),
            ),
            max_req_input_len=max_req_input_len,
        )

        def smart_resize(
            height: int,
            width: int,
            factor: int = self.IMAGE_FACTOR,
            min_pixels: int = self.MIN_PIXELS,
            max_pixels: int = self.MAX_PIXELS,
        ) -> tuple[int, int]:
            """
            Rescales the image so that the following conditions are met:

            1. Both dimensions (height and width) are divisible by 'factor'.

            2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

            3. The aspect ratio of the image is maintained as closely as possible.
            """
            if max(height, width) / min(height, width) > self.MAX_RATIO:
                raise ValueError(
                    f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
                )
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = math.sqrt((height * width) / max_pixels)
                h_bar = floor_by_factor(height / beta, factor)
                w_bar = floor_by_factor(width / beta, factor)
            elif h_bar * w_bar < min_pixels:
                beta = math.sqrt(min_pixels / (height * width))
                h_bar = ceil_by_factor(height * beta, factor)
                w_bar = ceil_by_factor(width * beta, factor)
            return h_bar, w_bar

        def resize_image(image, size_factor: int = self.IMAGE_FACTOR) -> Image.Image:
            width, height = image.size
            min_pixels = self.MIN_PIXELS
            max_pixels = self.MAX_PIXELS
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            return image

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        async def resize_image_async(image):
            return resize_image(image)

        images_are_preprocessed = self.mm_inputs_are_preprocessed(base_output.images)
        if base_output.images and not images_are_preprocessed:
            resize_tasks = [resize_image_async(image) for image in base_output.images]
            base_output.images = await asyncio.gather(*resize_tasks)

        ret = self.process_mm_data(
            input_text=base_output.input_text,
            images=None if images_are_preprocessed else base_output.images,
            audio=base_output.audios,
        )

        input_ids = ret["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids, mm_token_id=self.image_token_id
        )
        image_grid_thw = None
        video_grid_thw = None  # TODO
        items = []

        if base_output.images:
            if images_are_preprocessed:
                image_grid_thw = self._extract_processor_features(
                    base_output.images, "image_grid_thws"
                )
                precomputed_features = self._extract_processor_features(
                    base_output.images, "precomputed_features"
                )
                pixel_values = self._extract_processor_features(
                    base_output.images, "pixel_values"
                )
            else:
                image_grid_thw = ret["image_grid_thw"]
                pixel_values = ret["pixel_values"]
                precomputed_features = None
            items += [
                MultimodalDataItem(
                    pixel_values=pixel_values,
                    image_grid_thws=image_grid_thw,
                    video_grid_thws=video_grid_thw,
                    precomputed_features=precomputed_features,
                    image_offsets=image_offsets,
                    modality=Modality.IMAGE,
                )
            ]

        if "input_features" in ret and ret["input_features"] is not None:
            audio_offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=getattr(self, "audio_token_id", None),
            )
            item = MultimodalDataItem(
                audio_features=ret["input_features"],
                feature_attention_mask=ret["feature_attention_mask"],
                attention_mask=ret["attention_mask"],
                # TODO: unify feature and offsets across modalities
                audio_offsets=audio_offsets,
                modality=Modality.AUDIO,
            )
            items += [item]

        if self.hf_config.model_type == "qwen2_5_omni":
            feature_attention_mask = ret.get("feature_attention_mask", None)
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            else:
                audio_feature_lengths = None
            mrope_positions, mrope_position_delta = (
                MRotaryEmbedding.get_rope_index_omni(
                    input_ids=input_ids.unsqueeze(0),
                    config=self.hf_config.thinker_config,
                    image_grid_thw=ret.get("image_grid_thw", None),
                    video_grid_thw=ret.get("video_grid_thw", None),
                    audio_seqlens=audio_feature_lengths,
                    second_per_grids=ret.get("second_per_grids", None),
                )
            )
        else:
            mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.image_start_id,
                model_type=self.hf_config.model_type,
                tokens_per_second=getattr(
                    self.hf_config.vision_config, "tokens_per_second", None
                ),
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=ret.get("second_per_grid_ts", None),
            )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": items,
            "im_start_id": self.image_start_id,
            "im_end_id": self.image_end_id,
            "im_token_id": self.image_token_id,
            "audio_start_id": getattr(self, "audio_start_id", None),
            "audio_end_id": getattr(self, "audio_end_id", None),
            "audio_token_id": getattr(self, "audio_token_id", None),
            "video_token_id": self.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
