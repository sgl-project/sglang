# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image and video processor for Bailing multimodal checkpoints."""

from typing import Optional

import torch
from transformers import BaseImageProcessor

from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.bailing_mm import (
    BailingMM2NativeForConditionalGeneration,
    BailingMMNativeForConditionalGeneration,
)
from sglang.srt.models.bailing_mm_v3 import BailingMoeV3VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData, VideoData

DEFAULT_IMAGE_PATCH_TOKEN = "<|image_pad|>"
DEFAULT_FRAME_PATCH_TOKEN = "<|video_pad|>"
DEFAULT_VISION_START_TOKEN = "<|vision_start|>"
DEFAULT_VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_VIDEO_START_TOKEN = "<|video_start|>"
DEFAULT_VIDEO_END_TOKEN = "<|video_end|>"


class BailingMMMultimodalProcessor(BaseMultimodalProcessor):
    """Prepare image/video features and Bailing three-axis positions."""

    models = [
        BailingMMNativeForConditionalGeneration,
        BailingMM2NativeForConditionalGeneration,
        BailingMoeV3VLForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        tokenizer = getattr(_processor, "tokenizer", _processor)

        self.image_token_id = self._resolve_token_id(
            hf_config,
            tokenizer,
            "image_token_id",
            "image_patch_token",
            DEFAULT_IMAGE_PATCH_TOKEN,
        )
        self.video_token_id = self._resolve_token_id(
            hf_config,
            tokenizer,
            "video_token_id",
            "video_patch_token",
            DEFAULT_FRAME_PATCH_TOKEN,
        )
        image_token = self._wrapped_token(
            _processor,
            tokenizer,
            ("vision_start_token", "vision_bos_token"),
            DEFAULT_VISION_START_TOKEN,
            ("image_token",),
            DEFAULT_IMAGE_PATCH_TOKEN,
            ("vision_end_token", "vision_eos_token"),
            DEFAULT_VISION_END_TOKEN,
        )
        video_token = self._wrapped_token(
            _processor,
            tokenizer,
            ("video_start_token", "video_bos_token"),
            DEFAULT_VIDEO_START_TOKEN,
            ("video_token",),
            DEFAULT_FRAME_PATCH_TOKEN,
            ("video_end_token", "video_eos_token"),
            DEFAULT_VIDEO_END_TOKEN,
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            video_token=video_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        ).build(_processor)

    @staticmethod
    def _resolve_token_id(
        hf_config,
        tokenizer,
        config_attr: str,
        text_config_attr: str,
        fallback_token: str,
    ) -> int:
        token_id = getattr(hf_config, config_attr, None)
        if token_id is not None:
            return token_id
        for attr in ("text_config", "llm_config"):
            text_config = getattr(hf_config, attr, None)
            token_id = getattr(text_config, text_config_attr, None)
            if token_id is not None:
                return token_id
        return tokenizer.convert_tokens_to_ids(fallback_token)

    @staticmethod
    def _token_string(processor, tokenizer, attrs, fallback: str) -> str:
        for source in (processor, tokenizer):
            for attr in attrs:
                token = getattr(source, attr, None)
                if token is not None:
                    return token
        return fallback

    @classmethod
    def _wrapped_token(
        cls,
        processor,
        tokenizer,
        start_attrs,
        start_fallback,
        token_attrs,
        token_fallback,
        end_attrs,
        end_fallback,
    ) -> str:
        return (
            cls._token_string(processor, tokenizer, start_attrs, start_fallback)
            + cls._token_string(processor, tokenizer, token_attrs, token_fallback)
            + cls._token_string(processor, tokenizer, end_attrs, end_fallback)
        )

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ) -> dict:
        if audios:
            raise ValueError("Audio inputs are not supported by Ling-3.0-flash-VL")
        processor, _ = self._resolve_processor(processor)
        processor_kwargs = {
            "text": [input_text],
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        if videos:
            processor_kwargs["videos"] = videos
        image_processor = getattr(processor, "image_processor", None)
        device: Optional[str] = None
        if isinstance(image_processor, BaseImageProcessor):
            device = self._fast_image_processor_device(processor)
        if device is not None:
            processor_kwargs["device"] = device

        result = processor(**processor_kwargs)
        for feature_name in self.FEATURE_NAMES:
            feature = result.get(feature_name)
            if not isinstance(feature, torch.Tensor):
                continue
            feature = feature.to(dtype=torch.bfloat16)
            if not self.keep_mm_features_on_device:
                feature = feature.cpu()
            result[feature_name] = feature
        return result

    @staticmethod
    def _request_url(item):
        if isinstance(item, (ImageData, VideoData)):
            return item.url
        if isinstance(item, dict):
            if "url" not in item:
                raise ValueError("Bailing media dictionaries must contain a url")
            return item["url"]
        return item

    def _processor_fetch_mm_input(self, prompt, image_data, video_data):
        if isinstance(prompt, list):
            if not prompt or not isinstance(prompt[0], int):
                raise ValueError("Tokenized Bailing prompts must be a non-empty list")
            prompt = self._tokenizer.decode(prompt)
        if not isinstance(prompt, str):
            raise TypeError(
                f"Bailing prompt must be str or list[int], got {type(prompt)}"
            )

        contents = []
        for item in image_data or []:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._request_url(item)},
                }
            )
        for item in video_data or []:
            contents.append(
                {
                    "type": "video_url",
                    "video_url": {"url": self._request_url(item)},
                }
            )
        images, videos, audios = self._processor.process_vision_info(
            conversations=[{"content": contents}]
        )
        if audios:
            raise ValueError("Audio inputs are not supported by Ling-3.0-flash-VL")
        return BaseMultiModalProcessorOutput(
            images=images or [],
            videos=videos or [],
            audios=[],
            input_text=prompt,
        )

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        if audio_data or getattr(request_obj, "audio_data", None):
            raise ValueError("Audio inputs are not supported by Ling-3.0-flash-VL")
        base_output = self._processor_fetch_mm_input(
            input_text,
            image_data,
            getattr(request_obj, "video_data", None),
        )
        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                self.hf_config,
                image_grid_thw=ret.get("image_grid_thw"),
                video_grid_thw=ret.get("video_grid_thw"),
            )
        )
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            mrope_positions=mrope_positions.squeeze(1),
            mrope_position_delta=mrope_position_delta,
        )
