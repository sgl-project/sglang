# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
SGLang Multimodal Processor for MiniMax M2/M3 VL.

HF-compatible processor classes (MiniMaxVLProcessor, MiniMaxM2VLImageProcessor,
MiniMaxM2VLVideoProcessor) live in sglang.srt.configs.minimax_vl_processor to
avoid circular imports with model classes.
"""

import re
from typing import Dict, List, Optional

import pybase64
import torch
from torchvision.io import decode_image

from sglang.srt.configs.minimax_vl_processor import (
    get_video_tensor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.minimax_m3_vl import (
    MiniMaxM3SparseForConditionalGeneration
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData

# ==============================================================================
# SGLang Multimodal Processor
# ==============================================================================


class MiniMaxM3VLProcessor(BaseMultimodalProcessor):
    """
    SGLang Multimodal Processor for MiniMax M3 VL.

    Uses local MiniMaxM2VL{Image,Video}Processor classes (copied from Qwen2VL)
    with resize logic changed to vLLM's get_hw_multiple_of.
    """

    models = [
        MiniMaxM3SparseForConditionalGeneration,
    ]

    # Local image processor PIL images or tensors.
    gpu_image_decode = False

    # Whether to use padding when tokenizing text in process_mm_data.
    # M3's tokenizer does not have a pad_token, so disable padding.
    tokenizer_padding = False


    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    IMAGE_START_TOKEN = "]<]start of image[>["
    IMAGE_END_TOKEN = "]<]end of image[>["

    @staticmethod
    def _token_id(tokenizer, token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None, f"token id for {token!r} not found"
        return token_id

    @property
    def spatial_merge_size(self):
        return self._processor.image_processor.merge_size

    def _video_resize_config(self):
        video_processor = self._processor.video_processor
        image_factor = video_processor.patch_size * video_processor.merge_size
        max_size = video_processor.max_size
        if max_size is None:
            max_size = video_processor._max_size_from_size(video_processor.size)
        assert max_size is not None, "video processor max_size is required"
        return image_factor, max_size

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        tokenizer = _processor.tokenizer
        assert tokenizer is not None, "tokenizer is required"

        self.IM_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_TOKEN)
        self.VIDEO_TOKEN_ID = self._token_id(tokenizer, self.VIDEO_TOKEN)
        self.IM_START_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_START_TOKEN)
        self.IM_END_TOKEN_ID = self._token_id(tokenizer, self.IMAGE_END_TOKEN)
        self.video_fps = server_args.mm_process_config.get("video_fps")
        self.video_frame_max_size = server_args.mm_process_config.get(
            "video_frame_max_size"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=re.compile(
                r"<image>|<\|image\|>|<\|image_pad\|>|\]\<\]image\[\>\["
            ),
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
            video_token_regex=re.compile(r"<video>|<\|video\|>|\]\<\]video\[\>\["),
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: Optional[List],
        audio_data: Optional[List],  # Not used
        input_text: str,
        request_obj,
        **kwargs,
    ) -> Dict:
        """
        Process multimodal data asynchronously.

        Following qwen_vl.py pattern:
        1. load_mm_data() - load raw images/videos
        2. get_video_tensor() - resize videos only (no normalize)
        3. process_and_combine_mm_data() - call local processors for full preprocessing

        Args:
            image_data: List of image sources
            audio_data: Not used (no audio support)
            input_text: Input text with placeholders
            request_obj: Request object with video_data

        Returns:
            Dict with input_ids, mm_items, token IDs
        """

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Step 2: Sample + resize videos. Sampling/resize knobs come from
        # the global MiniMax video config, not per-request API extensions.
        video_metadata = None
        if base_output.videos:
            image_factor, max_size = self._video_resize_config()
            videos_processed = [
                await get_video_tensor(
                    video,
                    image_factor=image_factor,
                    max_size=max_size,
                    fps=self.video_fps,
                    frame_max_size=self.video_frame_max_size,
                )
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        # Step 3: Call base process_and_combine_mm_data which uses self._processor
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
            video_metadata=video_metadata,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )
