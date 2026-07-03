# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
SGLang Multimodal Processor for MiniMax M2/M3 VL.

HF-compatible processor classes (MiniMaxVLProcessor, MiniMaxM2VLImageProcessor,
MiniMaxM2VLVideoProcessor) live in sglang.srt.configs.minimax_vl_processor to
avoid circular imports with model classes.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torchvision.transforms import InterpolationMode

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import round_up


def get_hw_multiple_of(
    image_size: Tuple[int, int],
    multiple: int,
    max_size: Union[None, int, Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """
    Calculate target size that is multiple of given factor.
    Behavior depends on the type of max_size.

    Args:
        image_size: Original (width, height)
        multiple: Alignment factor (patch_size * spatial_merge_size)
        max_size: None, frame_max_size, or (max_w, max_h)

    Returns:
        Tuple[int, int]: (new_width, new_height) both divisible by multiple
    """
    w, h = image_size

    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        # ceil-align the rounded dims to the patch multiple
        new_w = round_up(round(w * ratio), multiple)
        new_h = round_up(round(h * ratio), multiple)
        return new_w, new_h

    # Round up to nearest multiple
    new_w = round_up(w, multiple)
    new_h = round_up(h, multiple)

    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0

        if new_w > max_w or new_h > max_h:
            # Scale down to fit within max_size while maintaining aspect ratio
            new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
            new_w = new_w_
            new_h = new_h_

            # Re-align to multiple
            new_w = (
                new_w
                if new_w % multiple == 0
                else new_w + (multiple - new_w % multiple)
            )
            new_h = (
                new_h
                if new_h % multiple == 0
                else new_h + (multiple - new_h % multiple)
            )

        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h

    return new_w, new_h


def _compute_sampled_frame_indices(
    total_frames: int,
    video_fps: float,
    fps: float,
    max_frames: Optional[int] = None,
) -> List[int]:
    """
    Pick frame indices that match the SFT extract_frame.py constant-mode
    behavior: keep frames whose timestamp is at least read_time_interval
    seconds (= 1/fps) apart from the previously kept frame, then always
    append the last frame if it was not already kept.
    """
    if total_frames <= 0 or video_fps <= 0 or fps <= 0:
        return [0] if total_frames > 0 else []

    read_time_interval = 1.0 / fps
    eps = 1e-4

    indices: List[int] = []
    prev_kept_ts = -float("inf")
    while True:
        if not indices:
            target_frame = 0
        else:
            target_ts = prev_kept_ts + read_time_interval - eps
            target_frame = math.ceil(target_ts * video_fps)
            target_frame = max(target_frame, indices[-1] + 1)
        if target_frame >= total_frames:
            break
        indices.append(target_frame)
        prev_kept_ts = target_frame / video_fps

    last_frame_idx = total_frames - 1
    last_ts = last_frame_idx / video_fps
    if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
        indices.append(last_frame_idx)

    if not indices:
        indices = [0]
    if max_frames is not None and len(indices) > max_frames > 0:
        last = indices[-1]
        if max_frames == 1:
            # max_frames == 1 would divide by (max_frames - 1) == 0 below;
            # keep only the last frame, matching the always-keep-last invariant.
            indices = [last]
        else:
            step = len(indices) / (max_frames - 1)
            indices = [indices[int(i * step)] for i in range(max_frames - 1)]
            indices.append(last)
    return indices


async def get_video_tensor(
    vr,
    image_factor: int,
    max_size: Tuple[int, int],
    fps: Optional[float] = None,
    frame_max_size: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """Sample/resize one MiniMax video and return text-expansion metadata."""
    if fps is None:
        fps = 1.0
    if frame_max_size is None:
        frame_max_size = max_size[0]
    if fps <= 0:
        raise ValueError(f"video fps must be > 0, got {fps}")

    if isinstance(vr, torch.Tensor):
        # data:video/jpeg fallback: frames are already selected; no timestamps.
        video_tchw = vr
        _, _, height, width = video_tchw.shape
        resized_width, resized_height = get_hw_multiple_of(
            (width, height), image_factor, max_size
        )
        resized = torchvision.transforms.functional.resize(
            video_tchw,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        return resized, {
            "total_num_frames": resized.shape[0],
            "fps": None,
            "frames_indices": None,
        }

    total_frames = len(vr)
    video_fps = vr.avg_fps
    if video_fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Invalid video metadata: fps={video_fps}, frames={total_frames}"
        )
    indices = _compute_sampled_frame_indices(total_frames, video_fps, fps, max_frames)
    video_tchw = vr.get_frames_as_tensor(indices)
    # NHWC uint8 -> NCHW float
    video_tchw = video_tchw.permute(0, 3, 1, 2).float()

    _, _, height, width = video_tchw.shape
    resized_width, resized_height = get_hw_multiple_of(
        (width, height), image_factor, frame_max_size
    )
    resized = torchvision.transforms.functional.resize(
        video_tchw,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
    )
    return resized, {
        "total_num_frames": total_frames,
        "fps": video_fps,
        "frames_indices": indices,
    }


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
        # Newer M3 video processors (transformers BaseVideoProcessor, Qwen2VL-style)
        # express their resize budget as a max_pixels area, not the older
        # max_size / _max_size_from_size per-dimension API. Derive an equivalent
        # square (max_w, max_h) cap from max_pixels so the pre-resize never exceeds
        # the HF processor's smart_resize area budget.
        max_size = getattr(video_processor, "max_size", None)
        if max_size is None:
            max_pixels = getattr(video_processor, "max_pixels", None)
            if max_pixels is not None:
                side = int(math.isqrt(int(max_pixels)))
                side -= side % image_factor
                max_size = (side, side)
            else:
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
        self.video_fps = self.video_config.pop("fps", None)
        self.video_frame_max_size = self.video_config.pop("frame_max_size", None)
        self.video_max_frames = self.video_config.pop("max_frames", None)

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

        base_output = await self.load_mm_data(
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
                    max_frames=self.video_max_frames,
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
