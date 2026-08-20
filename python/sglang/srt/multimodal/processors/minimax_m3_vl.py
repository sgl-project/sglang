# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""HF processor classes live in sglang.srt.configs.minimax_vl_processor to avoid circular imports with model classes."""

import math
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData, VideoData, round_up

MIN_SHORT_SIDE_PIXEL = 112
MAX_TOTAL_PIXELS_IMAGE = 12_845_056
MAX_TOTAL_PIXELS_VIDEO = 301_056_000


def resolve_media_size(
    width: int, height: int, max_long_side_pixel: Optional[int]
) -> Tuple[int, int]:
    """Target (width, height) for one image or video frame."""
    long_side, short_side = max(width, height), min(width, height)
    if max_long_side_pixel is not None and long_side > max_long_side_pixel:
        scale = max_long_side_pixel / long_side
    elif short_side < MIN_SHORT_SIDE_PIXEL:
        scale = MIN_SHORT_SIDE_PIXEL / short_side
    else:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def check_total_pixels(width: int, height: int, frames: int, limit: int) -> None:
    total = width * height * frames
    if total > limit:
        raise ValueError(
            f"media exceeds the total pixel limit: {width}x{height}"
            f"{f'x{frames} frames' if frames != 1 else ''} = {total} > {limit}"
        )


def _max_long_side_pixel(item) -> Optional[int]:
    """Read the per-item cap off whichever carrier the request used."""
    if isinstance(item, ImageData):
        return item.max_long_side_pixel
    if isinstance(item, VideoData):
        return (item.preprocess_kwargs or {}).get("max_long_side_pixel")
    if isinstance(item, dict):
        return item.get("max_long_side_pixel")
    return None


def get_hw_multiple_of(
    image_size: Tuple[int, int],
    multiple: int,
    max_size: Union[None, int, Tuple[int, int]] = None,
) -> Tuple[int, int]:
    w, h = image_size

    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        new_w = round_up(round(w * ratio), multiple)
        new_h = round_up(round(h * ratio), multiple)
        return new_w, new_h

    new_w = round_up(w, multiple)
    new_h = round_up(h, multiple)

    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0

        if new_w > max_w or new_h > max_h:
            new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
            new_w = new_w_
            new_h = new_h_

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
    """Frame indices must match SFT extract_frame.py constant-mode sampling (>=1/fps apart, always keep last) or eval diverges from training."""
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
            # max_frames == 1 would divide by (max_frames - 1) == 0 below; keep only the last frame.
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
    if fps is None:
        fps = 1.0
    if frame_max_size is None:
        frame_max_size = max_size[0]
    if fps <= 0:
        raise ValueError(f"video fps must be > 0, got {fps}")

    if isinstance(vr, torch.Tensor):
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


class MiniMaxM3VLProcessor(BaseMultimodalProcessor):
    models = [
        MiniMaxM3SparseForConditionalGeneration,
    ]

    gpu_image_decode = False

    # M3's tokenizer has no pad_token.
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
        # Newer M3 video processors expose max_pixels (area) instead of max_size; derive an equivalent (max_w, max_h) cap.
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

    def _rescale_images(self, images: List, image_items: List) -> List:
        """Apply the provider spec's long-side cap / short-side floor / ceiling."""
        rescaled = []
        for index, image in enumerate(images):
            item = image_items[index] if index < len(image_items) else None
            if not isinstance(image, Image.Image):
                # Preprocessed tensors and GPU-decoded images carry their own size.
                rescaled.append(image)
                continue
            cap = _max_long_side_pixel(item)
            width, height = resolve_media_size(image.width, image.height, cap)
            if cap is not None:
                # Only the caller-requested size is ours to bound. Without a cap
                # the HF processor's own area cap decides the final size, so the
                # decoded size is not the number to check.
                check_total_pixels(width, height, 1, MAX_TOTAL_PIXELS_IMAGE)
            if (width, height) != (image.width, image.height):
                image = image.resize((width, height), Image.BICUBIC)
            rescaled.append(image)
        return rescaled

    async def process_mm_data_async(
        self,
        image_data: Optional[List],
        audio_data: Optional[List],
        input_text: str,
        request_obj,
        **kwargs,
    ) -> Dict:
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        extra_kwargs = {}
        if base_output.images:
            base_output.images = self._rescale_images(
                base_output.images, image_data or []
            )
            image_caps = [_max_long_side_pixel(item) for item in image_data or []]
            if image_caps and all(cap is not None for cap in image_caps):
                # The long-side cap only binds if the processor's own area cap is
                # loose enough to let it through.
                extra_kwargs["images_kwargs"] = {"max_pixels": max(image_caps) ** 2}

        video_metadata = None
        if base_output.videos:
            image_factor, max_size = self._video_resize_config()
            video_items = list(request_obj.video_data or [])
            video_caps = [
                (
                    _max_long_side_pixel(video_items[index])
                    if index < len(video_items)
                    else None
                )
                for index in range(len(base_output.videos))
            ]
            videos_processed = [
                await get_video_tensor(
                    video,
                    image_factor=image_factor,
                    max_size=max_size,
                    fps=self.video_fps,
                    frame_max_size=cap or self.video_frame_max_size,
                    max_frames=self.video_max_frames,
                )
                for video, cap in zip(base_output.videos, video_caps)
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
            for video in base_output.videos:
                frames, _, height, width = video.shape
                check_total_pixels(width, height, frames, MAX_TOTAL_PIXELS_VIDEO)
            if video_caps and all(cap is not None for cap in video_caps):
                extra_kwargs["processor_video_config"] = {
                    **self.video_config,
                    "max_pixels": max(video_caps) ** 2,
                }

        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(
            base_output=base_output,
            mm_tokens=self.mm_tokens,
            video_metadata=video_metadata,
            **extra_kwargs,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
        )
