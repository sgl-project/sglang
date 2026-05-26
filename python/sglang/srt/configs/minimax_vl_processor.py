# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
MiniMax VL family HuggingFace-compatible Processor, ImageProcessor, VideoProcessor.
"""

import math
import re
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos


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
        new_w = round(w * ratio)
        new_h = round(h * ratio)
        # ceil-align to multiple
        new_w = (
            new_w if new_w % multiple == 0 else new_w + (multiple - new_w % multiple)
        )
        new_h = (
            new_h if new_h % multiple == 0 else new_h + (multiple - new_h % multiple)
        )
        return new_w, new_h

    # Round up to nearest multiple
    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)

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


def vllm_resize(
    height: int,
    width: int,
    factor: int,
    max_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Wrapper around get_hw_multiple_of.

    Args:
        height: Image height
        width: Image width
        factor: Alignment factor (patch_size * merge_size)
        max_size: (max_width, max_height) constraint

    Returns:
        Tuple[int, int]: (new_height, new_width)
    """
    new_w, new_h = get_hw_multiple_of((width, height), factor, max_size)
    return new_h, new_w


# ==============================================================================
# MiniMax M3 VL Image Processor Fast (Fast Mode - Torch based)
# Copied from Qwen2VLImageProcessorFast with resize changed to vLLM style
# ==============================================================================


class MiniMaxM3VLImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    max_size (`int` or `tuple[int, int]`, *optional*):
        The vLLM-style resize bound. If unset, it is inferred from `size`.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_size: Union[int, Tuple[int, int]]


class MiniMaxM3VLImageProcessor(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_size = None
    valid_kwargs = MiniMaxM3VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]):
        super().__init__(**kwargs)

    @staticmethod
    def _max_size_from_size(size):
        if size is None:
            return None
        if isinstance(size, int):
            return (size, size)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (size[0], size[1])
        if isinstance(size, dict):
            width = size.get("width") or size.get("max_width")
            height = size.get("height") or size.get("max_height")
            if width is not None and height is not None:
                return (width, height)
        if isinstance(size, SizeDict):
            width = size.width or size.max_width
            height = size.height or size.max_height
            if width is not None and height is not None:
                return (width, height)
        return None

    def _further_process_kwargs(self, size=None, max_size=None, **kwargs):
        kwargs = super()._further_process_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if max_size is None:
            max_size = self._max_size_from_size(size)
        if max_size is None:
            raise ValueError("max_size must be provided or inferable from size.")
        kwargs["max_size"] = max_size
        return kwargs

    def preprocess(
        self, images, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: List[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | List[float] | None,
        image_std: float | List[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_size: int | Tuple[int, int],
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = vllm_resize(
                    height, width, patch_size * merge_size, max_size
                )
                stacked_images = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1,
                    temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                    1,
                    1,
                    1,
                )
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_size = images_kwargs.get("max_size", self.max_size)
        if max_size is None:
            max_size = self._max_size_from_size(images_kwargs.get("size", self.size))

        resized_height, resized_width = vllm_resize(
            height, width, patch_size * merge_size, max_size
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


# ==============================================================================
# MiniMax M3 VL Video Processor (Torch based)
# Copied from Qwen2VLVideoProcessor with resize changed to vLLM style
# ==============================================================================


class MiniMaxM3VLVideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_size: Union[int, Tuple[int, int]]
    frame_max_size: int


class MiniMaxM3VLVideoProcessor(BaseVideoProcessor):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    do_sample_frames = False
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_size = None
    frame_max_size = None
    valid_kwargs = MiniMaxM3VLVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLVideoProcessorKwargs]):
        super().__init__(**kwargs)

    @staticmethod
    def _max_size_from_size(size):
        if size is None:
            return None
        if isinstance(size, int):
            return (size, size)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (size[0], size[1])
        if isinstance(size, dict):
            width = size.get("width") or size.get("max_width")
            height = size.get("height") or size.get("max_height")
            if width is not None and height is not None:
                return (width, height)
        if isinstance(size, SizeDict):
            width = size.width or size.max_width
            height = size.height or size.max_height
            if width is not None and height is not None:
                return (width, height)
        return None

    def _further_process_kwargs(self, size=None, max_size=None, **kwargs):
        kwargs = super()._further_process_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if max_size is None:
            max_size = self._max_size_from_size(size)
        if max_size is None:
            raise ValueError("max_size must be provided or inferable from size.")
        kwargs["max_size"] = max_size
        return kwargs

    def _preprocess(
        self,
        videos: List[torch.Tensor],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | List[float] | None,
        image_std: float | List[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_size: int | Tuple[int, int],
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = vllm_resize(
                    height, width, patch_size * merge_size, max_size
                )
                stacked_videos = stacked_videos.view(
                    batch_size * num_frames, channels, height, width
                )
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(
                    batch_size,
                    num_frames,
                    channels,
                    resized_height,
                    resized_width,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = stacked_videos.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )

            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channels = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )


# ==============================================================================
# Video preprocessing: sample frames by target fps and resize
# ==============================================================================


def _compute_sampled_frame_indices(
    total_frames: int, video_fps: float, fps: float
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
    return indices


async def get_video_tensor(
    vr,
    image_factor: int,
    max_size: Tuple[int, int],
    fps: Optional[float] = None,
    frame_max_size: Optional[int] = None,
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
    indices = _compute_sampled_frame_indices(total_frames, video_fps, fps)
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


class MiniMaxVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "videos_kwargs": {
            "do_resize": False,
            "return_metadata": True,
        },
    }


class MiniMaxVLProcessor(ProcessorMixin):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def __init__(
        self, image_processor=None, tokenizer=None, video_processor=None, **kwargs
    ):
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN)
        super().__init__(image_processor, tokenizer, video_processor)
        # Video expansion also uses image start/end tokens. Separate video
        # start/end tokens exist in the tokenizer, but the original MiniMax
        # serving path did not use them; keep that behavior for compatibility.
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_START_TOKEN
        )
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.VISION_END_TOKEN
        )

    @classmethod
    def _get_arguments_from_pretrained(
        cls, pretrained_model_name_or_path, processor_dict=None, **kwargs
    ):
        # Bypass Auto classes which would load fake processors from ckpt's auto_map
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        image_processor = MiniMaxM3VLImageProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        video_processor = MiniMaxM3VLVideoProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return [image_processor, tokenizer, video_processor]

    def _prune_video_tokens(
        self,
        input_text: str,
        video_segments: List[int],
        video_token: str,
    ) -> str:
        """
        Prune video tokens by temporal_patch_size (e.g., 2:1).

        Expects the prompt to carry exactly sum(video_segments) video
        tokens — i.e. one token per *sampled* frame. Then drops token.

        Args:
            input_text: prompt with N video_tokens per segment
            video_segments: actual sampled frame count per video segment
            video_token: the video token string, e.g. ']<]video[>['

        Returns:
            Pruned input_text with ~N/temporal_patch_size tokens per segment.
        """
        # If no videos or temporal_patch_size <= 1, no pruning needed
        if not video_segments or self.video_processor.temporal_patch_size <= 1:
            return input_text

        # Split while keeping delimiters
        special_tokens = [video_token]  # , image_token]
        pattern = "|".join(map(re.escape, special_tokens))
        parts = re.split(f"({pattern})", input_text)

        def is_timestamp(text: str) -> bool:
            """Check if text ends with timestamp format like ']<]0.0 seconds[>['"""
            return (
                text.endswith("seconds[>[")
                or text.endswith("seconds[>[ ")
                or text.endswith("seconds [>[")
                or text.endswith("seconds [>[ ")
            )

        def extract_timestamp(text: str) -> str:
            """Extract timestamp text from the end, starting from ']<]'"""
            start_index = text.rfind("]<]")
            if start_index == -1:
                raise ValueError(f"Failed to extract timestamp: {text}")
            return text[start_index:]

        # Build new text with pruned video tokens
        final_parts = []
        current_seg_idx = 0  # Which video segment we're in
        frame_in_seg = 0  # Frame index within current segment
        last_timestamp_len = 0  # Length of timestamp to potentially remove

        for part in parts:
            if part == video_token:
                if current_seg_idx < len(video_segments):
                    if frame_in_seg % self.video_processor.temporal_patch_size == 0:
                        # Keep this video token
                        final_parts.append(part)
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        last_timestamp_len = 0
                    else:
                        # Skip this video token
                        frame_in_seg += 1
                        if frame_in_seg >= video_segments[current_seg_idx]:
                            current_seg_idx += 1
                            frame_in_seg = 0
                        # Remove the timestamp that was already appended
                        if last_timestamp_len > 0:
                            # Truncate the last part to remove timestamp
                            assert len(final_parts) > 0
                            final_parts[-1] = final_parts[-1][:-last_timestamp_len]
                            last_timestamp_len = 0
                else:
                    # No more video segments, keep as is
                    final_parts.append(part)
                    last_timestamp_len = 0
            else:
                # Text part
                final_parts.append(part)
                # Check if this text ends with a timestamp
                if is_timestamp(part):
                    last_timestamp_len = len(extract_timestamp(part))
                else:
                    last_timestamp_len = 0

        return "".join(final_parts)

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs: Unpack[MiniMaxVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            MiniMaxVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            images_kwargs = output_kwargs["images_kwargs"]
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]

        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_kwargs = output_kwargs["videos_kwargs"]
            video_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = video_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = video_inputs.pop("video_metadata")
            else:
                video_metadata = video_inputs["video_metadata"]
        else:
            video_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        # Expand image tokens
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.IMAGE_TOKEN in text[i]:
                    num_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.IMAGE_TOKEN,
                        self.VISION_START_TOKEN
                        + placeholder * num_tokens
                        + self.VISION_END_TOKEN,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace(placeholder, self.IMAGE_TOKEN)

        # Expand video tokens
        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            placeholder = "]<]placeholder[>["
            index = 0
            for i in range(len(text)):
                while self.VIDEO_TOKEN in text[i]:
                    metadata = video_metadata[index]
                    grid_t = video_grid_thw[index][0]
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length

                    video_placeholder = ""
                    for frame_idx in range(grid_t):
                        if (
                            metadata.fps is not None
                            and metadata.frames_indices is not None
                        ):
                            ts = (
                                metadata.frames_indices[
                                    min(
                                        frame_idx
                                        * self.video_processor.temporal_patch_size,
                                        len(metadata.frames_indices) - 1,
                                    )
                                ]
                                / metadata.fps
                            )
                            video_placeholder += f"]<]{ts:.1f} seconds[>["
                        video_placeholder += (
                            self.VISION_START_TOKEN
                            + placeholder * frame_seqlen
                            + self.VISION_END_TOKEN
                        )

                    text[i] = text[i].replace(self.VIDEO_TOKEN, video_placeholder, 1)
                    index += 1
                text[i] = text[i].replace(placeholder, self.VIDEO_TOKEN)

        # Tokenize
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )
