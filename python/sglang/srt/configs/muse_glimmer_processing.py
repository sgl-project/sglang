# Copyright 2023-2026 SGLang Team
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
# ==============================================================================

import itertools
import json
import math
import os
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessorMixin,
    VideosKwargs,
)
from transformers.utils import TensorType, logging
from transformers.utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.utils.hub import cached_file
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import (
    VideoMetadata,
    group_videos_by_shape,
    reorder_videos,
)

logger = logging.get_logger(__name__)

PROCESSOR_CONFIG_NAME = "processor_config.json"


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_tokens: int,
) -> tuple[int, int]:
    """Patch grid closest to the aspect ratio; returns (height, width) in pixels."""
    ideal_patches_height = height / patch_size
    ideal_patches_width = width / patch_size
    ratio = (
        ideal_patches_width / ideal_patches_height if ideal_patches_height > 0 else 1.0
    )
    if ideal_patches_height * ideal_patches_width > max_tokens:
        ideal_patches_height = (max_tokens / ratio) ** 0.5
        ideal_patches_width = ideal_patches_height * ratio
    candidates = list(
        set(
            itertools.product(
                [math.floor(ideal_patches_height), math.ceil(ideal_patches_height)],
                [math.floor(ideal_patches_width), math.ceil(ideal_patches_width)],
            )
        )
    )
    candidates = [
        (patches_height, patches_width)
        for patches_height, patches_width in candidates
        if patches_height >= 1
        and patches_width >= 1
        and patches_height * patches_width <= max_tokens
    ]
    if not candidates:
        candidates = [
            (max(1, round(ideal_patches_height)), max(1, round(ideal_patches_width)))
        ]
    patches_height, patches_width = min(
        candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width)
    )
    return patches_height * patch_size, patches_width * patch_size


class MuseGlimmerImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_image_tokens: int


class MuseGlimmerImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.LANCZOS
    size = None
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_image_tokens = 4096
    valid_kwargs = MuseGlimmerImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        return_tensors: Optional[TensorType],
        patch_size: int,
        temporal_patch_size: int,
        max_image_tokens: int,
        merge_size: int,
        disable_grouping: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if resample == PILImageResampling.LANCZOS:
            # BICUBIC stands in for LANCZOS, which is CPU-only.
            resample = PILImageResampling.BICUBIC

        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = get_aspect_ratio_preserving_size(
                    height=height,
                    width=width,
                    patch_size=patch_size * merge_size,
                    max_tokens=max_image_tokens,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                    antialias=True,
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
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                temporal_patch_size * channel * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Patch rows a (height, width) image expands to."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        resized_height, resized_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_image_tokens,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)

    def _validate_preprocess_kwargs(self, **kwargs):
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)


class MuseGlimmerVideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    max_video_frame_tokens: int
    merge_size: int
    frame_timestamp_format: str


class MuseGlimmerVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_video_frame_tokens = 144
    num_frames = 96
    fps = 2.0
    do_sample_frames = True
    return_metadata = True
    frame_timestamp_format = "Time: {timestamp:.1f}s"
    valid_kwargs = MuseGlimmerVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def _validate_preprocess_kwargs(self, **kwargs):
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def resize(
        self,
        videos: torch.Tensor,
        resample,
        patch_size: int,
        merge_size: int,
        max_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        if resample == PILImageResampling.LANCZOS:
            # BICUBIC stands in for LANCZOS, which is CPU-only.
            resample = PILImageResampling.BICUBIC

        height, width = videos.shape[-2:]
        resized_height, resized_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size * merge_size,
            max_tokens=max_tokens,
        )
        return super().resize(
            videos,
            size=SizeDict(height=resized_height, width=resized_width),
            resample=resample,
            antialias=True,
        )

    def patchify(
        self,
        videos: torch.Tensor,
        patch_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape
        if pad := -num_frames % temporal_patch_size:
            videos = torch.cat([videos, videos[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5, 7)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            temporal_patch_size * channel * patch_size * patch_size,
        )
        return flatten_patches, grid_t, grid_h, grid_w

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        temporal_patch_size = (
            temporal_patch_size
            if temporal_patch_size is not None
            else self.temporal_patch_size
        )
        fps = fps if fps is not None else self.fps
        num_frames = num_frames if num_frames is not None else self.num_frames

        if metadata.fps is None or not math.isfinite(metadata.fps) or metadata.fps <= 0:
            logger.warning_once(
                "The `fps` of the input video could not be inferred. Defaulting to "
                "`fps=24`. Provide `video_metadata` for more accurate frame sampling."
            )
            metadata.fps = 24

        total_num_frames = metadata.total_num_frames
        if total_num_frames <= 0:
            raise ValueError("Video must contain at least one frame.")

        sampled_frames = min(
            int(total_num_frames * fps / metadata.fps), num_frames, total_num_frames
        )
        sampled_frames = max(
            temporal_patch_size,
            (sampled_frames // temporal_patch_size) * temporal_patch_size,
        )
        sampled_frames = min(sampled_frames, total_num_frames)
        return torch.linspace(0, total_num_frames - 1, sampled_frames).long()

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        return_tensors: Optional[TensorType],
        patch_size: int,
        temporal_patch_size: int,
        max_video_frame_tokens: int,
        merge_size: int,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_resize:
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    resample=resample,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    max_tokens=max_video_frame_tokens,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            flatten_patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * stacked_videos.shape[
                0
            ]

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )


class MuseGlimmerProcessor(ProcessorMixin):
    """Expands ``<|patch|>`` and ``<|video|>`` placeholders for Muse Glimmer."""

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if video_processor is None:
            video_processor = MuseGlimmerVideoProcessor()

        self.image_token = "<|patch|>"
        self.image_start_token = "<|image_start|>"
        self.image_end_token = "<|image_end|>"
        self.video_token = "<|video|>"
        self.video_sep_token = "<|vid_frame_separator|>"
        self.video_start_token = "<|vid_start|>"
        self.video_end_token = "<|vid_end|>"

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(
            self.image_start_token
        )
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

        super().__init__(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        revision = kwargs.pop("revision", None)
        use_fast = kwargs.pop("use_fast", True)

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            use_fast=use_fast,
        )
        image_kwargs, video_kwargs = _load_processor_component_kwargs(
            pretrained_model_name_or_path, revision
        )
        image_processor = MuseGlimmerImageProcessor(**image_kwargs)
        video_processor = MuseGlimmerVideoProcessor(**video_kwargs)
        return cls(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = int(
            image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        )
        return (
            self.image_start_token
            + self.image_token * num_image_tokens
            + self.image_end_token
        )

    def _process_videos(self, videos, **kwargs):
        # Timestamps in replace_video_token always require video_metadata.
        return super()._process_videos(videos, **{**kwargs, "return_metadata": True})

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        merge_length = self.video_processor.merge_size**2
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        n_frames = int(grid_thw[0])
        tokens_per_group = int(grid_thw[1:].prod() // merge_length)

        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None or not math.isfinite(metadata.fps) or metadata.fps <= 0:
            logger.warning_once(
                "MuseGlimmer requires frame timestamps to construct prompts, but the "
                "`fps` of the input video could not be inferred. Defaulting to "
                "`fps=24`. Please provide `video_metadata` for more accurate results."
            )
            metadata.fps = 24
        if metadata.frames_indices is None:
            metadata.frames_indices = list(range(metadata.total_num_frames))

        temporal_patch_size = self.video_processor.temporal_patch_size
        timestamps = list(metadata.timestamps[::temporal_patch_size])[:n_frames]
        while len(timestamps) < n_frames:
            timestamps.append(timestamps[-1] if timestamps else 0.0)

        timestamp_format = self.video_processor.frame_timestamp_format
        parts = [self.video_start_token]
        for g, ts in enumerate(timestamps):
            parts.append(timestamp_format.format(timestamp=ts))
            parts.append(self.video_token * tokens_per_group)
            parts.append(
                self.video_sep_token if g < n_frames - 1 else self.video_end_token
            )
        return "".join(parts)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Placeholder counts per image, without running the image processor."""
        vision_data = {}
        if image_sizes is not None:
            merge_size = self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, kwargs)
                for height, width in image_sizes
            ]
            vision_data.update(
                num_image_tokens=[
                    patches // merge_size**2 for patches in num_image_patches
                ],
                num_image_patches=num_image_patches,
            )
        return MultiModalData(**vision_data)


def _load_processor_component_kwargs(
    model_path: str, revision: Optional[str]
) -> tuple[dict, dict]:
    """Read image_processor and video_processor blocks from processor_config.json."""
    local = os.path.join(model_path, PROCESSOR_CONFIG_NAME)
    config_file = (
        local
        if os.path.isfile(local)
        else cached_file(model_path, PROCESSOR_CONFIG_NAME, revision=revision)
    )
    with open(config_file) as f:
        config = json.load(f)

    image_processor_config = config.get("image_processor", {})
    image_kwargs = {
        key: value
        for key, value in image_processor_config.items()
        if key != "image_processor_type"
    }
    video_processor_config = config.get("video_processor", {})
    video_kwargs = {
        key: value
        for key, value in video_processor_config.items()
        if key != "video_processor_type"
    }
    return image_kwargs, video_kwargs
