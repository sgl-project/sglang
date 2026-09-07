"""NPU patch for MiniMax M3 VL image and video preprocessing.

The MiniMax M3 VL image processor (MiniMaxM3VLImageProcessor) and video
processor (MiniMaxM3VLVideoProcessor) create 10-dimensional tensors during
patch extraction, which exceeds Ascend NPU's 8-dimension limit.

This patch restructures the computation using transform_patches_to_flatten
to stay within 8 dimensions, following the same pattern as the Qwen VL and
GLM-4.6V NPU patches.
"""

import math
from typing import List

import torch
from torchvision.transforms import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.utils import TensorType
from transformers.video_utils import group_videos_by_shape, reorder_videos

from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
    transform_patches_to_flatten,
)

MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def npu_wrapper_minimax_m3_image_preprocess(func):

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
        max_pixels: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = _smart_resize(
                    height,
                    width,
                    factor=factor,
                    max_pixels=max_pixels,
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

            flatten_patches = transform_patches_to_flatten(
                patches,
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h,
                grid_w,
                patch_size,
                merge_size,
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

    return _preprocess


def npu_wrapper_minimax_m3_video_preprocess(func):

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
        min_pixels: int,
        max_pixels: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = _smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
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

            flatten_patches = transform_patches_to_flatten(
                patches,
                batch_size,
                grid_t,
                temporal_patch_size,
                channels,
                grid_h,
                grid_w,
                patch_size,
                merge_size,
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

    return _preprocess


def npu_apply_minimax_m3_image_preprocess_patch(image_processor):
    cls = type(image_processor)
    if getattr(cls, "_sglang_npu_patched", False):
        return
    cls._preprocess = npu_wrapper_minimax_m3_image_preprocess(cls._preprocess)
    cls._sglang_npu_patched = True


def npu_apply_minimax_m3_video_preprocess_patch(video_processor):
    cls = type(video_processor)
    if getattr(cls, "_sglang_npu_video_patched", False):
        return
    cls._preprocess = npu_wrapper_minimax_m3_video_preprocess(cls._preprocess)
    cls._sglang_npu_video_patched = True
