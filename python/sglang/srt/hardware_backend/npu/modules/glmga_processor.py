"""NPU patches for GLMGA image and video preprocessing.

The GLMGA processors used by GLM5-Next build 10-dimensional tensors inside
``_preprocess`` while reordering image and video patches. Ascend operators only
support tensors with at most 8 dimensions, so the final reshape fails when the
torchvision processor runs on NPU.

These wrappers preserve the Transformers preprocessing flow and output layout,
but replace only the high-dimensional patch reordering with the NPU-safe helper
shared with Qwen VL.
"""

import inspect

import torch
import torchvision.transforms.v2.functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from transformers.models.glmga.image_processing_glmga import smart_resize
from transformers.utils import TensorType
from transformers.video_utils import group_videos_by_shape, reorder_videos

from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
    transform_patches_to_flatten,
)
from sglang.srt.utils import apply_module_patch


def _resize_glmga_inputs(
    processor,
    inputs: torch.Tensor,
    *,
    size: SizeDict,
    resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    factor: int,
    temporal_factor: int,
    num_frames: int,
) -> torch.Tensor:
    """Resize GLMGA inputs across the old and new Transformers APIs.

    Transformers 5.15 moved GLMGA's dynamic size calculation into the model
    processor's ``resize`` method and made ``factor`` and ``temporal_factor``
    required. Older releases expose only the backend ``resize`` method, so the
    caller must calculate the concrete height and width first.
    """
    resize_parameters = inspect.signature(processor.resize).parameters
    if "factor" in resize_parameters and "temporal_factor" in resize_parameters:
        return processor.resize(
            inputs,
            size=size,
            resample=resample,
            factor=factor,
            temporal_factor=temporal_factor,
        )

    height, width = inputs.shape[-2:]
    resized_height, resized_width = smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=temporal_factor,
        factor=factor,
        min_pixels=size.shortest_edge,
        max_pixels=size.longest_edge,
    )
    concrete_size = SizeDict(height=resized_height, width=resized_width)

    if inputs.ndim == 5:
        batch_size, frames, channel, height, width = inputs.shape
        inputs = inputs.reshape(batch_size * frames, channel, height, width)
        inputs = processor.resize(inputs, size=concrete_size, resample=resample)
        return inputs.reshape(
            batch_size, frames, channel, resized_height, resized_width
        )

    return processor.resize(inputs, size=concrete_size, resample=resample)


def _npu_safe_flatten_patches(
    patches: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    """Return the GLMGA flattened patch layout without creating a >8-D tensor."""
    temporal_remainder = patches.shape[1] % temporal_patch_size
    if temporal_remainder:
        pad = temporal_patch_size - temporal_remainder
        repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
        patches = torch.cat((patches, repeats), dim=1)

    batch_size, num_frames, channel, resized_height, resized_width = patches.shape
    grid_t = num_frames // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    flatten_patches = transform_patches_to_flatten(
        patches=patches,
        batch_size=batch_size,
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
        channel=channel,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    return flatten_patches, grid_t, grid_h, grid_w


# Func refers to transformers.models.glmga.image_processing_glmga.py
# GlmgaImageProcessor._preprocess
def npu_wrapper_glmga_image_preprocess(func):
    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = _resize_glmga_inputs(
                    self,
                    stacked_images,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    num_frames=temporal_patch_size,
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(
            resized_images_grouped, grouped_images_index
        )
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
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

            flatten_patches, grid_t, grid_h, grid_w = _npu_safe_flatten_patches(
                patches,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * patches.shape[0]

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

    return _preprocess


# Func refers to transformers.models.glmga.video_processing_glmga.py
# GlmgaVideoProcessor._preprocess
def npu_wrapper_glmga_video_preprocess(func):
    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = (
            PILImageResampling.BICUBIC
        ),
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_expand_factor: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = _resize_glmga_inputs(
                    self,
                    stacked_videos,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    num_frames=stacked_videos.shape[1],
                )
            resized_videos_grouped[shape] = stacked_videos

        resized_videos = reorder_videos(
            resized_videos_grouped, grouped_videos_index
        )
        grouped_videos, grouped_videos_index = group_videos_by_shape(
            resized_videos
        )
        processed_videos_grouped = {}
        processed_grids = {}

        for shape, stacked_videos in grouped_videos.items():
            # Preserve the upstream validation/shape access before normalization.
            get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )
            patches = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            flatten_patches, grid_t, grid_h, grid_w = _npu_safe_flatten_patches(
                patches,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * patches.shape[0]

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

    return _preprocess


_npu_glmga_preprocess_patched = False


def npu_apply_glmga_preprocess_patch():
    global _npu_glmga_preprocess_patched
    if _npu_glmga_preprocess_patched:
        return

    apply_module_patch(
        "transformers.models.glmga.image_processing_glmga.GlmgaImageProcessor",
        "_preprocess",
        [npu_wrapper_glmga_image_preprocess],
    )
    apply_module_patch(
        "transformers.models.glmga.video_processing_glmga.GlmgaVideoProcessor",
        "_preprocess",
        [npu_wrapper_glmga_video_preprocess],
    )
    _npu_glmga_preprocess_patched = True
