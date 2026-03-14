from typing import Optional

import torch
import torchvision.transforms.v2.functional as tvF
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import (
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import SizeDict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import TensorType

from sglang.srt.utils import apply_module_patch


# Func refers to transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.py
# Qwen2VLImageProcessorFast._preprocess
def npu_wrapper_preprocess(func):

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["tvF.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            ######################################
            # Start of modifications for sglang  #
            ######################################
            patches = patches.view(
                batch_size * grid_t,
                temporal_patch_size * channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)
            patches = patches.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h * grid_w,
                patch_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 3, 2, 5, 6)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                -1,
            )
            ######################################
            #  End of modifications for sglang   #
            ######################################

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

    return _preprocess


_npu_preprocess_patched = False


def npu_apply_qwen_image_preprocess_patch():
    global _npu_preprocess_patched
    if _npu_preprocess_patched:
        return
    apply_module_patch(
        "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast",
        "_preprocess",
        [npu_wrapper_preprocess],
    )
    _npu_preprocess_patched = True
