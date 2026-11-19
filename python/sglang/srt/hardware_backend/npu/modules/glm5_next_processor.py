"""NPU-safe patch extraction for Transformers GLM5-Next processors.

The generated Transformers GLM5-Next image and video processors materialize
9- and 10-dimensional tensors in ``patchify``. Ascend's ``aclnnPermute`` supports
at most 8 dimensions, so keep the upstream preprocessing flow and replace only
that layout transformation.
"""

import torch
from sgl_kernel_npu.operators.multimodal import flatten_patches


def _npu_image_patchify(
    self,
    images: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int]:
    flattened_patches, grid_t, grid_h, grid_w = flatten_patches(
        images.unsqueeze(1),
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    if grid_t != 1:
        raise ValueError(f"Expected one temporal image grid, got {grid_t}")
    return flattened_patches, grid_h, grid_w


def _npu_video_patchify(
    self,
    videos: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    return flatten_patches(
        videos,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )


def npu_apply_glm5_next_preprocess_patch(processor) -> None:
    """Patch the concrete GLM5-Next processors held by ``processor``.

    ``trust_remote_code`` loads classes below ``transformers_modules.*`` rather
    than ``transformers.models.glm5_next.*``. Patching the instances' concrete
    classes covers both layouts without depending on a generated module name.
    """
    type(processor.image_processor).patchify = _npu_image_patchify
    type(processor.video_processor).patchify = _npu_video_patchify
