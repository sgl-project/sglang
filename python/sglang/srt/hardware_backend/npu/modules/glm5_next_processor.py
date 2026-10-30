"""NPU-safe patch extraction for Transformers GLM5-Next processors.

The generated Transformers GLM5-Next image and video processors materialize
9- and 10-dimensional tensors in ``patchify``. Ascend's ``aclnnPermute`` supports
at most 8 dimensions, so keep the upstream preprocessing flow and replace only
that layout transformation.
"""

from functools import wraps

import torch
from sgl_kernel_npu.operators.mm import flatten_patches

from sglang.srt.hardware_backend.npu.modules.glm_vl_utils import (
    _patch_processor_method,
)


def npu_wrapper_glm5_next_image_patchify(func):
    @wraps(func)
    def patchify(
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

    return patchify


def npu_wrapper_glm5_next_video_patchify(func):
    @wraps(func)
    def patchify(
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

    return patchify


_NPU_PATCH_MARKER = "_sglang_npu_safe_glm5_next_patchify"


def npu_apply_glm5_next_preprocess_patch(processor) -> bool:
    """Patch the concrete GLM5-Next processors held by ``processor``.

    ``trust_remote_code`` loads classes below ``transformers_modules.*`` rather
    than ``transformers.models.glm5_next.*``. Patching the instances' concrete
    classes covers both layouts without depending on a generated module name.
    Return True only when every present inner processor exposes ``patchify``.
    """
    patches = []
    for kind, wrapper in (
        ("image", npu_wrapper_glm5_next_image_patchify),
        ("video", npu_wrapper_glm5_next_video_patchify),
    ):
        inner = getattr(processor, f"{kind}_processor", None)
        if inner is None:
            continue
        inner_type = type(inner)
        if not callable(getattr(inner_type, "patchify", None)):
            return False
        patches.append((inner, "patchify", wrapper, _NPU_PATCH_MARKER))

    return bool(patches) and all(_patch_processor_method(*args) for args in patches)
