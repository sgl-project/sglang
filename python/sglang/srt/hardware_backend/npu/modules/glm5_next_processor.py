"""NPU-safe patch extraction for Transformers GLM5-Next processors.

The generated Transformers GLM5-Next image and video processors materialize
9- and 10-dimensional tensors in ``patchify``. Ascend supports tensors with at
most 8 dimensions, so keep the upstream preprocessing flow and replace only
that layout transformation.
"""

from functools import wraps

import torch


def _npu_safe_flatten_patches(
    patches: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    """Flatten GLM5-Next patches without creating tensors above 8 dimensions."""
    temporal_remainder = patches.shape[1] % temporal_patch_size
    if temporal_remainder:
        pad = temporal_patch_size - temporal_remainder
        repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
        patches = torch.cat((patches, repeats), dim=1)

    batch_size, num_frames, channel, resized_height, resized_width = patches.shape
    grid_t = num_frames // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    # This is equivalent to Transformers' 10-D view/permute, split into two
    # transformations whose intermediate tensors have at most 8 dimensions.
    patches = patches.reshape(
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
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flatten_patches, grid_t, grid_h, grid_w


def npu_wrapper_glm5_next_image_patchify(func):
    @wraps(func)
    def patchify(
        self,
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        flatten_patches, grid_t, grid_h, grid_w = _npu_safe_flatten_patches(
            images.unsqueeze(1),
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        if grid_t != 1:
            raise ValueError(f"Expected one temporal image grid, got {grid_t}")
        return flatten_patches, grid_h, grid_w

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
        return _npu_safe_flatten_patches(
            videos,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
        )

    return patchify


_NPU_PATCH_MARKER = "_sglang_npu_safe_glm5_next_patchify"


def _patch_processor_patchify(processor, wrapper) -> None:
    """Patch the processor's actual class, including trust-remote-code classes."""
    if processor is None:
        return

    processor_type = type(processor)
    original = getattr(processor_type, "patchify", None)
    if original is None or getattr(original, _NPU_PATCH_MARKER, False):
        return

    patched = wrapper(original)
    setattr(patched, _NPU_PATCH_MARKER, True)
    setattr(processor_type, "patchify", patched)


def npu_apply_glm5_next_preprocess_patch(processor) -> None:
    """Patch the concrete GLM5-Next processors held by ``processor``.

    ``trust_remote_code`` loads classes below ``transformers_modules.*`` rather
    than ``transformers.models.glm5_next.*``. Patching the instances' concrete
    classes covers both layouts without depending on a generated module name.
    """
    _patch_processor_patchify(
        getattr(processor, "image_processor", None),
        npu_wrapper_glm5_next_image_patchify,
    )
    _patch_processor_patchify(
        getattr(processor, "video_processor", None),
        npu_wrapper_glm5_next_video_patchify,
    )
