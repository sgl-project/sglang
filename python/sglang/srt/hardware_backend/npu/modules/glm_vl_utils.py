"""Tensor layout and concrete-class patch helpers shared by GLM processors."""

import torch


def _npu_safe_flatten_patches(
    patches: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    """Flatten GLM patches without creating tensors above 8 dimensions."""
    temporal_remainder = patches.shape[1] % temporal_patch_size
    if temporal_remainder:
        pad = temporal_patch_size - temporal_remainder
        repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
        patches = torch.cat((patches, repeats), dim=1)

    batch_size, num_frames, channel, resized_height, resized_width = patches.shape
    grid_t = num_frames // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    # Split the upstream 10-D view/permute into two transformations with at
    # most 8 dimensions, preserving temporal, channel and spatial ordering.
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


def _patch_processor_method(processor, method: str, wrapper, marker: str) -> bool:
    """Patch the actual class, including remote classes; report supported APIs."""
    processor_type = type(processor)
    original = getattr(processor_type, method, None)
    if not callable(original):
        return False
    if getattr(original, marker, False):
        return True

    patched = wrapper(original)
    setattr(patched, marker, True)
    setattr(processor_type, method, patched)
    return True
