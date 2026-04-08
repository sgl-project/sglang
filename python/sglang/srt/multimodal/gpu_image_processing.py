"""GPU-accelerated image preprocessing utilities.

Provides torch GPU replacements for CPU-based PIL/numpy image processing
(resize, pad, normalize, patchify). Model-agnostic — reusable for
Kimi-K2.5, Qwen, InternVL, etc.
"""

import math
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def navit_resize_config(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None = None,
) -> dict:
    """Compute NaViT resize target dimensions and token count.

    Pure math — no image data needed, only (width, height).
    Reimplemented from HF media_utils.navit_resize_image.
    """
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), patch_limit_on_one_side * patch_size)
    new_h = min(max(1, int(height * scale)), patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> tuple[int, int]:
    """Get (width, height) from a CUDA tensor or PIL Image."""
    if isinstance(image, torch.Tensor):
        # nvJPEG returns (C, H, W) uint8
        return image.shape[2], image.shape[1]
    return image.size  # PIL returns (width, height)


def pil_to_cuda_chw(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to (C, H, W) uint8 CUDA tensor."""
    arr = np.asarray(image.convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process a single image on GPU: resize → pad → normalize → patchify.

    Returns:
        pixel_values: (num_patches, C, patch_size, patch_size) float32 CUDA
        grid_thw: (3,) int64 tensor [T, H//ps, W//ps]
    """
    if isinstance(image, Image.Image):
        image = pil_to_cuda_chw(image)

    # image is (C, H, W) uint8 on CUDA
    new_h, new_w = config["new_height"], config["new_width"]
    pad_h, pad_w = config["pad_height"], config["pad_width"]

    # Resize: (C, H, W) → (1, C, new_h, new_w)
    x = image.unsqueeze(0).float()
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)

    # Pad right and bottom
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

    # Normalize: (x / 255.0 - mean) * std_inv
    x = x / 255.0
    x = (x - image_mean) * image_std_inv

    # Patchify: (1, C, H_padded, W_padded) → (num_patches, C, ps, ps)
    _, C, H, W = x.shape
    T = 1
    gh, gw = H // patch_size, W // patch_size
    # reshape to (T, C, gh, ps, gw, ps) then permute to (T*gh*gw, C, ps, ps)
    x = x.view(T, C, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_size, patch_size)

    grid_thw = torch.tensor([T, gh, gw], dtype=torch.int64, device=x.device)
    return x, grid_thw


def gpu_preprocess_images(
    images: list[Union[torch.Tensor, Image.Image]],
    resize_configs: list[dict],
    image_mean: torch.Tensor,
    image_std_inv: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.

    Processes images with batch optimization: images with the same target
    padded size are grouped and processed together with a single
    F.interpolate call.

    Args:
        images: List of images (CUDA tensors from nvJPEG or PIL fallbacks).
        resize_configs: List of dicts from navit_resize_config().
        image_mean: (1, 3, 1, 1) float32 CUDA tensor.
        image_std_inv: (1, 3, 1, 1) float32 CUDA tensor.
        patch_size: Patch size for patchification.

    Returns:
        pixel_values: (total_patches, C, patch_size, patch_size) float32 CUDA
        grid_thws: (N, 3) int64 CUDA tensor
    """
    n = len(images)
    if n == 0:
        device = image_mean.device
        return (
            torch.empty(0, 3, patch_size, patch_size, device=device),
            torch.empty(0, 3, dtype=torch.int64, device=device),
        )

    # Group images by target padded dimensions for batch processing
    groups = defaultdict(list)
    for idx, (image, config) in enumerate(zip(images, resize_configs)):
        padded_h = config["new_height"] + config["pad_height"]
        padded_w = config["new_width"] + config["pad_width"]
        target_h = config["new_height"]
        target_w = config["new_width"]
        groups[(target_h, target_w, padded_h, padded_w)].append((idx, image, config))

    # Process each group
    all_patches = [None] * n
    all_grids = [None] * n

    for (target_h, target_w, padded_h, padded_w), group in groups.items():
        if len(group) == 1:
            # Single image — no batching overhead
            idx, image, config = group[0]
            patches, grid = _process_single_image(
                image, config, image_mean, image_std_inv, patch_size
            )
            all_patches[idx] = patches
            all_grids[idx] = grid
        else:
            # Batch: convert all to CUDA tensors, resize together
            tensors = []
            for _, image, _ in group:
                if isinstance(image, Image.Image):
                    image = pil_to_cuda_chw(image)
                tensors.append(image.unsqueeze(0).float())

            # All images in this group resize to the same target
            batch = torch.cat(tensors, dim=0)  # (B, C, H_various, W_various)

            # Resize to common target — need same input size for batch F.interpolate
            # Since input sizes may differ, resize individually then stack
            resized = []
            for t in tensors:
                r = F.interpolate(
                    t, size=(target_h, target_w), mode="bicubic", align_corners=False
                )
                resized.append(r)
            batch = torch.cat(resized, dim=0)  # (B, C, target_h, target_w)

            # Pad
            pad_h = padded_h - target_h
            pad_w = padded_w - target_w
            if pad_h > 0 or pad_w > 0:
                batch = F.pad(batch, (0, pad_w, 0, pad_h), value=0.0)

            # Normalize
            batch = batch / 255.0
            batch = (batch - image_mean) * image_std_inv

            # Patchify: (B, C, H, W) → per-image (num_patches, C, ps, ps)
            B, C, H, W = batch.shape
            T = 1
            gh, gw = H // patch_size, W // patch_size
            batch = batch.view(B, C, gh, patch_size, gw, patch_size)
            batch = batch.permute(0, 2, 4, 1, 3, 5).reshape(
                B, -1, C, patch_size, patch_size
            )

            grid = torch.tensor([T, gh, gw], dtype=torch.int64, device=batch.device)
            for i, (idx, _, _) in enumerate(group):
                all_patches[idx] = batch[i]  # (num_patches_per_image, C, ps, ps)
                all_grids[idx] = grid

    pixel_values = torch.cat(all_patches, dim=0)
    grid_thws = torch.stack(all_grids, dim=0)
    return pixel_values, grid_thws
