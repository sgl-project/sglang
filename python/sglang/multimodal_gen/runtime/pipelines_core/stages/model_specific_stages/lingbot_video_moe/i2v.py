# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200

# Qwen3-VL vision patch_size 16 times spatial_merge_size 2.
VLM_PATCH_FACTOR = 32

COND_LATENT_KEY = "lingbot_cond_latent"
VLM_IMAGE_KEY = "lingbot_vlm_image"
TEXT_ONLY_EMBEDS_KEY = "lingbot_text_only_embeds"


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * factor**2
    if min_pixels is None:
        min_pixels = IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def preprocess_condition_image(
    image: PIL.Image.Image, height: int, width: int
) -> torch.Tensor:
    """Cover-resize then center-crop the condition frame to a 1, 3, 1, H, W tensor in 0..1."""

    raw = (
        torch.from_numpy(np.array(image.convert("RGB")))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    old_h, old_w = raw.shape[-2:]
    scale = max(height / old_h, width / old_w)
    new_h = max(math.ceil(old_h * scale), height)
    new_w = max(math.ceil(old_w * scale), width)
    # Resize in uint8, matching the reference implementation.
    resized = F.interpolate(
        raw, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    top = int(round((new_h - height) / 2.0))
    left = int(round((new_w - width) / 2.0))
    cropped = resized[:, :, top : top + height, left : left + width] / 255.0
    return cropped.unsqueeze(2)


def pixel_to_vlm_image(
    pixel: torch.Tensor, factor: int = VLM_PATCH_FACTOR
) -> PIL.Image.Image:
    """Patch-aligned PIL view of the condition frame for Qwen3-VL."""

    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = PIL.Image.fromarray(array, mode="RGB")
    resized_height, resized_width = smart_resize(
        image.height, image.width, factor=factor
    )
    return image.resize((resized_width, resized_height))


def apply_first_frame_prefix(
    latents: torch.Tensor, cond_latent: torch.Tensor
) -> torch.Tensor:
    """Replace the leading latent frames with the clean condition latent.

    Returns a new tensor; the denoising latents are inference tensors, which
    cannot be updated in place.
    """

    cond_frames = cond_latent.shape[2]
    return torch.cat(
        [cond_latent.to(latents.dtype), latents[:, :, cond_frames:]], dim=2
    )
