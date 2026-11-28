# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

# Qwen3-VL vision patch_size 16 times spatial_merge_size 2.
VLM_PATCH_FACTOR = 32

COND_LATENT_KEY = "lingbot_cond_latent"
VLM_IMAGE_KEY = "lingbot_vlm_image"
TEXT_ONLY_EMBEDS_KEY = "lingbot_text_only_embeds"


def condition_frame(image: PIL.Image.Image | list[PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    if not isinstance(image, PIL.Image.Image):
        raise ValueError("LingBot-Video TI2V requires exactly one PIL reference frame")
    return image


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

    # Only image-conditioned requests need the Qwen media processor.
    from sglang.srt.multimodal.qwen_vl_media_processing import smart_resize

    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    image = PIL.Image.fromarray(array, mode="RGB")
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=factor,
        min_pixels=4 * factor**2,
        max_pixels=16384 * factor**2,
    )
    return image.resize((resized_width, resized_height))
