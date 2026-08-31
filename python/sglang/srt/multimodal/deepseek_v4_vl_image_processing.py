"""Image preprocessing for DeepSeek-V4-Flash-Vision.

A faithful port of ``inference/image_processor.py`` from the
``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`` checkpoint. Kept free of any
SGLang runtime imports so it can be unit-tested against the reference.

Two things happen here:

1.  **Resize solving.** The ViT sees ``patch_size``-aligned pixels; the aligner
    then folds every ``downsample_ratio``-squared block of ViT patches into one
    LLM token. ``solve_resize_ratio`` picks the largest aspect-preserving size
    whose resulting LLM-token block fits ``vision_max_n_token``.

2.  **Block layout.** One image does not occupy ``n_llm_h * n_llm_w`` contiguous
    LLM tokens. It occupies a longer block, in a row-pair-interleaved order the
    checkpoint calls the N-layout, framed by learned ``image_start`` /
    ``image_end`` embeddings, with a learned ``image_newline`` closing every grid
    row and learned ``image_pad`` filler for row/alignment padding.
    ``build_image_block`` returns the per-slot ``types`` in final sequence order
    plus ``perm``, which reorders the aligner's row-major output into the order
    its ``IMAGE`` slots appear in that block.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import msgspec
import numpy as np
import torch
from PIL import Image, ImageOps

# Slot kinds inside one image block, in final sequence order.
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
NUM_IMAGE_SLOT_KINDS = 5

# Every image block starts on a multiple of this many tokens, so an image never
# straddles a C4 compression group. Achieved with leading IMAGE_PAD slots.
COMPRESS_PAD_TO = 4


class DeepseekV4VisionParams(msgspec.Struct, frozen=True):
    """The ``vision_*`` fields of the checkpoint's ``config.json``."""

    patch_size: int = 14
    downsample_ratio: int = 3
    max_n_token: int = 384
    min_pixels: int = 147456
    # None disables the wide-image squash entirely.
    max_wh_ratio: Optional[int] = 8

    @classmethod
    def from_hf_config(cls, hf_config) -> DeepseekV4VisionParams:
        return cls(
            patch_size=hf_config.vision_patch_size,
            downsample_ratio=hf_config.vision_downsample_ratio,
            max_n_token=hf_config.vision_max_n_token,
            min_pixels=hf_config.vision_min_pixels,
            max_wh_ratio=hf_config.vision_max_wh_ratio,
        )


def grid_tokens(
    best_height: int, best_width: int, patch_size: int, downsample_ratio: int
) -> Tuple[int, int, int]:
    """LLM tokens an aligner grid occupies (N-layout, incl. row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> Tuple[int, int, int, int, int]:
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> Tuple[int, int, int, int]:
    # Reserve room for the worst-case leading COMPRESS_PAD_TO alignment padding
    # so the whole block still fits the per-image token budget.
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def _contain_size(size: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    """The size ``PIL.ImageOps.contain`` would resize ``size`` to (may be 0)."""
    width, height = size
    target_width, target_height = target
    if width == 0 or height == 0:
        return 0, 0
    if target_width / width < target_height / height:
        return target_width, round(height * target_width / width)
    return round(width * target_height / height), target_height


def preprocess_image(
    image: Image.Image, params: DeepseekV4VisionParams
) -> Tuple[torch.Tensor, int, int, int, int]:
    """Transform one PIL image into ViT patches plus its ViT and LLM grid dims.

    Returns ``(patches[n_vit_h * n_vit_w, 3, p, p] bf16, n_vit_h, n_vit_w,
    n_llm_h, n_llm_w)``.
    """
    p = params.patch_size
    image = image.convert("RGB")
    width, height = image.size
    if params.max_wh_ratio is not None and width > height * params.max_wh_ratio:
        width = height * params.max_wh_ratio
    if 0 < width * height < params.min_pixels:
        ratio = (params.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        params.downsample_ratio,
        params.max_n_token,
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if (
        params.max_wh_ratio is not None
        and image.width >= params.max_wh_ratio * image.height
    ):
        # Extreme panoramas are squashed rather than letterboxed: padding one to
        # the solved box would leave almost nothing but gray.
        image = image.resize((best_width, best_height))
    elif min(_contain_size(image.size, (best_width, best_height))) == 0:
        # Only max_wh_ratio-wide images are squashed above, so an extremely tall
        # one still reaches ImageOps.pad, where aspect-preserving containment
        # rounds its width to 0 and PIL raises. Squash it like a panorama rather
        # than fail the request (the reference implementation raises here).
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches.contiguous(), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Slot kinds in final sequence order, plus the aligner reorder and lead pad.

    ``start_pos`` is the absolute token index the block begins at; it only
    decides how many leading ``IMAGE_PAD`` slots align the block to
    ``COMPRESS_PAD_TO``.

    Returns ``(types[block_len], perm[n_llm_h * n_llm_w], compress_pad)`` where
    ``types == IMAGE`` marks the slots fed from the aligner and ``perm`` maps
    aligner row-major rows onto those slots in order.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    # N-layout: walk the grid in row pairs, column-major inside each pair.
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm, compress_pad
