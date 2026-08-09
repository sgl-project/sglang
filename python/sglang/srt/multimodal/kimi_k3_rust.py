"""Rust-accelerated Kimi-K3 image preprocessing (sglang-mm's ``kimi_k3`` module).

Library mode, like Inkling: the Python TokenizerManager keeps orchestrating
(decode, NaViT sizing, prompt expansion); resize -> transparent-bg composite ->
normalize -> patchify runs in Rust, bit-exact against the checkpoint's
PIL/numpy reference. Images whose PIL mode has no bit-exact Rust equivalent
report ``None`` and the caller falls back to the PIL path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sglang.srt.multimodal._core import kimi_k3 as _rs

# `TransparentBgConfig` defaults from the checkpoint's media_utils.py.
_BG_DEFAULTS = {
    "chessboard_square_size": 16,
    "chessboard_square_on_top_left": True,
    "chessboard_white_value": 255,
    "chessboard_gray_value": 200,
}


def bg_tuple(
    transparent_bg_config: Optional[dict],
) -> Optional[Tuple[str, int, bool, int, int]]:
    """Flatten a ``transparent_bg_config`` dict into the Rust binding's tuple."""
    if transparent_bg_config is None:
        return None
    cfg = {**_BG_DEFAULTS, **transparent_bg_config}
    return (
        cfg["pattern"],
        cfg["chessboard_square_size"],
        cfg["chessboard_square_on_top_left"],
        cfg["chessboard_white_value"],
        cfg["chessboard_gray_value"],
    )


def to_effective_array(image) -> Optional[np.ndarray]:
    """The u8 HWC array matching the reference's resize-in-original-mode
    semantics, or ``None`` when no such array exists.

    The reference converts modes only *after* the resize, so pre-converting is
    valid exactly when it commutes with PIL's per-channel resize: RGB passes
    through (even with a stray ``"transparency"`` info key, which the
    reference ignores on RGB), L/LA/RGBA convert by channel replication.
    Palette modes (PIL forces NEAREST) and CMYK-family conversions don't
    commute — no equivalent array.
    """
    if not isinstance(image, Image.Image):
        return None
    if image.mode == "RGB":
        return np.ascontiguousarray(np.asarray(image))
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        if image.mode == "L":
            return np.ascontiguousarray(np.asarray(image.convert("RGB")))
        return None
    if image.mode in ("RGBA", "LA"):
        return np.ascontiguousarray(np.asarray(image.convert("RGBA")))
    return None


def rust_preprocess_images(
    images: List,
    resize_configs: List[dict],
    *,
    patch_size: int,
    image_mean,
    image_std,
    transparent_bg_config: Optional[dict],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the Rust pipeline over a request's images.

    Returns ``(pixel_values, grid_thws)`` in the PIL path's schema, or
    ``None`` when any image has no effective array (the whole request then
    falls back to PIL so all its images take one path).
    """
    arrays = []
    for image in images:
        arr = to_effective_array(image)
        if arr is None:
            return None
        arrays.append(arr)

    bg = bg_tuple(transparent_bg_config)
    mean = tuple(float(v) for v in image_mean)
    std = tuple(float(v) for v in image_std)

    pixel_values = []
    grids = []
    for arr, cfg in zip(arrays, resize_configs):
        pix, grid = _rs.preprocess(
            arr,
            cfg["new_width"],
            cfg["new_height"],
            cfg["pad_width"],
            cfg["pad_height"],
            patch_size,
            mean,
            std,
            bg,
        )
        pixel_values.append(
            torch.from_numpy(pix.reshape(-1, 3, patch_size, patch_size))
        )
        grids.append(torch.tensor(grid, dtype=torch.int64))
    # Single image (the common case): skip torch.cat's whole-buffer copy.
    pixels = pixel_values[0] if len(pixel_values) == 1 else torch.cat(pixel_values)
    return pixels, torch.stack(grids)
