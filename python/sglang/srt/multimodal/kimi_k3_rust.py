"""Rust-accelerated Kimi-K3 image preprocessing (sglang-mm's ``kimi_k3`` module).

Library mode, like Inkling: the Python TokenizerManager keeps orchestrating
(decode, NaViT sizing, prompt expansion) and each decoded image's
resize -> transparent-bg composite -> normalize -> patchify runs in Rust,
bit-exact against the checkpoint's PIL/numpy reference
(``kimi_k3_vision_processing.py`` with ``transparent_bg_fill_stage ==
"after_resize"``).

The reference resizes each image in its *own* PIL mode and only converts
after the resize, so an input image is handed to Rust only when an
equivalent pre-converted RGB/RGBA array exists; anything else (palette
images, which PIL resizes with NEAREST; CMYK; bilevel; ...) reports "no
effective array" and the caller falls back to the PIL path for the request.
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
    """The u8 HWC array whose Rust pipeline output is bit-identical to the
    reference's resize-in-original-mode semantics, or ``None`` when no such
    array exists.

    * RGB passes through — including RGB images carrying a stray
      ``"transparency"`` info key, which ``fill_transparent_bg_with`` returns
      untouched before it ever inspects the alpha bands.
    * L converts to RGB up front: the conversion replicates the single channel
      and PIL's resize convolves channels independently, so convert-then-resize
      equals the reference's resize-then-convert.
    * RGBA / LA convert to RGBA up front, by the same per-channel argument.
    * Everything else has no equivalent array: PIL forces NEAREST resampling
      for palette ("P"/"1") modes, and CMYK-family conversions do not commute
      with the resize.
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

    Returns ``(pixel_values, grid_thws)`` matching the PIL path's schema —
    f32 ``(sum_patches, 3, ps, ps)`` and int64 ``(n_images, 3)`` — or ``None``
    when any image has no effective array (the caller falls back to PIL for
    the whole request so every image of a request takes one path).
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
    return torch.cat(pixel_values), torch.stack(grids)
