# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 image generation canvas and the resize back to the requested size.

A request smaller than ``min_generation_area`` pixels is generated on a larger
canvas with the same aspect ratio: both sides scale by
``sqrt(min_generation_area / (width * height))`` and round to the nearest 32px
multiple (one 2x2 patch of 16px VAE cells). A request at or above the area is
only aligned to the 32px grid. After decoding, the canvas image is resized back
to the requested width x height with PIL LANCZOS on the 8-bit image.

The resize runs on the accelerator when it can reproduce Pillow byte for byte
(see ``_gpu_lanczos_resize_chw``); otherwise it is the Pillow call itself.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from math import sqrt

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen import envs

logger = logging.getLogger(__name__)

MINIMAX_H3_IMAGE_PIXEL_GRID = 32
# 32x32 is the smallest aligned request, so the lower bound never enlarges a
# request; the upper bound is the 4096x4096 tier.
MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE = (32 * 32, 4096 * 4096)
# Pillow's 8-bit resampling keeps filter taps as fixed-point integers with this
# many fractional bits (``PRECISION_BITS = 32 - 8 - 2``).
_PIL_COEFF_BITS = 22


def minimax_h3_validate_min_generation_area(value: object, path: str) -> int:
    low, high = MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(
            f"{path} must be an integer pixel area within [{low}, {high}], got {value!r}"
        )
    return value


def _nearest_grid_multiple(value: float) -> int:
    # Python round: ties to even.
    grid = MINIMAX_H3_IMAGE_PIXEL_GRID
    return max(grid, int(round(value / grid)) * grid)


def minimax_h3_image_generation_canvas(
    width: int, height: int, min_generation_area: int
) -> tuple[int, int]:
    """The 32px-aligned ``(width, height)`` the DiT generates for a request."""
    scale = max(1.0, sqrt(min_generation_area / (width * height)))
    return _nearest_grid_multiple(width * scale), _nearest_grid_multiple(height * scale)


@functools.lru_cache(maxsize=None)
def _resize_pool(threads: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix="h3-resize")


def _pil_lanczos_resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """PIL LANCZOS resize of an HWC uint8 array, byte-identical to one ``Image.resize``.

    With an integral vertical scale the output rows are split into strips; each
    strip resizes the matching integral input box, which has the same scale and
    the same sample centers as the full call. Pillow releases the GIL while
    resampling, so the strips run in parallel. Any other geometry is a single call.
    """
    in_h, in_w = array.shape[:2]
    image = Image.fromarray(array)
    threads = max(
        1,
        min(envs.SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_THREADS, os.cpu_count() or 1),
    )
    if threads == 1 or height < 2 * threads or in_h % height != 0:
        return np.array(image.resize((width, height), Image.Resampling.LANCZOS))
    scale = in_h // height
    bounds = [height * i // threads for i in range(threads + 1)]
    out = np.empty((height, width, array.shape[2]), dtype=np.uint8)

    def strip(i: int) -> None:
        r0, r1 = bounds[i], bounds[i + 1]
        piece = image.resize(
            (width, r1 - r0),
            Image.Resampling.LANCZOS,
            box=(0, r0 * scale, in_w, r1 * scale),
        )
        out[r0:r1] = np.asarray(piece)

    list(_resize_pool(threads).map(strip, range(threads)))
    return out


@functools.lru_cache(maxsize=8)
def _pil_lanczos_coeffs(in_size: int, out_size: int) -> np.ndarray:
    """Dense ``[out_size, in_size]`` int64 matrix of Pillow's fixed-point LANCZOS taps.

    Mirrors Pillow's ``precompute_coeffs`` and ``normalize_coeffs_8bpc`` for a
    full-image box, so that a resize can be reproduced with exact integer math.
    """
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 3.0 * filterscale
    inv = 1.0 / filterscale

    def sinc(x: float) -> float:
        if x == 0.0:
            return 1.0
        x = x * math.pi
        return math.sin(x) / x

    def lanczos(x: float) -> float:
        return sinc(x) * sinc(x / 3.0) if -3.0 <= x < 3.0 else 0.0

    mat = np.zeros((out_size, in_size), dtype=np.int64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        taps = [lanczos((x + xmin - center + 0.5) * inv) for x in range(xmax)]
        total = sum(taps)
        for x, w in enumerate(taps):
            if total != 0.0:
                w /= total
            # C-style ``(int)(0.5 + w * 2**22)``, rounding half away from zero.
            fixed = (
                int(-0.5 + w * (1 << _PIL_COEFF_BITS))
                if w < 0
                else int(0.5 + w * (1 << _PIL_COEFF_BITS))
            )
            mat[xx, xmin + x] = fixed
    return mat


# The dense float64 matrices are up to a few hundred MB for the largest canvases
# (a 4096-wide axis is 128 MiB), so only a few geometries stay resident.
@functools.lru_cache(maxsize=4)
def _pil_lanczos_matrix_device(
    in_size: int, out_size: int, device: torch.device
) -> torch.Tensor:
    coeffs = _pil_lanczos_coeffs(in_size, out_size)
    return torch.from_numpy(coeffs.astype(np.float64)).to(device)


def _pil_round_clip(acc: torch.Tensor) -> torch.Tensor:
    # Pillow: out = clip8((1 << 21) + sum(k * px) >> 22); the arithmetic shift is a floor.
    half = float(1 << (_PIL_COEFF_BITS - 1))
    return torch.floor((acc + half) / float(1 << _PIL_COEFF_BITS)).clamp_(0, 255)


def _gpu_lanczos_resize_chw(q: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Pillow-exact LANCZOS resize of a ``[3, H, W]`` uint8 device tensor.

    Pillow resamples horizontally, rounds to 8 bit, then resamples vertically.
    Each pass here is a float64 matmul with the integer tap matrix: every
    product and partial sum is an integer far below 2**53, so the result is
    exact whatever the accumulation order of the GEMM (float64 is not affected
    by TF32). Returns a ``[3, height, width]`` uint8 tensor.
    """
    c, h, w = q.shape
    kx = _pil_lanczos_matrix_device(w, width, q.device)  # [width, w]
    ky = _pil_lanczos_matrix_device(h, height, q.device)  # [height, h]
    tmp = _pil_round_clip(q.reshape(c * h, w).to(torch.float64) @ kx.t())
    out = _pil_round_clip(torch.matmul(ky, tmp.view(c, h, width)))
    return out.to(torch.uint8)


_GPU_RESIZE_VERDICT: dict[tuple, bool] = {}


def _gpu_resize_is_exact(
    in_h: int, in_w: int, height: int, width: int, device: torch.device
) -> bool:
    """Whether the device resize equals Pillow for this geometry (checked once).

    The first request with a given (input size, output size) resizes one random
    8-bit noise image both ways. Noise exercises every filter tap, so a tap that
    differs from Pillow's by a single unit changes some output byte. This guards
    against a Pillow version that resamples differently or a device without a
    working float64 matmul; on any difference (or error) that geometry keeps
    using Pillow. The one-time cost is one Pillow resize of the canvas
    (~40 ms for 2720x1536 -> 1344x768) plus building the tap matrices.
    """
    key = (in_h, in_w, height, width, device.type)
    verdict = _GPU_RESIZE_VERDICT.get(key)
    if verdict is not None:
        return verdict
    try:
        noise = np.random.default_rng(1234).integers(
            0, 256, (in_h, in_w, 3), dtype=np.uint8
        )
        ref = np.asarray(
            Image.fromarray(noise).resize((width, height), Image.Resampling.LANCZOS)
        )
        q = torch.from_numpy(noise).to(device).permute(2, 0, 1).contiguous()
        got = _gpu_lanczos_resize_chw(q, width, height).permute(1, 2, 0)
        verdict = bool(np.array_equal(got.cpu().numpy(), ref))
    except Exception:
        logger.warning("H3 device LANCZOS resize self-check failed", exc_info=True)
        verdict = False
    _GPU_RESIZE_VERDICT[key] = verdict
    logger.info(
        "H3 device LANCZOS resize %dx%d -> %dx%d: %s",
        in_w,
        in_h,
        width,
        height,
        "byte-identical to Pillow, enabled" if verdict else "differs, using Pillow",
    )
    return verdict


def minimax_h3_resize_image_output(
    pixels: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Resize decoded ``[B, 3, H, W]`` pixels in [0, 1] to ``width x height``.

    Each image is quantized exactly as the output writer does
    (``(x * 255).clamp(0, 255).to(uint8)``), resized with PIL LANCZOS and
    returned on the host as ``uint8 / 255``, which the writer maps back to the
    same bytes.

    ``pixels`` may live on a CUDA/ROCm device: quantization (IEEE fp32 multiply,
    clamp, truncate) is bit-identical there, so only uint8 bytes cross to the
    host; and the resize runs on the device when it is verified byte-identical
    to Pillow for this geometry (``SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_GPU=0``
    disables that), else in Pillow on ``SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_THREADS``
    threads. Tensors on other backends are quantized on the host as before.
    """
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError(
            f"expected decoded [B, 3, H, W] pixels, got {list(pixels.shape)}"
        )
    if not pixels.is_cuda:
        pixels = pixels.cpu()
    use_device = (
        pixels.is_cuda
        and envs.SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_GPU
        and _gpu_resize_is_exact(
            int(pixels.shape[2]), int(pixels.shape[3]), height, width, pixels.device
        )
    )
    resized = []
    for image in pixels:
        quantized = (image * 255).clamp(0, 255).to(torch.uint8)
        if use_device:
            resized.append(_gpu_lanczos_resize_chw(quantized, width, height).cpu())
            continue
        array = quantized.permute(1, 2, 0).contiguous().cpu().numpy()
        output = _pil_lanczos_resize(array, width, height)
        resized.append(torch.from_numpy(output).permute(2, 0, 1))
    return torch.stack(resized).float().div_(255)


__all__ = [
    "MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE",
    "MINIMAX_H3_IMAGE_PIXEL_GRID",
    "minimax_h3_image_generation_canvas",
    "minimax_h3_resize_image_output",
    "minimax_h3_validate_min_generation_area",
]
