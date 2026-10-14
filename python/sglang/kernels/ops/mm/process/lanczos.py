# SPDX-License-Identifier: Apache-2.0
"""GPU RGB Lanczos resize, center crop and normalization with Pillow rounding."""

import math
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl


def _sinc(value: float) -> float:
    if value == 0.0:
        return 1.0
    angle = value * math.pi
    return math.sin(angle) / angle


@lru_cache(maxsize=128)
def _coefficients(source: int, target: int, device: torch.device):
    """Pillow's RGB resampling uses normalized signed 22-bit coefficients."""
    scale = source / target
    filter_scale = max(scale, 1.0)
    support = 3.0 * filter_scale
    inv_filter_scale = 1.0 / filter_scale
    taps = math.ceil(support) * 2 + 1
    coefficients = np.zeros((target, taps), dtype=np.int32)
    bounds = np.zeros((target, 2), dtype=np.int32)
    for out in range(target):
        center = (out + 0.5) * scale
        begin = max(int(center - support + 0.5), 0)
        end = min(int(center + support + 0.5), source)
        weights = []
        for pos in range(begin, end):
            distance = (pos - center + 0.5) * inv_filter_scale
            weights.append(
                _sinc(distance) * _sinc(distance / 3.0)
                if -3.0 <= distance < 3.0
                else 0.0
            )
        # Keep sequential double accumulation, including on Python versions
        # whose sum() uses compensated summation.
        total = 0.0
        for weight in weights:
            total += weight
        coefficients[out, : len(weights)] = [
            int(w / total * (1 << 22) + (0.5 if w >= 0.0 else -0.5)) for w in weights
        ]
        bounds[out] = (begin, len(weights))
    return (
        torch.from_numpy(coefficients).to(device),
        torch.from_numpy(bounds).to(device),
        taps,
    )


@lru_cache(maxsize=8)
def _normalization(device: torch.device):
    # Use the same float32 values as NumPy's existing pixel conversion. A table
    # avoids changing rounding through fused multiply-add or approximate divide.
    values = np.arange(256, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(values).to(device)


@triton.jit(
    do_not_specialize=[
        "source_height",
        "crop_left",
        "stride_y",
        "stride_x",
        "stride_channel",
        "taps",
    ]
)
def _horizontal_lanczos(
    src,
    dst,
    coefficients,
    bounds,
    source_height,
    crop_width: tl.constexpr,
    crop_left,
    stride_y,
    stride_x,
    stride_channel,
    taps,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < source_height * crop_width * 3
    channel = index % 3
    x = index // 3 % crop_width + crop_left
    y = index // (3 * crop_width)
    first = tl.load(bounds + x * 2, valid, 0)
    count = tl.load(bounds + x * 2 + 1, valid, 0)
    total = tl.full((block,), 1 << 21, tl.int32)
    for tap in range(taps):
        weight = tl.load(coefficients + x * taps + tap, valid, 0)
        value = tl.load(
            src + y * stride_y + (first + tap) * stride_x + channel * stride_channel,
            valid & (tap < count),
            0,
        ).to(tl.int32)
        total += value * weight
    pixel = tl.minimum(tl.maximum(total >> 22, 0), 255).to(tl.uint8)
    tl.store(dst + index, pixel, valid)


@triton.jit(do_not_specialize=["crop_top", "taps"])
def _vertical_lanczos_crop_normalize(
    src,
    dst,
    coefficients,
    bounds,
    normalization,
    crop_height: tl.constexpr,
    crop_width: tl.constexpr,
    crop_top,
    taps,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < crop_height * crop_width * 3
    channel = index // (crop_height * crop_width)
    y = index // crop_width % crop_height + crop_top
    x = index % crop_width
    first = tl.load(bounds + y * 2, valid, 0)
    count = tl.load(bounds + y * 2 + 1, valid, 0)
    total = tl.full((block,), 1 << 21, tl.int32)
    for tap in range(taps):
        weight = tl.load(coefficients + y * taps + tap, valid, 0)
        value = tl.load(
            src + ((first + tap) * crop_width + x) * 3 + channel,
            valid & (tap < count),
            0,
        ).to(tl.int32)
        total += value * weight
    pixel = tl.minimum(tl.maximum(total >> 22, 0), 255)
    value = tl.load(normalization + pixel)
    tl.store(dst + index, value, valid)


def resize_crop_normalize_rgb(
    image: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Resize CUDA uint8 HWC RGB to contiguous float32 CHW in [-1, 1].

    Uses aspect-preserving Lanczos resize followed by a center crop, matching
    Pillow's two-pass uint8 rounding. The input may have a padded row stride.
    The returned tensor owns its storage, while the input is borrowed only by
    the horizontal pass. Operations run on the current PyTorch CUDA stream.
    """
    if not image.is_cuda or torch.version.hip is not None:
        raise ValueError("Lanczos RGB preprocessing requires a CUDA tensor")
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Lanczos RGB input must be uint8 with shape [H, W, 3]")
    if height <= 0 or width <= 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Lanczos RGB input and output dimensions must be positive")

    source_h, source_w = image.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized_h = math.ceil(scale * source_h)
    resized_w = math.ceil(scale * source_w)
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    coeff_x, bounds_x, taps_x = _coefficients(source_w, resized_w, image.device)
    coeff_y, bounds_y, taps_y = _coefficients(source_h, resized_h, image.device)
    intermediate = torch.empty(
        (source_h, width, 3), dtype=torch.uint8, device=image.device
    )
    output = torch.empty((3, height, width), dtype=torch.float32, device=image.device)
    _horizontal_lanczos[(triton.cdiv(source_h * width * 3, 256),)](
        image,
        intermediate,
        coeff_x,
        bounds_x,
        source_h,
        width,
        left,
        *image.stride(),
        taps_x,
        256,
    )
    _vertical_lanczos_crop_normalize[(triton.cdiv(height * width * 3, 256),)](
        intermediate,
        output,
        coeff_y,
        bounds_y,
        _normalization(image.device),
        height,
        width,
        top,
        taps_y,
        256,
    )
    return output
