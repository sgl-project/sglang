"""GPU PIL-LANCZOS resampling with per-pass uint8 rounding.

Coefficient geometry and 22-bit quantization follow the shared Rust/Pillow
resampler. Keep the intermediate uint8 rounding; float bicubic is not equivalent.
"""

import math
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl


@lru_cache(maxsize=64)
def _coefficients(source: int, target: int):
    scale = source / target
    support = 3 * max(scale, 1.0)
    taps = 2 * math.ceil(support) + 1
    starts = np.empty(target, dtype=np.int32)
    weights = np.zeros((target, taps), dtype=np.int32)

    def sinc(x):
        return 1.0 if x == 0 else math.sin(x * math.pi) / (x * math.pi)

    for out in range(target):
        center = (out + 0.5) * scale
        start = max(0, int(center - support + 0.5))
        end = min(source, int(center + support + 0.5))
        values = []
        for i in range(start, end):
            x = (i - center + 0.5) / max(scale, 1.0)
            values.append(sinc(x) * sinc(x / 3) if -3 <= x < 3 else 0.0)
        total = sum(values)
        starts[out] = start
        for i, value in enumerate(values):
            value = value / total if total else value
            weights[out, i] = int(value * (1 << 22) + (-0.5 if value < 0 else 0.5))
    return starts, weights


@lru_cache(maxsize=32)
def _device_coefficients(source: int, target: int, device: int):
    starts, weights = _coefficients(source, target)
    return (torch.from_numpy(starts).to(device), torch.from_numpy(weights).to(device))


@triton.jit(do_not_specialize=["H", "W", "OW"])
def _horizontal(
    X,
    Y,
    Starts,
    Weights,
    H,
    W,
    OW,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < H * OW * 3
    c = idx % 3
    ox = idx // 3 % OW
    y = idx // (OW * 3)
    start = tl.load(Starts + ox, valid, 0)
    acc = tl.full((BLOCK,), 1 << 21, tl.int32)
    for k in range(K):
        weight = tl.load(Weights + ox * K + k, valid, 0)
        pixel = tl.load(X + (y * W + start + k) * 3 + c, valid & (start + k < W), 0).to(
            tl.int32
        )
        acc += pixel * weight
    value = tl.minimum(tl.maximum(acc >> 22, 0), 255)
    tl.store(Y + idx, value, valid)


@triton.jit(do_not_specialize=["H", "W", "OH"])
def _vertical_normalize(
    X,
    Y,
    Starts,
    Weights,
    Lut,
    H,
    W,
    OH,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < OH * W * 3
    c = idx % 3
    x = idx // 3 % W
    oy = idx // (W * 3)
    start = tl.load(Starts + oy, valid, 0)
    acc = tl.full((BLOCK,), 1 << 21, tl.int32)
    for k in range(K):
        weight = tl.load(Weights + oy * K + k, valid, 0)
        pixel = tl.load(
            X + ((start + k) * W + x) * 3 + c, valid & (start + k < H), 0
        ).to(tl.int32)
        acc += pixel * weight
    value = tl.minimum(tl.maximum(acc >> 22, 0), 255)
    normalized = tl.load(Lut + c * 256 + value, valid, 0)
    tl.store(Y + c * OH * W + oy * W + x, normalized, valid)


@lru_cache(maxsize=32)
def _device_lut(mean: tuple, std: tuple, device: int):
    values = torch.arange(256, dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0)
    table = (values[None, :] - torch.tensor(mean)[:, None]) / torch.tensor(std)[:, None]
    return table.to(device)


def resize_normalize_gpu(
    chw: torch.Tensor, target_h: int, target_w: int, mean: tuple, std: tuple
) -> torch.Tensor:
    """Accept RGB uint8 CHW, return normalized float32 CHW on the current GPU.

    CPU input incurs one uint8 H2D copy. CUDA input remains on its own device.
    Caches are bounded; no feature tensor or request data is retained.
    """
    if chw.ndim != 3 or chw.shape[0] != 3 or chw.dtype != torch.uint8:
        raise ValueError("expected RGB uint8 CHW input")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("resize dimensions must be positive")
    device = chw.device.index if chw.is_cuda else torch.cuda.current_device()
    with torch.cuda.device(device):
        h, w = chw.shape[-2:]
        src = chw.permute(1, 2, 0).contiguous().to(device)
        if w != target_w:
            starts, weights = _device_coefficients(w, target_w, device)
            intermediate = torch.empty(
                (h, target_w, 3), dtype=torch.uint8, device=device
            )
            _horizontal[(triton.cdiv(intermediate.numel(), 256),)](
                src,
                intermediate,
                starts,
                weights,
                h,
                w,
                target_w,
                weights.shape[1],
                256,
            )
        else:
            intermediate = src
        starts, weights = _device_coefficients(h, target_h, device)
        lut = _device_lut(tuple(mean), tuple(std), device)
        out = torch.empty((3, target_h, target_w), dtype=torch.float32, device=device)
        _vertical_normalize[(triton.cdiv(out.numel(), 256),)](
            intermediate,
            out,
            starts,
            weights,
            lut,
            h,
            target_w,
            target_h,
            weights.shape[1],
            256,
        )
        return out
