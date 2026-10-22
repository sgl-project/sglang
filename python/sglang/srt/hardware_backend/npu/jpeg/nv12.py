"""Route A: fused nearest-chroma NV12 -> RGB uint8 CHW on Ascend.

JPEG entropy decoding still belongs to DVPP. No CPU fallback is hidden here.
The rounding policy matches torch.round (ties to even); fusion is disabled to
preserve the staged float32 arithmetic of the historical torch_npu baseline.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _uint8_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    rounded = lower + increment.to(tl.float32)
    return tl.minimum(tl.maximum(rounded, 0.0), 255.0).to(tl.uint8)


@triton.jit
def _nv12_rgb(
    Packed,
    Out,
    W: tl.constexpr,
    H: tl.constexpr,
    WS: tl.constexpr,
    HS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pixel < W * H
    row, col = pixel // W, pixel % W
    y = tl.load(Packed + row * WS + col, valid, other=0).to(tl.float32)
    uv_index = WS * HS + (row // 2) * WS + (col // 2) * 2
    u = tl.load(Packed + uv_index, valid, other=128).to(tl.float32) - 128.0
    v = tl.load(Packed + uv_index + 1, valid, other=128).to(tl.float32) - 128.0
    r = y + 1.402 * v
    g = (y - 0.344136 * u) - 0.714136 * v
    b = y + 1.772 * u
    tl.store(Out + pixel, _uint8_even(r), valid)
    tl.store(Out + W * H + pixel, _uint8_even(g), valid)
    tl.store(Out + 2 * W * H + pixel, _uint8_even(b), valid)


def nv12_to_rgb_triton(packed, info, *, block=1024):
    """Read a padded NV12 buffer, return asynchronously on the current stream.

    Caller must keep the input alive and respect external producer completion.
    This implementation intentionally retains the existing D2D ownership copy.
    """
    w, h, ws, hs = info.width, info.height, info.width_stride, info.height_stride
    if packed.device.type != "npu" or packed.dtype != torch.uint8:
        raise ValueError("expected uint8 NPU input")
    if packed.ndim != 1 or not packed.is_contiguous():
        raise ValueError("expected a contiguous one-dimensional NV12 buffer")
    if min(w, h) <= 0 or w % 2 or h % 2 or ws < w or hs < h or ws % 2 or hs % 2:
        raise ValueError("invalid even NV12 dimensions/strides")
    if packed.numel() < ws * hs * 3 // 2:
        raise ValueError("NV12 input is too short")
    if block not in (256, 512, 1024, 2048):
        raise ValueError("unsupported block size")
    torch.npu.set_device(packed.device)
    out = torch.empty((3, h, w), dtype=torch.uint8, device=packed.device)
    _nv12_rgb[(triton.cdiv(w * h, block),)](
        packed, out, w, h, ws, hs, BLOCK=block, enable_fp_fusion=False
    )
    return out
