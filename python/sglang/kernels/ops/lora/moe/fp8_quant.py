"""FP8 E4M3 quantization matching upstream's group and whole-row kernels.

Whole-row division uses fast math only on SM90; group quantization always does.
Preserve their different scale formulas and zero handling for bytewise parity.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_WHOLE_ROW_FAST_DIV = tl.constexpr(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
)


@triton.jit
def quantize_fp8_groups(grouped, WHOLE_ROW: tl.constexpr):
    """Return FP32 codes and per-group scales for [groups, group_size] input."""
    amax = tl.max(tl.abs(grouped), axis=1)
    if WHOLE_ROW:
        if _WHOLE_ROW_FAST_DIV:
            scale = amax / 448.0
            inv = tl.where(scale == 0.0, 0.0, 1.0 / scale)
        else:
            scale = libdevice.div_rn(amax, 448.0)
            inv = tl.where(scale == 0.0, 0.0, libdevice.div_rn(1.0, scale))
        q = grouped * inv[:, None]
    else:
        amax = tl.maximum(amax, 1e-10)
        scale = amax * (1.0 / 448.0)
        q = grouped * (448.0 / amax)[:, None]
    return tl.clamp(q, -448.0, 448.0), scale
