"""Match upstream FP8 scale arithmetic, with zero codes for zero-scale rows.

Whole-row division uses fast math only on SM90; group quantization always does.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.arch import current_arch

_WHOLE_ROW_FAST_DIV = tl.constexpr(current_arch().is_sm(9, 0))


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
