"""Rounding boundaries shared by the DSpark Q/K and context-write kernels.

Rounded mode requires enable_fp_fusion=False at launch: contraction can remove
the intermediate product rounding even with explicit narrow casts.
"""

import triton
import triton.language as tl


@triton.jit
def bf16_split_half_rope(x1, x2, cosine, sine, ROUND_INTERMEDIATES: tl.constexpr):
    if ROUND_INTERMEDIATES:
        cosine = cosine.to(tl.bfloat16).to(tl.float32)
        sine = sine.to(tl.bfloat16).to(tl.float32)
        xc = (x1 * cosine).to(tl.bfloat16).to(tl.float32)
        xs = (x1 * sine).to(tl.bfloat16).to(tl.float32)
        yc = (x2 * cosine).to(tl.bfloat16).to(tl.float32)
        ys = (x2 * sine).to(tl.bfloat16).to(tl.float32)
        return xc - ys, yc + xs
    return x1 * cosine - x2 * sine, x2 * cosine + x1 * sine
