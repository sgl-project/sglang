"""Large HIP mHC boundaries with compensated BF16 projection."""

import triton
import triton.language as tl


@triton.jit
def _hc_post_combine(
    X,
    R,
    P,
    C,
    PRE,
    RO,
    Y,
    H: tl.constexpr,
    HAS_POST: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.program_id(1) * B + tl.arange(0, B)
    mask = d < H
    r0 = tl.load(R + row * 4 * H + d, mask, 0).to(tl.float32)
    r1 = tl.load(R + (row * 4 + 1) * H + d, mask, 0).to(tl.float32)
    r2 = tl.load(R + (row * 4 + 2) * H + d, mask, 0).to(tl.float32)
    r3 = tl.load(R + (row * 4 + 3) * H + d, mask, 0).to(tl.float32)
    if HAS_POST:
        x = tl.load(X + row * H + d, mask, 0).to(tl.float32)
    # Keep the native HIP boundary's separate multiplies/adds and BF16 rounding.
    y = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(4):
        if HAS_POST:
            v = x * tl.load(P + row * 4 + k)
            v = v + r0 * tl.load(C + row * 16 + k)
            v = v + r1 * tl.load(C + row * 16 + 4 + k)
            v = v + r2 * tl.load(C + row * 16 + 8 + k)
            v = v + r3 * tl.load(C + row * 16 + 12 + k)
            rounded = v.to(tl.bfloat16)
            tl.store(RO + (row * 4 + k) * H + d, rounded, mask)
        else:
            rounded = tl.load(R + (row * 4 + k) * H + d, mask, 0)
        if HAS_COMBINE:
            y = y + rounded.to(tl.float32) * tl.load(PRE + row * 4 + k)
    if HAS_COMBINE:
        tl.store(Y + row * H + d, y, mask)


def hc_boundary_bf16x3_partials(
    x, residual, post, comb, pre_prev, residual_out, y, weight_parts
):
    from .mhc import hc_mix_stats_bf16x3_partials

    m, _, h = residual.shape
    if x is not None or pre_prev is not None:
        _hc_post_combine[(m, triton.cdiv(h, 1024))](
            x if x is not None else residual,
            residual,
            post if post is not None else residual,
            comb if comb is not None else residual,
            pre_prev if pre_prev is not None else residual,
            residual_out if residual_out is not None else residual,
            y if y is not None else residual,
            H=h,
            HAS_POST=x is not None,
            HAS_COMBINE=pre_prev is not None,
            B=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    values = residual_out if residual_out is not None else residual
    return hc_mix_stats_bf16x3_partials(values.flatten(1), weight_parts)
