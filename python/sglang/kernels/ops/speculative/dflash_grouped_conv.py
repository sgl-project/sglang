"""DFlash grouped dynamic convolution, fused with its neighbours (one launch each side).

The convolution mixes each token with the ``taps - 1`` tokens before it inside its draft block: per channel group
``g`` and tap ``j``, ``out[t] += (base[j] + delta[t, j, g]) * x[t - j]`` while ``t - j`` stays in the block. The
deltas come from one projection of the sublayer input, laid out ``[side, tap, group]`` (side 0 convolves the input,
side 1 the output).

* ``conv_prepare`` reads that projection as the row-major GEMM's split-K partials (summed here, rounded to BF16
  like the GEMM output) or as the BF16 GEMM output, convolves the input and keeps the side-1 deltas for later.
* ``conv_finish_add_rmsnorm`` convolves the sublayer output, adds the residual and applies the next RMSNorm.

Products and sums run in FP32 and round once, as the inductor-compiled reference does; the convolution output is
rounded to BF16 before the residual add, where the reference materializes it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _delta(C, t, col, M, NC: tl.constexpr, SK: tl.constexpr):
    if SK == 1:
        d = tl.load(C + t * NC + col).to(tl.float32)
    else:
        d = tl.load(C + t * NC + col)
        for s in tl.static_range(1, SK):
            d += tl.load(C + (s * M + t) * NC + col)
    # The reference reads these as the BF16 output of the projection GEMM.
    return d.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _conv_prepare_kernel(
    C,
    X,
    BASE,
    Y,
    D1,
    M,
    block_size,
    H: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    TAPS: tl.constexpr,
    SK: tl.constexpr,
    BG: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1) * BG + tl.arange(0, BG)
    c = g[:, None] * GS + tl.arange(0, GS)[None, :]
    NC: tl.constexpr = 2 * TAPS * G
    pos = t % block_size
    acc = tl.zeros((BG, GS), tl.float32)
    for j in tl.static_range(TAPS):
        d0 = _delta(C, t, j * G + g, M, NC, SK)
        d1 = _delta(C, t, (TAPS + j) * G + g, M, NC, SK)
        tl.store(D1 + (t * TAPS + j) * G + g, d1.to(D1.dtype.element_ty))
        coef = tl.load(BASE + j * H + c).to(tl.float32) + d0[:, None]
        x = tl.load(X + (t - j) * H + c, mask=(pos >= j) & (c >= 0), other=0.0)
        acc += coef * x.to(tl.float32)
    tl.store(Y + t * H + c, acc.to(Y.dtype.element_ty))


@triton.jit
def _conv_finish_add_rmsnorm_kernel(
    X,
    D1,
    BASE,
    R,
    W,
    Y,
    RO,
    block_size,
    eps,
    d1_stride_t,
    d1_stride_j,
    H: tl.constexpr,
    G: tl.constexpr,
    GS: tl.constexpr,
    TAPS: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.arange(0, G)
    c = g[:, None] * GS + tl.arange(0, GS)[None, :]
    pos = t % block_size
    acc = tl.zeros((G, GS), tl.float32)
    for j in tl.static_range(TAPS):
        d = tl.load(D1 + t * d1_stride_t + j * d1_stride_j + g).to(tl.float32)
        coef = tl.load(BASE + j * H + c).to(tl.float32) + d[:, None]
        x = tl.load(X + (t - j) * H + c, mask=(pos >= j) & (c >= 0), other=0.0)
        acc += coef * x.to(tl.float32)
    s = acc.to(tl.bfloat16).to(tl.float32) + tl.load(R + t * H + c).to(tl.float32)
    tl.store(RO + t * H + c, s.to(RO.dtype.element_ty))
    rstd = tl.rsqrt(tl.sum(tl.sum(s * s, axis=1), axis=0) / H + eps)
    w = tl.load(W + c).to(tl.float32)
    tl.store(Y + t * H + c, (s * rstd * w).to(Y.dtype.element_ty))


def conv_prepare(
    coefficients: torch.Tensor,
    hidden_states: torch.Tensor,
    base: torch.Tensor,
    *,
    block_size: int,
    taps: int,
    num_groups: int,
):
    """Side-0 convolution of ``hidden_states [M, H]`` and the side-1 deltas ``[M, taps, G]`` (BF16).

    ``coefficients`` is the projection of ``hidden_states``: FP32 split-K partials ``[SK, M, 2 * taps * G]`` or
    the BF16 product ``[M, 2 * taps * G]``. ``base`` is the side-0 base kernel ``[taps, H]``.
    """
    m, h = hidden_states.shape
    sk = 1 if coefficients.dim() == 2 else coefficients.shape[0]
    out = torch.empty_like(hidden_states)
    d1 = torch.empty(
        (m, taps, num_groups), device=hidden_states.device, dtype=torch.bfloat16
    )
    group_size = h // num_groups
    # 64 groups x 16 channels per program: 8 programs per row at H = 8192. Arbitrary, not tuned.
    bg = min(num_groups, 64)
    _conv_prepare_kernel[(m, num_groups // bg)](
        coefficients,
        hidden_states,
        base,
        out,
        d1,
        m,
        block_size,
        h,
        num_groups,
        group_size,
        taps,
        sk,
        bg,
        num_warps=4,
    )
    return out, d1


def conv_finish_add_rmsnorm(
    hidden_states: torch.Tensor,
    deltas: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    block_size: int,
):
    """``RMSNorm(conv(hidden_states) + residual)`` and the new residual, both BF16 ``[M, H]``.

    ``deltas [M, taps, G]`` are the side-1 deltas (any strides with a unit group stride); ``base`` is the side-1
    base kernel ``[taps, H]``.
    """
    m, h = hidden_states.shape
    taps, num_groups = deltas.shape[1:]
    out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(residual)
    _conv_finish_add_rmsnorm_kernel[(m,)](
        hidden_states,
        deltas,
        base,
        residual,
        weight,
        out,
        residual_out,
        block_size,
        eps,
        deltas.stride(0),
        deltas.stride(1),
        h,
        num_groups,
        h // num_groups,
        taps,
        num_warps=8,
    )
    return out, residual_out


def conv_applies(
    hidden_states: torch.Tensor, *, block_size: int, taps: int, num_groups: int
) -> bool:
    """Whether the fused kernels cover this convolution (power-of-two groups and group size, BF16 rows)."""
    h = hidden_states.shape[-1]
    group_size = h // num_groups
    return (
        hidden_states.dim() == 2
        and hidden_states.is_contiguous()
        and hidden_states.dtype == torch.bfloat16
        and num_groups * group_size == h
        and num_groups & (num_groups - 1) == 0
        and group_size & (group_size - 1) == 0
        and 1 <= taps <= block_size
    )
