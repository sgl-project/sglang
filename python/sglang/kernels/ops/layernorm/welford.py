# SPDX-License-Identifier: Apache-2.0
"""ATen-compatible vectorized LayerNorm Welford reduction."""

import triton
import triton.language as tl


@triton.jit
def _rcp4(x):
    # nvcc's reciprocal fast path (always taken for our integer counts).
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .f32 r0, e, e2;
        rcp.approx.f32 r0, $1;
        fma.rn.f32 e, $1, r0, 0fBF800000;
        sub.ftz.f32 e2, 0f80000000, e;
        fma.rn.f32 $0, r0, e2, r0;
        }""",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _welford_push(val, mean, m2, cnt, valid, MASKED: tl.constexpr):
    # ``valid`` masks lanes whose aten thread does not execute this
    # iteration (their state must stay untouched).
    delta = val - mean
    new_cnt = cnt + 1.0
    recip = _rcp4(new_cnt)
    new_mean = tl.fma(delta, recip, mean)
    t = val - new_mean
    new_m2 = tl.fma(delta, t, m2)
    if MASKED:
        new_mean = tl.where(valid, new_mean, mean)
        new_m2 = tl.where(valid, new_m2, m2)
        new_cnt = tl.where(valid, new_cnt, cnt)
    return new_mean, new_m2, new_cnt


@triton.jit
def _welford_combine(mean_b, m2_b, cnt_b, mean_a, m2_a, cnt_a):
    # b = self / lower lane, a = other / upper lane; the op order matters.
    count = cnt_a + cnt_b
    pos = count > 0.0
    coef = _rcp4(tl.where(pos, count, 1.0))
    delta = mean_b - mean_a
    n_b = coef * cnt_b
    d2 = delta * delta
    n_a = cnt_a * coef
    s = m2_a + m2_b
    t1 = n_b * mean_b
    mean = tl.fma(mean_a, n_a, t1)
    t2 = cnt_a * d2
    m2 = tl.fma(n_b, t2, s)
    mean = tl.where(pos, mean, 0.0)
    m2 = tl.where(pos, m2, 0.0)
    return mean, m2, count


@triton.jit
def _split_halves(x, rows: tl.constexpr, half: tl.constexpr):
    # (rows, 2*half) -> two (rows, half) tensors pairing lane i with i+half,
    # the tree a shfl.down fold with the largest offset first produces.
    x = tl.reshape(x, (rows, 2, half), can_reorder=False)
    x = tl.permute(x, (0, 2, 1))
    return tl.split(x)


@triton.jit
def _fold_halves(mean, m2, cnt, rows: tl.constexpr, half: tl.constexpr):
    mb, ma = _split_halves(mean, rows, half)
    sb, sa = _split_halves(m2, rows, half)
    cb, ca = _split_halves(cnt, rows, half)
    return _welford_combine(mb, sb, cb, ma, sa, ca)


@triton.jit
def _fold_tree_32(mean, m2, cnt, rows: tl.constexpr):
    # shfl.down offsets 16, 8, 4, 2, 1 over the 32 lane states -> (rows, 1).
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 16)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 8)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 4)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 2)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, rows, 1)
    return mean, m2, cnt


@triton.jit
def _push_vec4(
    x4,
    mean,
    m2,
    cnt,
    valid,
    rows: tl.constexpr,
    lanes: tl.constexpr,
    MASKED: tl.constexpr,
):
    # Push the 4 elements of one aligned_vector<bf16, 4> in exact serial
    # order.  x4 is (rows, lanes, 4) fp32.
    g = tl.reshape(x4, (rows, lanes, 2, 2), can_reorder=False)
    p02, p13 = tl.split(g)  # elements (0, 2) / (1, 3)
    e0, e2 = tl.split(tl.reshape(p02, (rows, lanes, 1, 2), can_reorder=False))
    e1, e3 = tl.split(tl.reshape(p13, (rows, lanes, 1, 2), can_reorder=False))
    e0 = tl.reshape(e0, (rows, lanes), can_reorder=False)
    e1 = tl.reshape(e1, (rows, lanes), can_reorder=False)
    e2 = tl.reshape(e2, (rows, lanes), can_reorder=False)
    e3 = tl.reshape(e3, (rows, lanes), can_reorder=False)
    mean, m2, cnt = _welford_push(e0, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e1, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e2, mean, m2, cnt, valid, MASKED)
    mean, m2, cnt = _welford_push(e3, mean, m2, cnt, valid, MASKED)
    return mean, m2, cnt


@triton.jit
def layernorm_welford(x_ptr, row_base, row_mask, D: tl.constexpr, ROWS: tl.constexpr):
    lanes = tl.arange(0, 128)
    mean = tl.zeros((ROWS, 128), dtype=tl.float32)
    m2 = tl.zeros((ROWS, 128), dtype=tl.float32)
    cnt = tl.zeros((ROWS, 128), dtype=tl.float32)

    # pass 1: per-"thread" serial Welford in aten's exact element order.
    # Out-of-range rows compute garbage that is never stored.
    for i in tl.static_range((D + 511) // 512):
        cols = i * 512 + lanes[:, None] * 4 + tl.arange(0, 4)[None, :]
        if (i + 1) * 512 <= D:
            x4 = tl.load(
                x_ptr + row_base[:, None, None] + cols[None, :, :],
                mask=row_mask[:, None, None],
                other=0.0,
            ).to(tl.float32)
            mean, m2, cnt = _push_vec4(
                x4, mean, m2, cnt, row_mask, ROWS, 128, MASKED=False
            )
        else:
            # partial tail chunk (D % 512 != 0): aten threads whose vector
            # index i*128+t reaches N/4 skip the iteration, leaving their
            # Welford state untouched.
            vec_valid = (i * 128 + lanes < D // 4)[None, :]
            x4 = tl.load(
                x_ptr + row_base[:, None, None] + cols[None, :, :],
                mask=row_mask[:, None, None] & vec_valid[:, :, None],
                other=0.0,
            ).to(tl.float32)
            mean, m2, cnt = _push_vec4(
                x4, mean, m2, cnt, vec_valid, ROWS, 128, MASKED=True
            )

    # warp fold trees, then the (0,2)/(1,3)/(0,1) inter-warp combines.
    mean = tl.reshape(mean, (ROWS * 4, 32), can_reorder=False)
    m2 = tl.reshape(m2, (ROWS * 4, 32), can_reorder=False)
    cnt = tl.reshape(cnt, (ROWS * 4, 32), can_reorder=False)
    mean, m2, cnt = _fold_tree_32(mean, m2, cnt, ROWS * 4)
    mean = tl.reshape(mean, (ROWS, 4), can_reorder=False)
    m2 = tl.reshape(m2, (ROWS, 4), can_reorder=False)
    cnt = tl.reshape(cnt, (ROWS, 4), can_reorder=False)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, ROWS, 2)
    mean, m2, cnt = _fold_halves(mean, m2, cnt, ROWS, 1)

    return mean, m2
