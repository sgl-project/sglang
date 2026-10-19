"""Fused mHC (manifold hyper-connection) kernels for Volundr decode: stream mixing and combine.

Why. A decode step of Volundr launched ~8,600 kernels, ~6,100 of them for mHC: per layer the
Sinkhorn projection alone is 10 iterations x (sum, clamp, div) x 2 = 60 one-microsecond kernels over a
4x4 matrix, plus ~25 more for the pre/post/res bookkeeping, the stream mix and the combine. This module
replaces them with two Triton kernels per layer:

  mhc_mix      d (the `dyn` projection output) + static params -> pre / post / res (Sinkhorn) and the
               layer input x_in = sum_s bf16(pre_s * H_s)
  mhc_combine  H_out = bf16(mixed + bf16(post * bf16(y - x_in))), plus the NEXT layer's H.mean(1)

What is NOT replaced: the RMSNorm, the `dyn` GEMM and the stream-mixing bmm (torch.einsum on bf16 =
cuBLAS tensor-core MMA, whose internal accumulation a scalar kernel would not reproduce bit-for-bit).

Bit-exactness is the design constraint, not an afterthought. Every rounding step of the torch reference
is reproduced in the same order:
  * fp32 adds of static params and deltas; exp via libdevice (the same __nv_expf torch's exp kernel
    compiles to); IEEE division (div_rn); clamp_min(1e-9) as a max with the same fp32 constant
  * torch's reduction order for a 4-element sum: sum(-1) is a 4-lane warp-shuffle tree
    (v0+v2)+(v1+v3), sum(-2) is one thread's serial ((v0+v1)+v2)+v3 (ROW_PAIRWISE / COL_PAIRWISE pick
    the order; the defaults were selected by bit-exact comparison against torch)
  * FMA contraction is OFF (enable_fp_fusion=False): with it on, Triton fuses mixed + bf16(post*delta)
    into one FMA and skips the bf16 rounding of the product torch performs (measured: 20 % of elements
    one ulp off)
  * every bf16 intermediate torch materialises (pre.to(bf16), each pre_s * H_s product, y - x_in,
    post * delta, the final add, the mean) is rounded to bf16 at the same point here.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mhc_mix_kernel(
    D_ptr,
    d_stride,
    PRE_ptr,
    POST_ptr,
    RES_ptr,
    H_ptr,
    h_st,
    h_ss,
    X_ptr,
    x_st,
    POSTO_ptr,
    RESO_ptr,
    DIM,
    RES32_ptr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PAIRWISE: tl.constexpr,
    COL_PAIRWISE: tl.constexpr,
    STORE_F32: tl.constexpr,
):
    t = tl.program_id(0)
    pb = tl.program_id(1)
    drow = D_ptr + t * d_stride
    # pre = pre_static.float() + d[:, :4]  -> pre.to(bf16)
    p0 = (
        (tl.load(PRE_ptr + 0).to(tl.float32) + tl.load(drow + 0).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    p1 = (
        (tl.load(PRE_ptr + 1).to(tl.float32) + tl.load(drow + 1).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    p2 = (
        (tl.load(PRE_ptr + 2).to(tl.float32) + tl.load(drow + 2).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    p3 = (
        (tl.load(PRE_ptr + 3).to(tl.float32) + tl.load(drow + 3).to(tl.float32))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    offs = pb * BLOCK + tl.arange(0, BLOCK)
    msk = offs < DIM
    hb = H_ptr + t * h_st + offs
    h0 = tl.load(hb, mask=msk, other=0.0).to(tl.float32)
    h1 = tl.load(hb + h_ss, mask=msk, other=0.0).to(tl.float32)
    h2 = tl.load(hb + 2 * h_ss, mask=msk, other=0.0).to(tl.float32)
    h3 = tl.load(hb + 3 * h_ss, mask=msk, other=0.0).to(tl.float32)
    # (pre_bf16 * H) is a bf16 tensor in torch: each product rounded, then summed in fp32, serially
    q0 = (p0 * h0).to(tl.bfloat16).to(tl.float32)
    q1 = (p1 * h1).to(tl.bfloat16).to(tl.float32)
    q2 = (p2 * h2).to(tl.bfloat16).to(tl.float32)
    q3 = (p3 * h3).to(tl.bfloat16).to(tl.float32)
    x = ((q0 + q1) + q2) + q3
    tl.store(X_ptr + t * x_st + offs, x.to(tl.bfloat16), mask=msk)
    if pb == 0:
        # post = post_static.float() + d[:, 4:8]; only post.to(bf16) is ever consumed
        prow = POSTO_ptr + t * 4
        tl.store(
            prow + 0,
            (
                tl.load(POST_ptr + 0).to(tl.float32) + tl.load(drow + 4).to(tl.float32)
            ).to(tl.bfloat16),
        )
        tl.store(
            prow + 1,
            (
                tl.load(POST_ptr + 1).to(tl.float32) + tl.load(drow + 5).to(tl.float32)
            ).to(tl.bfloat16),
        )
        tl.store(
            prow + 2,
            (
                tl.load(POST_ptr + 2).to(tl.float32) + tl.load(drow + 6).to(tl.float32)
            ).to(tl.bfloat16),
        )
        tl.store(
            prow + 3,
            (
                tl.load(POST_ptr + 3).to(tl.float32) + tl.load(drow + 7).to(tl.float32)
            ).to(tl.bfloat16),
        )
        # logits = res_static + d[8:], fp32 (exactly torch's res_static.float() + d.float())
        l00 = tl.load(RES_ptr + 0).to(tl.float32) + tl.load(drow + 8).to(tl.float32)
        l01 = tl.load(RES_ptr + 1).to(tl.float32) + tl.load(drow + 9).to(tl.float32)
        l02 = tl.load(RES_ptr + 2).to(tl.float32) + tl.load(drow + 10).to(tl.float32)
        l03 = tl.load(RES_ptr + 3).to(tl.float32) + tl.load(drow + 11).to(tl.float32)
        l10 = tl.load(RES_ptr + 4).to(tl.float32) + tl.load(drow + 12).to(tl.float32)
        l11 = tl.load(RES_ptr + 5).to(tl.float32) + tl.load(drow + 13).to(tl.float32)
        l12 = tl.load(RES_ptr + 6).to(tl.float32) + tl.load(drow + 14).to(tl.float32)
        l13 = tl.load(RES_ptr + 7).to(tl.float32) + tl.load(drow + 15).to(tl.float32)
        l20 = tl.load(RES_ptr + 8).to(tl.float32) + tl.load(drow + 16).to(tl.float32)
        l21 = tl.load(RES_ptr + 9).to(tl.float32) + tl.load(drow + 17).to(tl.float32)
        l22 = tl.load(RES_ptr + 10).to(tl.float32) + tl.load(drow + 18).to(tl.float32)
        l23 = tl.load(RES_ptr + 11).to(tl.float32) + tl.load(drow + 19).to(tl.float32)
        l30 = tl.load(RES_ptr + 12).to(tl.float32) + tl.load(drow + 20).to(tl.float32)
        l31 = tl.load(RES_ptr + 13).to(tl.float32) + tl.load(drow + 21).to(tl.float32)
        l32 = tl.load(RES_ptr + 14).to(tl.float32) + tl.load(drow + 22).to(tl.float32)
        l33 = tl.load(RES_ptr + 15).to(tl.float32) + tl.load(drow + 23).to(tl.float32)
        mx = tl.maximum(
            tl.maximum(
                tl.maximum(
                    tl.maximum(
                        tl.maximum(
                            tl.maximum(
                                tl.maximum(
                                    tl.maximum(
                                        tl.maximum(
                                            tl.maximum(
                                                tl.maximum(
                                                    tl.maximum(
                                                        tl.maximum(
                                                            tl.maximum(
                                                                tl.maximum(l00, l01),
                                                                l02,
                                                            ),
                                                            l03,
                                                        ),
                                                        l10,
                                                    ),
                                                    l11,
                                                ),
                                                l12,
                                            ),
                                            l13,
                                        ),
                                        l20,
                                    ),
                                    l21,
                                ),
                                l22,
                            ),
                            l23,
                        ),
                        l30,
                    ),
                    l31,
                ),
                l32,
            ),
            l33,
        )
        m00 = libdevice.exp(l00 - mx)
        m01 = libdevice.exp(l01 - mx)
        m02 = libdevice.exp(l02 - mx)
        m03 = libdevice.exp(l03 - mx)
        m10 = libdevice.exp(l10 - mx)
        m11 = libdevice.exp(l11 - mx)
        m12 = libdevice.exp(l12 - mx)
        m13 = libdevice.exp(l13 - mx)
        m20 = libdevice.exp(l20 - mx)
        m21 = libdevice.exp(l21 - mx)
        m22 = libdevice.exp(l22 - mx)
        m23 = libdevice.exp(l23 - mx)
        m30 = libdevice.exp(l30 - mx)
        m31 = libdevice.exp(l31 - mx)
        m32 = libdevice.exp(l32 - mx)
        m33 = libdevice.exp(l33 - mx)
        for _it in tl.static_range(ITERS):
            if ROW_PAIRWISE:
                r0 = (m00 + m02) + (m01 + m03)
            else:
                r0 = ((m00 + m01) + m02) + m03
            r0 = tl.maximum(r0, 1e-9)
            if ROW_PAIRWISE:
                r1 = (m10 + m12) + (m11 + m13)
            else:
                r1 = ((m10 + m11) + m12) + m13
            r1 = tl.maximum(r1, 1e-9)
            if ROW_PAIRWISE:
                r2 = (m20 + m22) + (m21 + m23)
            else:
                r2 = ((m20 + m21) + m22) + m23
            r2 = tl.maximum(r2, 1e-9)
            if ROW_PAIRWISE:
                r3 = (m30 + m32) + (m31 + m33)
            else:
                r3 = ((m30 + m31) + m32) + m33
            r3 = tl.maximum(r3, 1e-9)
            m00 = libdevice.div_rn(m00, r0)
            m01 = libdevice.div_rn(m01, r0)
            m02 = libdevice.div_rn(m02, r0)
            m03 = libdevice.div_rn(m03, r0)
            m10 = libdevice.div_rn(m10, r1)
            m11 = libdevice.div_rn(m11, r1)
            m12 = libdevice.div_rn(m12, r1)
            m13 = libdevice.div_rn(m13, r1)
            m20 = libdevice.div_rn(m20, r2)
            m21 = libdevice.div_rn(m21, r2)
            m22 = libdevice.div_rn(m22, r2)
            m23 = libdevice.div_rn(m23, r2)
            m30 = libdevice.div_rn(m30, r3)
            m31 = libdevice.div_rn(m31, r3)
            m32 = libdevice.div_rn(m32, r3)
            m33 = libdevice.div_rn(m33, r3)
            if COL_PAIRWISE:
                c0 = (m00 + m10) + (m20 + m30)
            else:
                c0 = ((m00 + m10) + m20) + m30
            c0 = tl.maximum(c0, 1e-9)
            if COL_PAIRWISE:
                c1 = (m01 + m11) + (m21 + m31)
            else:
                c1 = ((m01 + m11) + m21) + m31
            c1 = tl.maximum(c1, 1e-9)
            if COL_PAIRWISE:
                c2 = (m02 + m12) + (m22 + m32)
            else:
                c2 = ((m02 + m12) + m22) + m32
            c2 = tl.maximum(c2, 1e-9)
            if COL_PAIRWISE:
                c3 = (m03 + m13) + (m23 + m33)
            else:
                c3 = ((m03 + m13) + m23) + m33
            c3 = tl.maximum(c3, 1e-9)
            m00 = libdevice.div_rn(m00, c0)
            m01 = libdevice.div_rn(m01, c1)
            m02 = libdevice.div_rn(m02, c2)
            m03 = libdevice.div_rn(m03, c3)
            m10 = libdevice.div_rn(m10, c0)
            m11 = libdevice.div_rn(m11, c1)
            m12 = libdevice.div_rn(m12, c2)
            m13 = libdevice.div_rn(m13, c3)
            m20 = libdevice.div_rn(m20, c0)
            m21 = libdevice.div_rn(m21, c1)
            m22 = libdevice.div_rn(m22, c2)
            m23 = libdevice.div_rn(m23, c3)
            m30 = libdevice.div_rn(m30, c0)
            m31 = libdevice.div_rn(m31, c1)
            m32 = libdevice.div_rn(m32, c2)
            m33 = libdevice.div_rn(m33, c3)
        rrow = RESO_ptr + t * 16
        tl.store(rrow + 0, m00.to(tl.bfloat16))
        tl.store(rrow + 1, m01.to(tl.bfloat16))
        tl.store(rrow + 2, m02.to(tl.bfloat16))
        tl.store(rrow + 3, m03.to(tl.bfloat16))
        tl.store(rrow + 4, m10.to(tl.bfloat16))
        tl.store(rrow + 5, m11.to(tl.bfloat16))
        tl.store(rrow + 6, m12.to(tl.bfloat16))
        tl.store(rrow + 7, m13.to(tl.bfloat16))
        tl.store(rrow + 8, m20.to(tl.bfloat16))
        tl.store(rrow + 9, m21.to(tl.bfloat16))
        tl.store(rrow + 10, m22.to(tl.bfloat16))
        tl.store(rrow + 11, m23.to(tl.bfloat16))
        tl.store(rrow + 12, m30.to(tl.bfloat16))
        tl.store(rrow + 13, m31.to(tl.bfloat16))
        tl.store(rrow + 14, m32.to(tl.bfloat16))
        tl.store(rrow + 15, m33.to(tl.bfloat16))
        if STORE_F32:
            tl.store(RES32_ptr + t * 16 + 0, m00)
            tl.store(RES32_ptr + t * 16 + 1, m01)
            tl.store(RES32_ptr + t * 16 + 2, m02)
            tl.store(RES32_ptr + t * 16 + 3, m03)
            tl.store(RES32_ptr + t * 16 + 4, m10)
            tl.store(RES32_ptr + t * 16 + 5, m11)
            tl.store(RES32_ptr + t * 16 + 6, m12)
            tl.store(RES32_ptr + t * 16 + 7, m13)
            tl.store(RES32_ptr + t * 16 + 8, m20)
            tl.store(RES32_ptr + t * 16 + 9, m21)
            tl.store(RES32_ptr + t * 16 + 10, m22)
            tl.store(RES32_ptr + t * 16 + 11, m23)
            tl.store(RES32_ptr + t * 16 + 12, m30)
            tl.store(RES32_ptr + t * 16 + 13, m31)
            tl.store(RES32_ptr + t * 16 + 14, m32)
            tl.store(RES32_ptr + t * 16 + 15, m33)


@triton.jit
def _mhc_combine_kernel(
    M_ptr,
    m_st,
    m_ss,
    POST_ptr,
    Y_ptr,
    y_st,
    X_ptr,
    x_st,
    O_ptr,
    o_st,
    o_ss,
    MEAN_ptr,
    mean_st,
    DIM,
    BLOCK: tl.constexpr,
    WRITE_MEAN: tl.constexpr,
):
    t = tl.program_id(0)
    pb = tl.program_id(1)
    offs = pb * BLOCK + tl.arange(0, BLOCK)
    msk = offs < DIM
    y = tl.load(Y_ptr + t * y_st + offs, mask=msk, other=0.0).to(tl.float32)
    xi = tl.load(X_ptr + t * x_st + offs, mask=msk, other=0.0).to(tl.float32)
    delta = (y - xi).to(tl.bfloat16).to(tl.float32)  # y - x_in, a bf16 tensor
    mb = M_ptr + t * m_st + offs
    ob = O_ptr + t * o_st + offs
    w0 = tl.load(POST_ptr + t * 4 + 0).to(tl.float32)
    w1 = tl.load(POST_ptr + t * 4 + 1).to(tl.float32)
    w2 = tl.load(POST_ptr + t * 4 + 2).to(tl.float32)
    w3 = tl.load(POST_ptr + t * 4 + 3).to(tl.float32)
    o0 = (
        tl.load(mb, mask=msk, other=0.0).to(tl.float32)
        + (w0 * delta).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    o1 = (
        tl.load(mb + m_ss, mask=msk, other=0.0).to(tl.float32)
        + (w1 * delta).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    o2 = (
        tl.load(mb + 2 * m_ss, mask=msk, other=0.0).to(tl.float32)
        + (w2 * delta).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    o3 = (
        tl.load(mb + 3 * m_ss, mask=msk, other=0.0).to(tl.float32)
        + (w3 * delta).to(tl.bfloat16).to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(ob, o0, mask=msk)
    tl.store(ob + o_ss, o1, mask=msk)
    tl.store(ob + 2 * o_ss, o2, mask=msk)
    tl.store(ob + 3 * o_ss, o3, mask=msk)
    if WRITE_MEAN:
        # the next layer's H.mean(1): fp32 serial sum, times 0.25, rounded once
        s = ((o0.to(tl.float32) + o1.to(tl.float32)) + o2.to(tl.float32)) + o3.to(
            tl.float32
        )
        tl.store(MEAN_ptr + t * mean_st + offs, (s * 0.25).to(tl.bfloat16), mask=msk)


_BLOCK = 1024
ROW_PAIRWISE = True  # torch sum(-1) over 4 contiguous fp32: (v0+v2)+(v1+v3) (measured)
COL_PAIRWISE = False  # torch sum(-2) over 4 strided fp32: one thread, serial


def mhc_mix(
    d,
    pre_static,
    post_static,
    res_static,
    H,
    iters,
    row_pairwise=None,
    col_pairwise=None,
    res32=None,
):
    """d: [T, 24] (dyn output, bf16), H: [T, 4, DIM] bf16 -> (x_in [T, DIM], post_bf16 [T, 4], res_bf16 [T, 4, 4])."""
    T, n, DIM = H.shape
    assert (
        n == 4
        and d.shape[-1] == 2 * n + n * n
        and H.stride(2) == 1
        and d.stride(-1) == 1
    )
    x = torch.empty(T, DIM, dtype=H.dtype, device=H.device)
    post = torch.empty(T, n, dtype=torch.bfloat16, device=H.device)
    res = torch.empty(T, n, n, dtype=torch.bfloat16, device=H.device)
    if T == 0:
        return x, post, res
    grid = (T, triton.cdiv(DIM, _BLOCK))
    _mhc_mix_kernel[grid](
        d,
        d.stride(0),
        pre_static,
        post_static,
        res_static,
        H,
        H.stride(0),
        H.stride(1),
        x,
        x.stride(0),
        post,
        res,
        DIM,
        res32 if res32 is not None else res,
        ITERS=iters,
        BLOCK=_BLOCK,
        STORE_F32=res32 is not None,
        ROW_PAIRWISE=ROW_PAIRWISE if row_pairwise is None else row_pairwise,
        COL_PAIRWISE=COL_PAIRWISE if col_pairwise is None else col_pairwise,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return x, post, res


def mhc_combine(mixed, post_bf16, y, x_in, want_mean: bool):
    """mixed: [T, 4, DIM] (= einsum(res, H)), post_bf16 [T, 4], y / x_in [T, DIM]
    -> (H_out [T, 4, DIM], H_out.mean(1) or None)."""
    T, n, DIM = mixed.shape
    assert mixed.stride(2) == 1 and y.stride(-1) == 1 and x_in.stride(-1) == 1
    out = torch.empty(T, n, DIM, dtype=mixed.dtype, device=mixed.device)
    mean = (
        torch.empty(T, DIM, dtype=mixed.dtype, device=mixed.device)
        if want_mean
        else None
    )
    if T == 0:
        return out, mean
    grid = (T, triton.cdiv(DIM, _BLOCK))
    _mhc_combine_kernel[grid](
        mixed,
        mixed.stride(0),
        mixed.stride(1),
        post_bf16,
        y,
        y.stride(0),
        x_in,
        x_in.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        mean if want_mean else out,
        mean.stride(0) if want_mean else 0,
        DIM,
        BLOCK=_BLOCK,
        WRITE_MEAN=bool(want_mean),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, mean


# ---------------------------------------------------------------- torch references (the plain path)
def ref_sinkhorn(logits, iters):
    m = torch.exp(logits.float() - logits.float().amax(dim=(-1, -2), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(-1, keepdim=True).clamp_min(1e-9)
        m = m / m.sum(-2, keepdim=True).clamp_min(1e-9)
    return m


def ref_mix_from_d(d, pre_static, post_static, res_static, H, iters):
    n = 4
    d = d.float()
    pre = pre_static.float() + d[..., :n]
    post = post_static.float() + d[..., n : 2 * n]
    res = ref_sinkhorn(res_static.float() + d[..., 2 * n :].view(-1, n, n), iters)
    x_in = (pre.to(H.dtype).unsqueeze(-1) * H).sum(1)
    return x_in, post, res


def ref_combine(H, post, res, delta):
    mixed = torch.einsum("tij,tjd->tid", res.to(H.dtype), H)
    return mixed + post.to(H.dtype).unsqueeze(-1) * delta.unsqueeze(1)
