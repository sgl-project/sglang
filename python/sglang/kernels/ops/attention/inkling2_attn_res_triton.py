# SPDX-License-Identifier: Apache-2.0
"""Single-launch Triton block attention-residual reduction (pre-Hopper).

:mod:`inkling2_attn_res_ref` matches the CuTe-DSL kernel bit for bit, but it
does so with ~60 eager kernels per call (fp32 widening, pow/mean/rsqrt, matmul,
cat, softmax, broadcast-mac) and the model calls it two or three times per
layer, so on A100 it became 56% of the decode step. This module keeps the same
math in one kernel:

    vnorm_n     = rsqrt(mean(x_n^2) + eps)
    logit_{q,n} = (q . x_n) * vnorm_n          (q pre-scaled by H**-0.5)
    out_q       = sum_n softmax_n(logit)_{q,n} * (vnorm_n if normalize_values) * x_n

One CTA owns one token: it scans every source row twice (score, then weighted
sum) so the row stays in L2, and the whole aggregation becomes a single launch.
The math is unchanged from the reference and from the CuTe kernel.
"""

import torch
import triton
import triton.language as tl

_EPS = 1e-6
_MAX_SOURCES = 16
# Cap on elements per thread: keeps the register footprint flat while the
# source count grows (LoopMessageBank can pass a dozen blocks).
_ELEMS_PER_THREAD = 32


@triton.jit
def _attn_res_kernel(
    # Device addresses of the [T, H] rows. They are scalar arguments rather than
    # a device pointer table on purpose: a table would need a host-to-device
    # copy at call time, and decode graph capture builds fresh buffers while
    # recording, so that copy would land inside the capture.
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    p10,
    p11,
    p12,
    p13,
    p14,
    p15,
    addend,  # [T, H], only read when HAS_ADDEND
    queries,  # [N_Q, H] fp32, pre-scaled
    out,  # [N_Q, T, H] fp32
    stride_src,
    stride_addend,
    stride_q,
    stride_out,  # T * H
    eps,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BF16: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    valid_n = offs_n < N
    addr = tl.full([BLOCK_N], p0, tl.int64)
    addr = tl.where(offs_n == 1, p1, addr)
    addr = tl.where(offs_n == 2, p2, addr)
    addr = tl.where(offs_n == 3, p3, addr)
    addr = tl.where(offs_n == 4, p4, addr)
    addr = tl.where(offs_n == 5, p5, addr)
    addr = tl.where(offs_n == 6, p6, addr)
    addr = tl.where(offs_n == 7, p7, addr)
    addr = tl.where(offs_n == 8, p8, addr)
    addr = tl.where(offs_n == 9, p9, addr)
    addr = tl.where(offs_n == 10, p10, addr)
    addr = tl.where(offs_n == 11, p11, addr)
    addr = tl.where(offs_n == 12, p12, addr)
    addr = tl.where(offs_n == 13, p13, addr)
    addr = tl.where(offs_n == 14, p14, addr)
    addr = tl.where(offs_n == 15, p15, addr)
    if BF16:
        src = addr.to(tl.pointer_type(tl.bfloat16))
    else:
        src = addr.to(tl.pointer_type(tl.float32))
    row_ptr = src + token * stride_src
    # arange needs a power of two; padded query columns are computed but never
    # stored, and the softmax is per query so they cannot leak into a real one.
    offs_q = tl.arange(0, BLOCK_Q)
    # The pending addend rides on the last block only.
    is_last = (offs_n == N - 1).to(tl.float32)
    src_mask = valid_n[:, None]

    # Pass 1: row L2 statistics and the N_Q query dot products.
    sumsq = tl.zeros([BLOCK_N], tl.float32)
    dots = tl.zeros([BLOCK_N, BLOCK_Q], tl.float32)
    for chunk in tl.static_range(0, (H + BLOCK_H - 1) // BLOCK_H):
        offs_h = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x = tl.load(
            row_ptr[:, None] + offs_h[None, :],
            mask=src_mask & mask_h[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_ADDEND:
            a = tl.load(addend + token * stride_addend + offs_h, mask=mask_h, other=0.0)
            x += a.to(tl.float32)[None, :] * is_last[:, None]
        sumsq += tl.sum(x * x, 1)
        for q in tl.static_range(N_Q):
            qv = tl.load(queries + q * stride_q + offs_h, mask=mask_h, other=0.0)
            dq = tl.sum(x * qv[None, :], 1)
            dots += tl.where((offs_q == q)[None, :], dq[:, None], 0.0)

    # Softmax over sources, per query.
    vnorm = tl.rsqrt(sumsq / H + eps)
    logit = tl.where(src_mask, dots * vnorm[:, None], -float("inf"))
    peak = tl.max(logit, 0)
    prob = tl.exp(logit - peak[None, :])
    prob = prob / tl.sum(prob, 0)[None, :]
    if NORMALIZE_VALUES:
        prob = prob * vnorm[:, None]

    # Pass 2: weighted sum, written one H chunk at a time.
    for chunk in tl.static_range(0, (H + BLOCK_H - 1) // BLOCK_H):
        offs_h = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x = tl.load(
            row_ptr[:, None] + offs_h[None, :],
            mask=src_mask & mask_h[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_ADDEND:
            a = tl.load(addend + token * stride_addend + offs_h, mask=mask_h, other=0.0)
            x += a.to(tl.float32)[None, :] * is_last[:, None]
        for q in tl.static_range(N_Q):
            w = tl.sum(tl.where((offs_q == q)[None, :], prob, 0.0), 1)
            y = tl.sum(w[:, None] * x, 0)
            tl.store(out + q * stride_out + token * H + offs_h, y, mask=mask_h)


def _fallback(blocks, partial_addend, queries, normalize_values):
    from sglang.kernels.ops.attention.inkling2_attn_res_ref import attn_res_ref

    return attn_res_ref(
        blocks,
        partial_addend,
        queries,
        normalize_values=normalize_values,
    )


def attn_res_fused(
    blocks,
    partial_addend,
    queries,
    *,
    normalize_values=False,
    eps=_EPS,
):
    """Drop-in replacement for ``attn_res_ref`` on pre-Hopper devices.

    Falls back to the torch reference for the shapes this kernel does not
    cover (no CUDA, unsupported dtype, more than _MAX_SOURCES blocks, or a
    query tensor that is not contiguous fp32).
    """
    base = blocks[0]
    if (
        not base.is_cuda
        or base.dtype not in (torch.bfloat16, torch.float32)
        or queries.dtype != torch.float32
        or not queries.is_contiguous()
        or len(blocks) > _MAX_SOURCES
    ):
        return _fallback(blocks, partial_addend, queries, normalize_values)
    tokens, width = base.shape
    if tokens == 0 or width == 0:
        return _fallback(blocks, partial_addend, queries, normalize_values)

    sources = [*blocks, partial_addend] if partial_addend is not None else list(blocks)
    for source in sources:
        assert source.shape == base.shape and source.dtype == base.dtype
        assert source.is_contiguous()

    n_q = queries.shape[0]
    n = len(blocks)
    out = torch.empty((n_q, tokens, width), dtype=torch.float32, device=base.device)
    block_n = triton.next_power_of_2(n)
    # Wide tiles keep the H scan to a couple of latency rounds; the source axis
    # is only a handful of rows so it costs almost nothing in registers.
    block_h = min(1024, max(128, triton.next_power_of_2(width)))
    addend = partial_addend if partial_addend is not None else base
    addr = [int(b.data_ptr()) for b in blocks]
    addr += [addr[-1]] * (_MAX_SOURCES - n)
    _attn_res_kernel[(tokens,)](
        *addr,
        addend,
        queries,
        out,
        base.stride(0),
        addend.stride(0),
        queries.stride(0),
        tokens * width,
        eps,
        H=width,
        N=n,
        BLOCK_N=block_n,
        N_Q=n_q,
        BLOCK_Q=triton.next_power_of_2(n_q),
        BLOCK_H=block_h,
        BF16=base.dtype == torch.bfloat16,
        NORMALIZE_VALUES=normalize_values,
        HAS_ADDEND=partial_addend is not None,
        num_warps=min(16, max(4, block_n * block_h // (32 * _ELEMS_PER_THREAD))),
    )
    return tuple(out.unbind(0))
