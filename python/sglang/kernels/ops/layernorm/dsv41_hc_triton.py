"""Triton-ascend HC (hyper-connection) kernels for DeepSeek-V4.1 (A2/A3/A5).

Plan-1 two-kernel split of the ``hc_pre_from_prev`` sublayer boundary. The
hc chain on NPU runs an all-torch
fallback whose sinkhorn loop alone is ~114 [T,4,4] elementwise launches per
mix call (~1.1e4 launches per decode step at 40 layers); these two kernels
collapse each sublayer boundary to two launches:

* ``dsv41_hc_mix_sinkhorn_combine`` -- one program per token: mix GEMV +
  sum-of-squares over the [T, M*H] residual stream, sigmoid/sinkhorn on the
  4x4 comb entirely in registers, then the pre-weighted combine. Emits the
  boundary state ``mix3 [T, 24]`` (pre | post | comb, the row layout of
  ``mixes``).
* ``dsv41_hc_post`` -- out[j,h] = post_j*x_h + sum_k comb_jk*res_kh, the
  residual-stream fold after each sublayer.

Semantic source: mhc.py ``_hc_split_sinkhorn_torch`` / ``hc_combine`` /
``hc_post_torch_impl`` and DeepseekV4DecoderLayer. The eps placement is
load-bearing: pre = sigmoid + eps, post = 2*sigmoid (no eps), comb gets eps
once after the row softmax and then in every row/col normalize denominator.
Batch-invariance: one program per token; the H-chunk loops keep a fixed
order, so no cross-program reduction exists at all.

triton-ascend device code: the kernels compile and run only on the card, so
off-device hosts validate the math through a CPU mirror and a CANNON recipe
leg that live on the dsv41-hc-triton-a3 branch. Only hc_mult=4 is supported
(power-of-two tl.arange tiles).

NOTE: the jit kernels are defined at MODULE level under a guarded import.
triton resolves names inside jit code from module globals only -- kernels
defined inside a factory function (local ``tl``, closure helpers) fail on
card with NameError('tl is not defined').
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORTABLE = True
except Exception:  # no triton: the module must still import for the gate
    _TRITON_IMPORTABLE = False

MIX_HC = 24  # (2 + hc_mult) * hc_mult for hc_mult=4


def triton_available() -> bool:
    return _TRITON_IMPORTABLE


if _TRITON_IMPORTABLE:
    # Kernels live at module level on purpose: triton resolves names inside
    # jit code from module globals only, so closure-defined kernels (a
    # factory importing tl locally) fail on card with
    # NameError('tl is not defined').

    @triton.jit
    def dsv41_hc_mix_sinkhorn_combine_kernel(
        x_ptr,        # [T, K] bf16/fp16, K = M*H (the residual stream, flattened)
        fn_ptr,       # [MIX_HC, K] fp32 (hc_fn weight, F.linear layout)
        scale_ptr,    # [3] fp32
        base_ptr,     # [MIX_HC] fp32
        pre_ptr,      # [T, M] fp32, previous boundary's pre (combine weights)
        y_ptr,        # [T, H] bf16/fp16 out
        mix3_ptr,     # [T, MIX_HC] fp32 out: pre | post | comb
        K,            # runtime: M * H
        PRE_STRIDE,   # runtime: row stride of pre_ptr
        H: tl.constexpr,
        M: tl.constexpr,
        SINKHORN_ITERS: tl.constexpr,
        RMS_EPS: tl.constexpr,
        HC_EPS: tl.constexpr,
        BH: tl.constexpr,
        HAS_PRE: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.arange(0, BH)
        offs_m = tl.arange(0, M)
        offs_c = tl.arange(0, M * M)

        # pass 1: mix GEMV in three segments + sum of squares, fp32 accumulate
        acc_pre = tl.zeros([M, BH], dtype=tl.float32)
        acc_post = tl.zeros([M, BH], dtype=tl.float32)
        acc_comb = tl.zeros([M * M, BH], dtype=tl.float32)
        ssq = 0.0
        for h0 in range(0, K, BH):
            xv = tl.load(x_ptr + t * K + h0 + offs).to(tl.float32)
            wp = tl.load(fn_ptr + offs_m[:, None] * K + (h0 + offs)[None, :])
            wq = tl.load(fn_ptr + (M + offs_m)[:, None] * K + (h0 + offs)[None, :])
            wc = tl.load(fn_ptr + (2 * M + offs_c)[:, None] * K + (h0 + offs)[None, :])
            acc_pre += wp * xv[None, :]
            acc_post += wq * xv[None, :]
            acc_comb += wc * xv[None, :]
            ssq += tl.sum(xv * xv)
        rsqrt = tl.rsqrt(ssq / K + RMS_EPS)
        pre_mix = tl.sum(acc_pre, axis=1) * rsqrt
        post_mix = tl.sum(acc_post, axis=1) * rsqrt
        comb_mix = tl.reshape(tl.sum(acc_comb, axis=1) * rsqrt, (M, M))

        # register epilogue: gate + sinkhorn (eps placement mirrors the torch
        # reference exactly -- see the module docstring)
        scale0 = tl.load(scale_ptr + 0)
        scale1 = tl.load(scale_ptr + 1)
        scale2 = tl.load(scale_ptr + 2)
        pre = tl.sigmoid(pre_mix * scale0 + tl.load(base_ptr + offs_m)) + HC_EPS
        post = 2.0 * tl.sigmoid(post_mix * scale1 + tl.load(base_ptr + M + offs_m))
        comb = comb_mix * scale2 + tl.load(base_ptr + 2 * M + offs_c).reshape(M, M)
        # initial row softmax (stabilized, eps added after the divide), then
        # column normalize with eps inside the denominator
        comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
        comb = comb / tl.sum(comb, axis=1)[:, None] + HC_EPS
        comb = comb / (tl.sum(comb, axis=0)[None, :] + HC_EPS)
        for _ in tl.static_range(SINKHORN_ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + HC_EPS)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + HC_EPS)

        # mix3 row = pre(M) | post(M) | comb(M*M): the row stride is the
        # full width M*(2+M) = 24, not the 3*M group count
        tl.store(mix3_ptr + t * (M * (2 + M)) + offs_m, pre)
        tl.store(mix3_ptr + t * (M * (2 + M)) + M + offs_m, post)
        tl.store(
            mix3_ptr + t * (M * (2 + M)) + 2 * M + offs_m[:, None] * M + offs_m[None, :],
            comb,
        )

        # pass 2: combine with the previous boundary's pre
        if HAS_PRE:
            pp = tl.load(pre_ptr + t * PRE_STRIDE + offs_m)
        for h0 in range(0, H, BH):
            if HAS_PRE:
                xv2 = tl.load(
                    x_ptr + t * K + offs_m[:, None] * H + (h0 + offs)[None, :]
                ).to(tl.float32)
                yv = tl.sum(pp[:, None] * xv2, axis=0)
            else:
                # first boundary: no previous pre exists, copy copy 0
                yv = tl.load(x_ptr + t * K + h0 + offs).to(tl.float32)
            tl.store(y_ptr + t * H + h0 + offs, yv.to(y_ptr.dtype.element_ty))

    @triton.jit
    def dsv41_hc_post_kernel(
        x_ptr,        # [T, H] bf16/fp16 (sublayer output)
        res_ptr,      # [T, M, H] bf16/fp16, contiguous
        post_ptr,     # [T, M] fp32
        comb_ptr,     # [T, M, M] fp32
        out_ptr,      # [T, M, H] bf16/fp16 out
        POST_STRIDE,  # runtime row strides (post/comb may be views of mix3)
        COMB_S0,
        COMB_S1,
        H: tl.constexpr,
        M: tl.constexpr,
        BH: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.arange(0, BH)
        for h0 in range(0, H, BH):
            xv = tl.load(x_ptr + t * H + h0 + offs).to(tl.float32)
            for j in tl.static_range(M):
                pj = tl.load(post_ptr + t * POST_STRIDE + j)
                acc = pj * xv
                for k in tl.static_range(M):
                    ck = tl.load(comb_ptr + t * COMB_S0 + j * COMB_S1 + k)
                    rk = tl.load(res_ptr + t * M * H + k * H + h0 + offs).to(
                        tl.float32
                    )
                    acc += ck * rk
                tl.store(
                    out_ptr + t * M * H + j * H + h0 + offs,
                    acc.to(out_ptr.dtype.element_ty),
                )

_K = None


def _get_kernels():
    global _K
    if not _TRITON_IMPORTABLE:
        raise RuntimeError("triton-ascend not importable")
    if _K is None:
        _K = (dsv41_hc_mix_sinkhorn_combine_kernel, dsv41_hc_post_kernel)
    return _K


def dsv41_hc_mix_sinkhorn_combine(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_prev: torch.Tensor | None,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One hc pre-chain for the whole batch: mix GEMV + rms + gate/sinkhorn +
    combine. Returns ``(y [T, H], mix3 [T, 24])``; mix3 rows are
    ``pre(4) | post(4) | comb(16)`` in fp32.

    ``pre_prev`` is the previous boundary's pre [T, 4] fp32 (any row stride);
    ``None`` selects copy 0 (the first boundary), still emitting mix3.
    """
    if not triton_available():
        raise RuntimeError("triton-ascend not importable")
    (k1, _) = _get_kernels()
    T, K = x.shape
    hc_m = hc_fn.shape[0]
    if hc_m != MIX_HC:
        raise NotImplementedError(
            f"dsv41 hc triton kernels support hc_mult=4 (mix rows {MIX_HC}), "
            f"got {hc_m}"
        )
    m = 4
    h = K // m
    if hc_fn.shape[1] != K or not x.is_contiguous():
        raise ValueError(
            f"shape/contiguity mismatch: x {tuple(x.shape)}, hc_fn "
            f"{tuple(hc_fn.shape)}"
        )
    if hc_fn.dtype != torch.float32:
        raise ValueError(
            "hc_fn must be fp32 (cast once at init, not per call); got "
            f"{hc_fn.dtype}"
        )
    mix3 = torch.empty((T, MIX_HC), dtype=torch.float32, device=x.device)
    y = torch.empty((T, h), dtype=x.dtype, device=x.device)
    has_pre = pre_prev is not None
    if not has_pre:
        # dummy pointer, never dereferenced under HAS_PRE=False
        pre_prev = mix3
    k1[(T,)](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_prev,
        y,
        mix3,
        K,
        pre_prev.stride(0),
        H=h,
        M=m,
        SINKHORN_ITERS=sinkhorn_iters,
        RMS_EPS=rms_eps,
        HC_EPS=hc_eps,
        BH=256,
        HAS_PRE=has_pre,
        num_warps=8,
    )
    return y, mix3


def dsv41_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Residual-stream fold after a sublayer:
    ``out[t,j,:] = post[t,j]*x[t,:] + sum_k comb[t,j,k]*residual[t,k,:]``.
    ``post``/``comb`` may be strided views of a mix3 row (strides passed
    through). Returns a fresh contiguous [T, M, H] tensor of x's dtype.
    """
    if not triton_available():
        raise RuntimeError("triton-ascend not importable")
    (_, k2) = _get_kernels()
    T, H = x.shape
    m = residual.shape[1]
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous")
    out = torch.empty_like(residual)
    k2[(T,)](
        x,
        residual,
        post,
        comb,
        out,
        post.stride(0),
        comb.stride(0),
        comb.stride(1),
        H=H,
        M=m,
        BH=256,
        num_warps=8,
    )
    return out
