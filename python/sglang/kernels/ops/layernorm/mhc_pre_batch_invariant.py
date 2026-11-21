"""Batch-invariant mHC pre for deterministic inference on ROCm.

Same math as ``_mhc_pre_torch`` in ``mhc.py``: per token, the RMS of the
flattened (n, h) residual, the 2n + n*n mixes ``fn @ x * rsqrt``, sigmoid pre /
post weights, a softmax + Sinkhorn-normalised n x n combination, and the layer
input ``sum_j pre_j * residual_j``. AITER's ``mhc_pre`` chooses its split-K from
the token count, so a token's result depends on its batch; the Torch fallback is
invariant but slow.

Every token's reduction order here is fixed by constants and the weight shape
only, never by the token count:

1. ``_mix_partial_kernel``: rows in tiles of ``BM``; K = n*h is cut into
   ``NSPLIT`` fixed splits, each walked in order in ``BK`` steps, with an fp32 sum
   of squares and an fp32 dot of the row tile against fn^T.
2. ``_mix_finish_kernel``: per (token, h-block), the ``NSPLIT`` partials are
   summed in split order, then the activations, the Sinkhorn iterations and the
   layer input for that h-block. Every h-block program recomputes the small
   per-token math identically; only h-block 0 writes post and comb.

A row's partial sums never mix with other rows of its tile, so the tile height
does not change its bits: prefill batches use taller tiles, and the 16x16 MFMA
shape is pinned so both tile heights reduce K the same way.
"""

import torch
import triton
import triton.language as tl

# Measured on MI350X at GLM-5.3's shape (hc_mult 4, hidden 4096): 32 splits are
# the fastest invariant choice for decode; any fixed value keeps invariance.
_NSPLIT = 32
_BM = 16
# Prefill-sized batches use taller row tiles; the switch point is a speed
# choice only, since a row's bits do not depend on the tile height.
_BM_LARGE = 128
_BM_LARGE_MIN_TOKENS = 257
_BK = 128
_BH = 4096


@triton.jit
def _mix_partial_kernel(
    x_ptr,
    fn_ptr,
    pmix_ptr,
    psq_ptr,
    S,
    K,
    KS,
    stride_x,
    stride_fn,
    NMIX: tl.constexpr,
    NMIX_P: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < S
    cols = tl.arange(0, NMIX_P)
    cmask = cols < NMIX
    acc = tl.zeros((BM, NMIX_P), dtype=tl.float32)
    sq = tl.zeros((BM,), dtype=tl.float32)
    k_lo = split * KS
    for k0 in range(0, KS, BK):
        ks = k_lo + k0 + tl.arange(0, BK)
        kmask = (ks < K) & (k0 + tl.arange(0, BK) < KS)
        x = tl.load(
            x_ptr + rows[:, None].to(tl.int64) * stride_x + ks[None, :],
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        sq += tl.sum(x * x, axis=1)
        w = tl.load(
            fn_ptr + cols[None, :] * stride_fn + ks[:, None],
            mask=kmask[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x, w, input_precision="ieee")
    base = split.to(tl.int64) * S + rows.to(tl.int64)
    tl.store(
        pmix_ptr + base[:, None] * NMIX_P + cols[None, :], acc, mask=rmask[:, None]
    )
    tl.store(psq_ptr + base, sq, mask=rmask)


@triton.jit
def _mix_finish_kernel(
    x_ptr,
    pmix_ptr,
    psq_ptr,
    scale_ptr,
    hbase_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    S,
    K,
    H,
    stride_x,
    stride_out,
    rms_eps,
    pre_eps,
    sink_eps,
    post_mult,
    sink_repeat,
    HC: tl.constexpr,
    NMIX_P: tl.constexpr,
    NSPLIT: tl.constexpr,
    BH: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    i = tl.arange(0, HC)
    t64 = t.to(tl.int64)
    pre_raw = tl.zeros((HC,), dtype=tl.float32)
    post_raw = tl.zeros((HC,), dtype=tl.float32)
    comb_raw = tl.zeros((HC, HC), dtype=tl.float32)
    sq = 0.0
    for s in tl.static_range(NSPLIT):
        row = (s * S + t64) * NMIX_P
        pre_raw += tl.load(pmix_ptr + row + i)
        post_raw += tl.load(pmix_ptr + row + HC + i)
        comb_raw += tl.load(pmix_ptr + row + 2 * HC + i[:, None] * HC + i[None, :])
        sq += tl.load(psq_ptr + s * S + t64)
    r = 1.0 / tl.sqrt_rn(sq / K + rms_eps)
    s0 = tl.load(scale_ptr)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)
    pre = tl.sigmoid(pre_raw * r * s0 + tl.load(hbase_ptr + i)) + pre_eps
    post = post_mult * tl.sigmoid(post_raw * r * s1 + tl.load(hbase_ptr + HC + i))
    comb = comb_raw * r * s2 + tl.load(
        hbase_ptr + 2 * HC + i[:, None] * HC + i[None, :]
    )
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + sink_eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + sink_eps)
    for _ in range(sink_repeat - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + sink_eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + sink_eps)
    if hb == 0:
        tl.store(post_ptr + t64 * HC + i, post)
        tl.store(comb_ptr + t64 * HC * HC + i[:, None] * HC + i[None, :], comb)
    h = hb * BH + tl.arange(0, BH)
    hmask = h < H
    xr = tl.load(
        x_ptr + t64 * stride_x + i[:, None] * H + h[None, :],
        mask=hmask[None, :],
        other=0.0,
    ).to(tl.float32)
    li = tl.sum(pre[:, None] * xr, axis=0)
    tl.store(
        out_ptr + t64 * stride_out + h, li.to(out_ptr.dtype.element_ty), mask=hmask
    )


def mhc_pre_batch_invariant(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same signature and outputs as ``_mhc_pre_torch``: post [s, n, 1] fp32,
    comb [s, n, n] fp32 and layer_input [s, h] in the residual's dtype."""
    s, n, h = residual.shape
    k = n * h
    nmix = 2 * n + n * n
    assert fn.shape == (nmix, k), f"fn must be ({nmix}, {k}); got {tuple(fn.shape)}"
    assert sinkhorn_repeat >= 1
    device = residual.device
    post = torch.empty((s, n, 1), dtype=torch.float32, device=device)
    comb = torch.empty((s, n, n), dtype=torch.float32, device=device)
    out = torch.empty((s, h), dtype=residual.dtype, device=device)
    if s == 0:
        return post, comb, out
    x = residual.reshape(s, k)
    if x.stride(1) != 1:
        x = x.contiguous()
    fn = fn if fn.stride(1) == 1 else fn.contiguous()
    nmix_p = triton.next_power_of_2(nmix)
    split_len = triton.cdiv(triton.cdiv(k, _NSPLIT), _BK) * _BK
    pmix = torch.empty((_NSPLIT, s, nmix_p), dtype=torch.float32, device=device)
    psq = torch.empty((_NSPLIT, s), dtype=torch.float32, device=device)
    bm = _BM_LARGE if s >= _BM_LARGE_MIN_TOKENS else _BM
    _mix_partial_kernel[(triton.cdiv(s, bm), _NSPLIT)](
        x,
        fn,
        pmix,
        psq,
        s,
        k,
        split_len,
        x.stride(0),
        fn.stride(0),
        NMIX=nmix,
        NMIX_P=nmix_p,
        BM=bm,
        BK=_BK,
        num_warps=4,
        matrix_instr_nonkdim=16,
    )
    _mix_finish_kernel[(s, triton.cdiv(h, _BH))](
        x,
        pmix,
        psq,
        hc_scale.float().contiguous(),
        hc_base.float().contiguous(),
        post,
        comb,
        out,
        s,
        k,
        h,
        x.stride(0),
        out.stride(0),
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        int(sinkhorn_repeat),
        HC=n,
        NMIX_P=nmix_p,
        NSPLIT=_NSPLIT,
        BH=_BH,
        num_warps=4,
    )
    return post, comb, out
