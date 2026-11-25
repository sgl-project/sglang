"""Hy4-Flash iHC post + pre + RMSNorm, with the previous block's output RMSNorm
folded in at every token count. SM90 (H200-tuned), hc_mult 4, hidden 3072, bf16.

Math per token; bf16 roundings sit where the unfused path stores bf16:
    o         = bf16(out * rsqrt(mean(out^2) + out_eps) * w_out)
    new_res   = bf16(post[m] * o + residual[m])        (or residual, without output)
    s         = rsqrt(mean(new_res^2) + norm_eps);  mix = (new_res . hc_fn^T) * s
    pre       = sigmoid(mix[:4] * scale[0] + base[:4]) + hc_eps
    next_post = magnitude * sigmoid(mix[4:] * scale[1] + base[4:]) + hc_eps
    r         = bf16(sum_m pre[m] * new_res[m]);  y = bf16(r * rsqrt(mean(r^2) + eps) * w)

Small T (split-K, one launch): CTA (s, t) owns one hidden slice of all 4 streams
of token t. It computes the out-norm (re-reading the 6 KB `out` row), stores its
new_res slice and writes partial sumsq + 8 hc_fn dots. The last CTA of the token,
picked by an acq_rel ticket on a self-resetting counter, reduces the partials and
runs the gates, combine and RMSNorm over the whole row from L2.

Large T (2-3 launches): out-norm rsqrt per row, then stage 1 on tensor cores (CTA =
BT tokens x a KR-wide K range; hc_fn split into two bf16 terms in registers), then
stage 2 per row. Stage 1 prefetches every load one chunk ahead into registers and
runs with num_stages=1. Workaround for a Triton 3.8 pipelining race: with bf16 hc_fn
tiles loaded straight into tl.dot in a tl.range(num_stages=2 or 3) loop, the next
activation copy overwrote the shared buffer the previous async wgmma still read
(30/30 runs differed at T=7726). Re-check and drop when upgrading past Triton 3.8.x.

Concurrency: the split-K tickets are one buffer per device. Two small-T calls in
flight at once on different streams of one device would corrupt each other; SGLang
runs one Hy4-Flash forward at a time, on one stream. The counter reset of call i is
ordered before call i+1 by the kernel boundary, so the split-K launch must not use
programmatic dependent launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch

HC_MULT = 4
HIDDEN_SIZE = 3072
# [mix_0..7 (pre 0..3, post 4..7), sumsq, pad]
_PART_STRIDE = 16

# Split-K one-launch path up to these T, tiled path above. Measured crossovers on
# H200 (L2 flushed): with output ~56 graph-replayed and ~100 eager, without ~32.
_SPLITK_MAX_T_OUT = 64
_SPLITK_MAX_T_NOOUT = 32
_SPLITK_WARPS = 4
# One ticket per token of a split-K call.
_TICKET_SLOTS = max(_SPLITK_MAX_T_OUT, _SPLITK_MAX_T_NOOUT)

# Tiled stage-1 configs (BT tokens, BK chunk, KR K-range per CTA, warps) by max T,
# tuned on H200: smaller T takes a shorter K range so the grid still fills the GPU.
# KR is capped at 512: the y mismatch vs an fp32 reference grows with the K range
# one tensor-core accumulator covers, and KR 2048 was only ~4% faster.
_TILED_CFGS_OUT = (
    (128, (32, 64, 128, 4)),
    (256, (64, 64, 128, 4)),
    (384, (32, 64, 256, 4)),
    (768, (64, 64, 256, 4)),
    (2048, (64, 64, 512, 4)),
    (None, (32, 64, 512, 4)),
)
_TILED_CFGS_NOOUT = (
    (128, (32, 64, 128, 4)),
    (256, (64, 64, 128, 4)),
    (768, (64, 64, 256, 4)),
    (None, (64, 64, 512, 4)),
)
# hc_fn as two bf16 terms: |hc_fn - (hi + lo)| <= 2^-16 |hc_fn|.
_DOT_NTERMS = 2
_STAGE2_WARPS = 8

_tickets_by_device: dict[tuple[str, int | None], torch.Tensor] = {}


@triton.jit
def _load_finish_params(
    scale_ptr,
    base_ptr,
    rmsw_ptr,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PS: tl.constexpr,
):
    j = tl.arange(0, PS)
    h = tl.arange(0, BLOCK_H)
    w_r = tl.load(rmsw_ptr + h, mask=h < HIDDEN, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + tl.where(j < HC, 0, 1))
    bs = tl.load(base_ptr + j, mask=j < 2 * HC, other=0.0)
    return w_r, sc, bs


@triton.jit
def _finish_row(
    t,
    src,
    part_ptr,
    w_r,
    sc,
    bs,
    y_ptr,
    next_post_ptr,
    norm_eps,
    hc_eps,
    rms_eps,
    magnitude,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PS: tl.constexpr,
    CG: tl.constexpr,
):
    """Token t: reduce the NSPLIT partials, compute the gates, then the combine
    and RMSNorm over the whole row `src` (4 streams at stride HIDDEN)."""
    K: tl.constexpr = HC * HIDDEN
    NG: tl.constexpr = 2 * HC
    j = tl.arange(0, PS)
    sp = tl.arange(0, NSPLIT_P2)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN
    part_offs = (t * NSPLIT + sp)[:, None] * PS + j[None, :]
    part_mask = (sp < NSPLIT)[:, None] & (j <= NG)[None, :]
    # Row loads first so their latency overlaps the gate math. CG bypasses L1:
    # the row and partials were written by other CTAs of this launch.
    if CG:
        x0 = tl.load(src + h, mask=h_mask, other=0.0, cache_modifier=".cg")
        x1 = tl.load(src + HIDDEN + h, mask=h_mask, other=0.0, cache_modifier=".cg")
        x2 = tl.load(src + 2 * HIDDEN + h, mask=h_mask, other=0.0, cache_modifier=".cg")
        x3 = tl.load(src + 3 * HIDDEN + h, mask=h_mask, other=0.0, cache_modifier=".cg")
        pv = tl.load(
            part_ptr + part_offs, mask=part_mask, other=0.0, cache_modifier=".cg"
        )
    else:
        x0 = tl.load(src + h, mask=h_mask, other=0.0)
        x1 = tl.load(src + HIDDEN + h, mask=h_mask, other=0.0)
        x2 = tl.load(src + 2 * HIDDEN + h, mask=h_mask, other=0.0)
        x3 = tl.load(src + 3 * HIDDEN + h, mask=h_mask, other=0.0)
        pv = tl.load(part_ptr + part_offs, mask=part_mask, other=0.0)
    gmask = j < NG
    is_pre = j < HC

    tot = tl.sum(pv, axis=0)  # [PS]: mix_0..7, sumsq
    ss_tot = tl.sum(tl.where(j == NG, tot, 0.0), axis=0)
    rs = tl.rsqrt(ss_tot / K + norm_eps)
    sig = tl.sigmoid(tot * rs * sc + bs)
    gate = tl.where(is_pre, sig + hc_eps, magnitude * sig + hc_eps)
    tl.store(next_post_ptr + t * HC + (j - HC), gate, mask=gmask & (j >= HC))
    p0 = tl.sum(tl.where(j == 0, gate, 0.0), axis=0)
    p1 = tl.sum(tl.where(j == 1, gate, 0.0), axis=0)
    p2 = tl.sum(tl.where(j == 2, gate, 0.0), axis=0)
    p3 = tl.sum(tl.where(j == 3, gate, 0.0), axis=0)

    acc = p0 * x0.to(tl.float32)
    acc = acc + p1 * x1.to(tl.float32)
    acc = acc + p2 * x2.to(tl.float32)
    acc = acc + p3 * x3.to(tl.float32)
    # The unfused path stores the iHC output in bf16 before the norm reads it.
    r = acc.to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(r * r, axis=0) / HIDDEN + rms_eps)
    tl.store(y_ptr + t * HIDDEN + h, (r * inv * w_r).to(tl.bfloat16), mask=h_mask)


@triton.jit
def _ihc_splitk_kernel(
    out_ptr,
    res_ptr,
    post_ptr,
    onw_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    rmsw_ptr,
    new_res_ptr,
    y_ptr,
    next_post_ptr,
    part_ptr,
    ticket_ptr,
    out_eps,
    norm_eps,
    hc_eps,
    rms_eps,
    magnitude,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BD: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_OUT: tl.constexpr,
    OUT_NORM: tl.constexpr,
    PS: tl.constexpr,
):
    K: tl.constexpr = HC * HIDDEN
    NG: tl.constexpr = 2 * HC
    s = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)

    m = tl.arange(0, HC)
    d = s * BD + tl.arange(0, BD)
    d_mask = d < HIDDEN
    offs = m[:, None] * HIDDEN + d[None, :]  # [HC, BD]
    mask2 = d_mask[None, :] & (m[:, None] < HC)
    res = tl.load(res_ptr + t * K + offs, mask=mask2, other=0.0).to(tl.float32)
    n = tl.arange(0, NG)
    wfn = tl.load(
        fn_ptr + n[:, None, None] * K + offs[None, :, :],
        mask=mask2[None, :, :],
        other=0.0,
    )

    if HAS_OUT:
        o = tl.load(out_ptr + t * HIDDEN + d, mask=d_mask, other=0.0).to(tl.float32)
        g = tl.load(post_ptr + t * HC + m)
        if OUT_NORM:
            h = tl.arange(0, BLOCK_H)
            row = tl.load(out_ptr + t * HIDDEN + h, mask=h < HIDDEN, other=0.0)
            row = row.to(tl.float32)
            w_o = tl.load(onw_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
            inv_o = tl.rsqrt(tl.sum(row * row, axis=0) / HIDDEN + out_eps)
            # The unfused out-norm stores bf16.
            o = (o * inv_o * w_o).to(tl.bfloat16).to(tl.float32)
        nr = (g[:, None] * o[None, :] + res).to(tl.bfloat16)
        tl.store(new_res_ptr + t * K + offs, nr, mask=mask2)
        x = nr.to(tl.float32)
    else:
        x = res

    ss = tl.sum(tl.sum(x * x, axis=1), axis=0)
    mix = tl.sum(tl.sum(wfn * x[None, :, :], axis=2), axis=1)  # [NG]
    pbase = part_ptr + (t * NSPLIT + s) * PS
    tl.store(pbase + n, mix)
    tl.store(pbase + NG, ss)

    # Loaded before the ticket so the last CTA does not wait on HBM for them.
    w_r, sc, bs = _load_finish_params(
        scale_ptr, base_ptr, rmsw_ptr, HIDDEN, HC, BLOCK_H, PS
    )

    # Publish this CTA's partials and new_res slice (CTA barrier, then a
    # gpu-scope release), then take a ticket; the last CTA of the token finishes.
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr + t, 1, sem="acq_rel", scope="gpu")
    if ticket == NSPLIT - 1:
        if HAS_OUT:
            src = new_res_ptr + t * K
        else:
            src = res_ptr + t * K
        _finish_row(
            t,
            src,
            part_ptr,
            w_r,
            sc,
            bs,
            y_ptr,
            next_post_ptr,
            norm_eps,
            hc_eps,
            rms_eps,
            magnitude,
            HIDDEN,
            HC,
            NSPLIT,
            NSPLIT_P2,
            BLOCK_H,
            PS,
            True,
        )
        # Self-reset: the buffer is zero again for the next call or graph replay.
        tl.store(ticket_ptr + t, 0)


@triton.jit(do_not_specialize=["T"])
def _out_inv_rms_kernel(
    out_ptr,
    inv_ptr,
    T,
    eps,
    HIDDEN: tl.constexpr,
    RB: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = (tl.program_id(0) * RB + tl.arange(0, RB)).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    x = tl.load(
        out_ptr + rows[:, None] * HIDDEN + h[None, :],
        mask=(rows < T)[:, None] & (h < HIDDEN)[None, :],
        other=0.0,
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / HIDDEN + eps)
    tl.store(inv_ptr + rows, inv, mask=rows < T)


@triton.jit
def _load_fn_chunk(fn_ptr, k0, kk, n, K: tl.constexpr, NG: tl.constexpr):
    """hc_fn [NG, K] fp32 -> the K-major [BK, 16] tile at k0; columns >= NG zero."""
    return tl.load(
        fn_ptr + n[None, :] * K + (k0 + kk)[:, None],
        mask=(n < NG)[None, :],
        other=0.0,
    )


@triton.jit
def _split_dot(x, w, acc, NTERMS: tl.constexpr):
    """acc += x @ w with fp32 w split into NTERMS bf16 terms; each tensor-core
    product of a bf16 term with the bf16 x is exact in the fp32 accumulator."""
    rest = w
    for i in tl.static_range(NTERMS):
        wb = rest.to(tl.bfloat16)
        acc = tl.dot(x, wb, acc)
        if i + 1 < NTERMS:
            rest = rest - wb.to(tl.float32)
    return acc


@triton.jit(do_not_specialize=["T"])
def _ihc_dot_partials_kernel(
    out_ptr,
    inv_o_ptr,
    res_ptr,
    post_ptr,
    onw_ptr,
    fn_ptr,
    new_res_ptr,
    part_ptr,
    T,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    KR: tl.constexpr,
    NSPLIT: tl.constexpr,
    HAS_OUT: tl.constexpr,
    OUT_NORM: tl.constexpr,
    PS: tl.constexpr,
    IDX64: tl.constexpr,
    NTERMS: tl.constexpr,
):
    """CTA (s, b): tokens [b*BT, b*BT+BT) x the flat K range [s*KR, s*KR+KR),
    in BK chunks that each lie in one stream. Loads are prefetched one chunk
    ahead by hand, outside Triton's pipeliner (see the module docstring)."""
    K: tl.constexpr = HC * HIDDEN
    NG: tl.constexpr = 2 * HC
    s = tl.program_id(0)
    b = tl.program_id(1)
    rows = b * BT + tl.arange(0, BT)
    if IDX64:
        rows = rows.to(tl.int64)
    r_mask = rows < T
    kk = tl.arange(0, BK)
    n = tl.arange(0, 16)
    mask = r_mask[:, None] & (kk < BK)[None, :]
    if OUT_NORM:
        inv_o = tl.load(inv_o_ptr + rows, mask=r_mask, other=0.0)[:, None]
    acc = tl.zeros((BT, 16), dtype=tl.float32)
    ss = tl.zeros((BT,), dtype=tl.float32)
    NCH: tl.constexpr = KR // BK
    k0 = s * KR
    w_nx = _load_fn_chunk(fn_ptr, k0, kk, n, K, NG)
    if HAS_OUT:
        stream = k0 // HIDDEN
        d0 = k0 - stream * HIDDEN
        o_nx = tl.load(
            out_ptr + rows[:, None] * HIDDEN + (d0 + kk)[None, :], mask=mask, other=0.0
        )
        r_nx = tl.load(
            res_ptr + rows[:, None] * K + (k0 + kk)[None, :], mask=mask, other=0.0
        )
        for c in range(NCH):
            k0 = s * KR + c * BK
            stream = k0 // HIDDEN
            d0 = k0 - stream * HIDDEN
            offs = rows[:, None] * K + (k0 + kk)[None, :]
            o_cur = o_nx
            r_cur = r_nx
            w = w_nx
            k1 = s * KR + tl.minimum(c + 1, NCH - 1) * BK
            s1 = k1 // HIDDEN
            o_nx = tl.load(
                out_ptr + rows[:, None] * HIDDEN + (k1 - s1 * HIDDEN + kk)[None, :],
                mask=mask,
                other=0.0,
            )
            r_nx = tl.load(
                res_ptr + rows[:, None] * K + (k1 + kk)[None, :], mask=mask, other=0.0
            )
            w_nx = _load_fn_chunk(fn_ptr, k1, kk, n, K, NG)
            o = o_cur.to(tl.float32)
            if OUT_NORM:
                w_o = tl.load(onw_ptr + d0 + kk).to(tl.float32)[None, :]
                o = (o * inv_o * w_o).to(tl.bfloat16).to(tl.float32)
            g = tl.load(post_ptr + rows * HC + stream, mask=r_mask, other=0.0)
            x = (g[:, None] * o + r_cur.to(tl.float32)).to(tl.bfloat16)
            tl.store(new_res_ptr + offs, x, mask=mask)
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
            acc = _split_dot(x, w, acc, NTERMS)
    else:
        x_nx = tl.load(
            res_ptr + rows[:, None] * K + (k0 + kk)[None, :], mask=mask, other=0.0
        )
        for c in range(NCH):
            x = x_nx
            w = w_nx
            k1 = s * KR + tl.minimum(c + 1, NCH - 1) * BK
            x_nx = tl.load(
                res_ptr + rows[:, None] * K + (k1 + kk)[None, :], mask=mask, other=0.0
            )
            w_nx = _load_fn_chunk(fn_ptr, k1, kk, n, K, NG)
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
            acc = _split_dot(x, w, acc, NTERMS)
    pbase = part_ptr + (rows * NSPLIT + s) * PS
    tl.store(pbase[:, None] + n[None, :], acc, mask=r_mask[:, None] & (n < NG)[None, :])
    tl.store(pbase + NG, ss, mask=r_mask)


@triton.jit
def _ihc_stage2_kernel(
    src_ptr,
    part_ptr,
    scale_ptr,
    base_ptr,
    rmsw_ptr,
    y_ptr,
    next_post_ptr,
    norm_eps,
    hc_eps,
    rms_eps,
    magnitude,
    HIDDEN: tl.constexpr,
    HC: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    w_r, sc, bs = _load_finish_params(
        scale_ptr, base_ptr, rmsw_ptr, HIDDEN, HC, BLOCK_H, PS
    )
    _finish_row(
        t,
        src_ptr + t * (HC * HIDDEN),
        part_ptr,
        w_r,
        sc,
        bs,
        y_ptr,
        next_post_ptr,
        norm_eps,
        hc_eps,
        rms_eps,
        magnitude,
        HIDDEN,
        HC,
        NSPLIT,
        NSPLIT_P2,
        BLOCK_H,
        PS,
        False,
    )


def hy4_ihc_single_pass_supported(
    *, hc_mult: int, hidden_size: int, dtype: torch.dtype
) -> bool:
    """Coverage: NVIDIA SM90 (the local GPU), hc_mult 4, hidden 3072, bf16
    activations. The tile tables were measured on H200 only; every other case
    takes the generic kernels."""
    return (
        hc_mult == HC_MULT
        and hidden_size == HIDDEN_SIZE
        and dtype == torch.bfloat16
        and current_arch().is_sm(9, 0)
    )


def _tickets(device: torch.device) -> torch.Tensor:
    index = torch.cuda.current_device() if device.index is None else device.index
    key = (device.type, index)
    tickets = _tickets_by_device.get(key)
    if tickets is None:
        # A zeroing during capture is only recorded into the graph, not executed:
        # eager calls and other graphs would read uninitialized tickets.
        assert not torch.cuda.is_current_stream_capturing(), (
            "hy4_ihc_single_pass: allocate the ticket buffer outside CUDA-graph "
            "capture (call prepare_hy4_ihc_single_pass at model init)"
        )
        tickets = torch.zeros(
            _TICKET_SLOTS, dtype=torch.int32, device=torch.device(device.type, index)
        )
        _tickets_by_device[key] = tickets
    return tickets


def _outputs(output, residual):
    T, hc, hidden = residual.shape
    y = torch.empty((T, hidden), dtype=residual.dtype, device=residual.device)
    next_post = torch.empty((T, hc), dtype=torch.float32, device=residual.device)
    new_res = residual if output is None else torch.empty_like(residual)
    return new_res, y, next_post


def _splitk(
    output,
    residual,
    post,
    hc_fn,
    hc_scale,
    hc_base,
    magnitude,
    norm_eps,
    hc_eps,
    rms_weight,
    rms_eps,
    out_norm_weight,
    out_norm_eps,
):
    T, hc, hidden = residual.shape
    has_out = output is not None
    new_res, y, next_post = _outputs(output, residual)
    bd = 128 if T <= 8 else 256
    nsplit = triton.cdiv(hidden, bd)
    part = torch.empty(
        (T, nsplit, _PART_STRIDE), dtype=torch.float32, device=residual.device
    )
    _ihc_splitk_kernel[(nsplit, T)](
        output if has_out else residual,
        residual,
        post if has_out else hc_scale,
        out_norm_weight if out_norm_weight is not None else rms_weight,
        hc_fn,
        hc_scale,
        hc_base,
        rms_weight,
        new_res,
        y,
        next_post,
        part,
        _tickets(residual.device),
        float(out_norm_eps),
        float(norm_eps),
        float(hc_eps),
        float(rms_eps),
        float(magnitude),
        HIDDEN=hidden,
        HC=hc,
        BD=bd,
        NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit),
        BLOCK_H=triton.next_power_of_2(hidden),
        HAS_OUT=has_out,
        OUT_NORM=has_out and out_norm_weight is not None,
        PS=_PART_STRIDE,
        num_warps=_SPLITK_WARPS,
        enable_fp_fusion=False,
    )
    return new_res, y, next_post


def _tiled_cfg(T: int, has_out: bool):
    for t_max, cfg in _TILED_CFGS_OUT if has_out else _TILED_CFGS_NOOUT:
        if t_max is None or T <= t_max:
            return cfg


def _tiled(
    output,
    residual,
    post,
    hc_fn,
    hc_scale,
    hc_base,
    magnitude,
    norm_eps,
    hc_eps,
    rms_weight,
    rms_eps,
    out_norm_weight,
    out_norm_eps,
    cfg,
):
    T, hc, hidden = residual.shape
    dev = residual.device
    k_total = hc * hidden
    has_out = output is not None
    out_norm = has_out and out_norm_weight is not None
    new_res, y, next_post = _outputs(output, residual)
    bt, bk, kr, warps = cfg
    nsplit = k_total // kr
    part = torch.empty((T, nsplit, _PART_STRIDE), dtype=torch.float32, device=dev)
    block_h = triton.next_power_of_2(hidden)
    if out_norm:
        inv_o = torch.empty((T,), dtype=torch.float32, device=dev)
        _out_inv_rms_kernel[(triton.cdiv(T, 4),)](
            output,
            inv_o,
            T,
            float(out_norm_eps),
            HIDDEN=hidden,
            RB=4,
            BLOCK_H=block_h,
            num_warps=8,
            enable_fp_fusion=False,
        )
    else:
        inv_o = hc_scale
    _ihc_dot_partials_kernel[(nsplit, triton.cdiv(T, bt))](
        output if has_out else residual,
        inv_o,
        residual,
        post if has_out else hc_scale,
        out_norm_weight if out_norm else rms_weight,
        hc_fn,
        new_res,
        part,
        T,
        HIDDEN=hidden,
        HC=hc,
        BT=bt,
        BK=bk,
        KR=kr,
        NSPLIT=nsplit,
        HAS_OUT=has_out,
        OUT_NORM=out_norm,
        PS=_PART_STRIDE,
        # int64 indexing measured faster with output; without it, only when needed.
        IDX64=has_out or T * k_total >= 2**31,
        NTERMS=_DOT_NTERMS,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    _ihc_stage2_kernel[(T,)](
        new_res,
        part,
        hc_scale,
        hc_base,
        rms_weight,
        y,
        next_post,
        float(norm_eps),
        float(hc_eps),
        float(rms_eps),
        float(magnitude),
        HIDDEN=hidden,
        HC=hc,
        NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit),
        BLOCK_H=block_h,
        PS=_PART_STRIDE,
        num_warps=_STAGE2_WARPS,
        enable_fp_fusion=False,
    )
    return new_res, y, next_post


def hy4_ihc_single_pass(
    output: torch.Tensor | None,
    residual: torch.Tensor,
    post: torch.Tensor | None,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor,
    rms_eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same contract as `hy4_ihc_post_pre_norm`, except that `out_norm_weight`
    is accepted at every T. Only for inputs `hy4_ihc_single_pass_supported`
    accepts. Returns (new residual, y, next post); without `output` the
    residual is returned as the new residual."""
    if out_norm_weight is not None and output is None:
        raise ValueError("the output RMSNorm needs the block output")
    residual = residual.contiguous()
    hc_fn = hc_fn.contiguous()
    T, hc, hidden = residual.shape
    assert hc == HC_MULT and hidden == HIDDEN_SIZE, residual.shape
    assert residual.dtype == torch.bfloat16, residual.dtype
    assert hc_fn.dtype == torch.float32 and hc_fn.shape == (2 * hc, hc * hidden)
    if output is not None:
        assert output.dtype == torch.bfloat16, output.dtype
        output = output.contiguous()
        post = post.contiguous()
    if T == 0:
        return _outputs(output, residual)
    kwargs = dict(
        output=output,
        residual=residual,
        post=post,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        magnitude=magnitude,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        rms_weight=rms_weight,
        rms_eps=rms_eps,
        out_norm_weight=out_norm_weight,
        out_norm_eps=out_norm_eps,
    )
    splitk_max_t = _SPLITK_MAX_T_NOOUT if output is None else _SPLITK_MAX_T_OUT
    if T <= splitk_max_t:
        return _splitk(**kwargs)
    return _tiled(**kwargs, cfg=_tiled_cfg(T, output is not None))


def prepare_hy4_ihc_single_pass(
    *, device: torch.device, out_norm_variants: tuple[bool, ...]
) -> None:
    """Allocate the split-K ticket buffer and compile every kernel variant the
    dispatcher can pick, so serving never compiles. Call eagerly, before any
    CUDA-graph capture, once per device. `out_norm_variants`: whether calls
    with an output pass `out_norm_weight` (True), omit it (False), or both."""
    _tickets(device)
    # Explicit dtypes: model init runs under a bf16 default dtype, and a pointer's
    # element type is part of the compiled variant.
    f32 = dict(dtype=torch.float32, device=device)
    weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    params = dict(
        hc_fn=torch.zeros((2 * HC_MULT, HC_MULT * HIDDEN_SIZE), **f32),
        hc_scale=torch.ones(2, **f32),
        hc_base=torch.zeros(2 * HC_MULT, **f32),
        magnitude=2.0,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rms_weight=weight,
        rms_eps=1e-6,
        out_norm_eps=1e-6,
    )

    # T is not specialized, so a few tokens compile the same kernels as any T of
    # the same variant: the two split-K block sizes and each tiled config.
    variants = [dict(has_out=False, out_norm_weight=None)] + [
        dict(has_out=True, out_norm_weight=weight if norm else None)
        for norm in out_norm_variants
    ]
    for variant in variants:
        cfgs = _TILED_CFGS_OUT if variant["has_out"] else _TILED_CFGS_NOOUT
        for T in (1, 9):
            _splitk(**_warmup_inputs(T, variant, device), **params)
        for _, cfg in cfgs:
            _tiled(**_warmup_inputs(1, variant, device), **params, cfg=cfg)
    torch.cuda.synchronize(device)


def _warmup_inputs(T: int, variant: dict, device: torch.device) -> dict:
    residual = torch.ones(
        (T, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
    )
    if not variant["has_out"]:
        return dict(output=None, residual=residual, post=None, out_norm_weight=None)
    return dict(
        output=torch.ones((T, HIDDEN_SIZE), dtype=torch.bfloat16, device=device),
        residual=residual,
        post=torch.ones((T, HC_MULT), dtype=torch.float32, device=device),
        out_norm_weight=variant["out_norm_weight"],
    )
