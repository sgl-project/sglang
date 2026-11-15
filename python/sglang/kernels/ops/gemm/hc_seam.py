"""Fused decode HC seam for qwen4_exp gated hyper-connections (Qwen3.8-Flash-Next: HC 4 x H 2560, low rank 320).

One seam is the work between the TP all-reduce of a block output ``y`` and the next block's GEMM input:

    a     = gate of the previous stage, 2 * sigmoid(xn_prev . W_inject_prev / HC)          (fp32 [M, HC])
    R'    = bf16(R + a[c] * y)                                       (combine; skipped for a mix-only seam)
    xn    = bf16(R' * rsqrt(mean_branch(R'^2) + eps) * (1 + w_norm))                       (per-branch norm)
    t     = bf16(silu(xn . W_down^T / HC))                                                  ([M, 320])
    mixed = bf16(mean_c(sigmoid(t . W_up[c]^T) * xn[c]))                                   ([M, H])
    a'    = 2 * sigmoid(xn . W_inject^T / HC)                         (gate the next combine will use)

Three launches, no atomics and no inter-CTA waits (deterministic, CUDA-graph safe):
  K_a  split-K over the 10240-wide row: combine, branch sum of squares, and the norm-deferred down-projection
       partials (the per-branch rsqrt is a scalar per (row, branch), so it is applied after the dot).
  K_f  fixed-order reduction of the partials: rsqrt and t, one output per thread; plus programs that compute a' from
       the exact bf16 xn, as the stock combine does.
  K_b  up projection of all HC branches in one dot pair, sigmoid, exact bf16 xn, gated mean.
``hc_gated_combine`` (K_c) is the combine alone, for a boundary whose mix runs later.
Each kernel takes M before its pointers, inside the kernel-argument registers gfx950 preloads; after the pointers it
fell past that range and cost a scalar load before the first vector load. K_f's eps stays last: its first use comes
after the partials' loads.

K_a and K_b tile rows on their slowest grid axis: up to _ROW_TILE rows a call runs one BM = next_pow2(rows) tile,
above that tiles of _ROW_TILE rows, so every row goes through the arithmetic of a single-tile call.
The per-device workspace is reused by every seam; seams on one stream run in order, so it needs no clearing.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

HC = 4
HS = 2560
LOWRANK = 320
MAX_ROWS = 256  # rows per call (workspace size)
_ROW_TILE = 64

# Fixed launch configuration, chosen by tile sweeps on gfx950. K_b reads each operand in a few wide loads; chains of
# small loads measured slower.
KA_KB = 256  # row chunk per K_a program (divides HS)
KA_BK = 64
KA_BN = 64
KA_WARPS = 4
KF_BLOCK = 256  # outputs of t per K_f reduce program, one per thread
KF_BKG = 512
KF_ROW_GATES = 129  # from these rows one K_f program computes all of a row's gates
_XCDS = 8  # gfx950 (SPX mode) dispatches workgroups round-robin over its 8 XCDs by linear id
KB_BJ = 16
KB_LA = 256  # LOWRANK = KB_LA + (LOWRANK - KB_LA); both parts powers of two
KB_WARPS = 4
KB_WARPS_BM32 = 2  # four warps would each run the whole 32-row MFMA tile
KC_BLOCK = 2048


@triton.jit
def _hc_combine_apply_kernel(
    y_ptr,
    r_ptr,
    a_ptr,
    out_ptr,
    total,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    D: tl.constexpr = HC_ * HS_
    m = offs // D
    k = offs % D
    c = k // HS_
    i = k % HS_
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + m * HS_ + i, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + m * HC_ + c, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.fma(a, y, r).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _hc_seam_down_kernel(
    M,
    y_ptr,
    r_ptr,
    a_ptr,
    rout_ptr,
    wn_ptr,
    wd_ptr,
    p_ptr,
    ss_ptr,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    L: tl.constexpr,
    BM: tl.constexpr,
    KB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    HAS_COMBINE: tl.constexpr,
):
    D: tl.constexpr = HC_ * HS_
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    k_base = pid_k * KB
    c = k_base // HS_
    offs_m = tl.program_id(2) * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < L
    first_n = pid_n == 0
    if HAS_COMBINE:
        a_m = tl.load(a_ptr + offs_m * HC_ + c, mask=mask_m, other=0.0)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    ss = tl.zeros((BM,), dtype=tl.float32)
    for kk in range(0, KB, BK):
        k = k_base + kk + tl.arange(0, BK)
        r = tl.load(
            r_ptr + offs_m[:, None] * D + k[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        if HAS_COMBINE:
            y = tl.load(
                y_ptr + offs_m[:, None] * HS_ + (k - c * HS_)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            rn = tl.fma(a_m[:, None], y, r).to(rout_ptr.dtype.element_ty)
            tl.store(
                rout_ptr + offs_m[:, None] * D + k[None, :],
                rn,
                mask=mask_m[:, None] & first_n,
            )
            r = rn.to(tl.float32)
        ss += tl.sum(r * r, axis=1)
        wn = tl.load(wn_ptr + k).to(tl.float32)
        z = (r * (1.0 + wn)[None, :]).to(wd_ptr.dtype.element_ty)
        wd = tl.load(
            wd_ptr + offs_n[:, None] * D + k[None, :], mask=mask_n[:, None], other=0.0
        )
        acc = tl.dot(z, tl.trans(wd), acc)

    tl.store(
        p_ptr + pid_k * M * L + offs_m[:, None] * L + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(ss_ptr + pid_k * M + offs_m, ss, mask=mask_m & first_n)


@triton.jit
def _gate_rs(
    ss_ptr, M, m, c, eps, HS_: tl.constexpr, NKC_B: tl.constexpr, NKP: tl.constexpr
):
    """rsqrt(mean square) of branch c of row m: its split-K sums of squares, one masked load summed as a tree."""
    offs_j = tl.arange(0, NKP)
    ssb = tl.sum(
        tl.load(ss_ptr + (c * NKC_B + offs_j) * M + m, mask=offs_j < NKC_B, other=0.0),
        axis=0,
    )
    return tl.math.rsqrt(ssb / HS_ + eps)


@triton.jit
def _gate_one(
    M,
    ss_ptr,
    r_ptr,
    wn_ptr,
    wi_ptr,
    aout_ptr,
    eps,
    m,
    cp,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    NKC_B: tl.constexpr,
    NKP: tl.constexpr,
    BKG: tl.constexpr,
    NQP: tl.constexpr,
):
    """Gate cp of row m: the row's exact bf16 xn dotted with W_inject row cp."""
    D: tl.constexpr = HC_ * HS_
    offs_q = tl.arange(0, NQP)
    mask_q = (offs_q < HS_ // BKG)[:, None]
    offs_k = tl.arange(0, BKG)
    accv = tl.zeros((NQP, BKG), dtype=tl.float32)
    for c in tl.static_range(HC_):
        rs = _gate_rs(ss_ptr, M, m, c, eps, HS_, NKC_B, NKP)
        k = c * HS_ + offs_q[:, None] * BKG + offs_k[None, :]
        xr = tl.load(r_ptr + m * D + k, mask=mask_q, other=0.0).to(tl.float32)
        wn = tl.load(wn_ptr + k, mask=mask_q, other=0.0).to(tl.float32)
        xn = ((xr * rs) * (1.0 + wn)).to(r_ptr.dtype.element_ty).to(tl.float32)
        wi = tl.load(wi_ptr + cp * D + k, mask=mask_q, other=0.0).to(tl.float32)
        accv += xn * wi
    g = tl.sum(tl.sum(accv, axis=1), axis=0)
    tl.store(aout_ptr + m * HC_ + cp, 2.0 / (1.0 + tl.exp(-g * (1.0 / HC_))))


@triton.jit
def _gate_row(
    M,
    ss_ptr,
    r_ptr,
    wn_ptr,
    wi_ptr,
    aout_ptr,
    eps,
    m,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    NKC_B: tl.constexpr,
    NKP: tl.constexpr,
    BKG: tl.constexpr,
):
    """All four gates of row m: its exact bf16 xn, built once in [BKG] chunks, with one [BKG] accumulator per gate in
    xn's layout (an [HC, BKG] W_inject tile was 4.5 us slower per seam at 129-193 rows on gfx950)."""
    tl.static_assert(HC_ == 4)
    D: tl.constexpr = HC_ * HS_
    offs_k = tl.arange(0, BKG)
    acc0 = tl.zeros((BKG,), dtype=tl.float32)
    acc1 = tl.zeros((BKG,), dtype=tl.float32)
    acc2 = tl.zeros((BKG,), dtype=tl.float32)
    acc3 = tl.zeros((BKG,), dtype=tl.float32)
    for c in tl.static_range(HC_):
        rs = _gate_rs(ss_ptr, M, m, c, eps, HS_, NKC_B, NKP)
        for q in range(HS_ // BKG):
            k = c * HS_ + q * BKG + offs_k
            xr = tl.load(r_ptr + m * D + k).to(tl.float32)
            wn = tl.load(wn_ptr + k).to(tl.float32)
            xn = ((xr * rs) * (1.0 + wn)).to(r_ptr.dtype.element_ty).to(tl.float32)
            acc0 += xn * tl.load(wi_ptr + k).to(tl.float32)
            acc1 += xn * tl.load(wi_ptr + D + k).to(tl.float32)
            acc2 += xn * tl.load(wi_ptr + 2 * D + k).to(tl.float32)
            acc3 += xn * tl.load(wi_ptr + 3 * D + k).to(tl.float32)
    tl.store(
        aout_ptr + m * HC_ + 0,
        2.0 / (1.0 + tl.exp(-tl.sum(acc0, axis=0) * (1.0 / HC_))),
    )
    tl.store(
        aout_ptr + m * HC_ + 1,
        2.0 / (1.0 + tl.exp(-tl.sum(acc1, axis=0) * (1.0 / HC_))),
    )
    tl.store(
        aout_ptr + m * HC_ + 2,
        2.0 / (1.0 + tl.exp(-tl.sum(acc2, axis=0) * (1.0 / HC_))),
    )
    tl.store(
        aout_ptr + m * HC_ + 3,
        2.0 / (1.0 + tl.exp(-tl.sum(acc3, axis=0) * (1.0 / HC_))),
    )


@triton.jit
def _hc_seam_reduce_kernel(
    M,
    p_ptr,
    ss_ptr,
    r_ptr,
    wn_ptr,
    wi_ptr,
    t_ptr,
    rs_ptr,
    aout_ptr,
    eps,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    L: tl.constexpr,
    KB: tl.constexpr,
    NKP: tl.constexpr,
    BLOCK: tl.constexpr,
    BKG: tl.constexpr,
    NQP: tl.constexpr,
    XCDS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ROW_GATES: tl.constexpr,
):
    """The first programs each own BLOCK outputs of t, one per thread: every branch's split-K sums of squares and
    partials are added in chunk order, so all loads are independent of the sums. With HAS_GATE, gate programs
    follow: one per (row, gate) in blocks of XCDS rows, so a row's programs share an XCD and read its R' into one
    L2; with ROW_GATES one per row for all HC gates, which reads R' once."""
    NKC_B: tl.constexpr = HS_ // KB  # split-K chunks per branch (<= NKP)
    pid = tl.program_id(0)
    n_red = (M * L + BLOCK - 1) // BLOCK
    if pid < n_red:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M * L
        m = offs // L
        t_raw = tl.zeros((BLOCK,), dtype=tl.float32)
        for c in tl.static_range(HC_):
            ssb = tl.load(ss_ptr + c * NKC_B * M + m, mask=mask, other=0.0)
            u = tl.load(p_ptr + c * NKC_B * (M * L) + offs, mask=mask, other=0.0)
            for j in tl.static_range(1, NKC_B):
                kc = c * NKC_B + j
                ssb += tl.load(ss_ptr + kc * M + m, mask=mask, other=0.0)
                u += tl.load(p_ptr + kc * (M * L) + offs, mask=mask, other=0.0)
            rs = tl.math.rsqrt(ssb / HS_ + eps)
            tl.store(rs_ptr + m * HC_ + c, rs, mask=mask & (offs % L == 0))
            t_raw += rs * u
        a = t_raw * (1.0 / HC_)
        t = a * tl.sigmoid(a)
        tl.store(t_ptr + offs, t.to(t_ptr.dtype.element_ty), mask=mask)
    elif HAS_GATE:
        g = pid - n_red
        if ROW_GATES:
            _gate_row(
                M,
                ss_ptr,
                r_ptr,
                wn_ptr,
                wi_ptr,
                aout_ptr,
                eps,
                g,
                HC_,
                HS_,
                NKC_B,
                NKP,
                BKG,
            )
        else:
            row = g // (XCDS * HC_) * XCDS + g % XCDS
            if row < M:
                _gate_one(
                    M,
                    ss_ptr,
                    r_ptr,
                    wn_ptr,
                    wi_ptr,
                    aout_ptr,
                    eps,
                    row,
                    g % (XCDS * HC_) // XCDS,
                    HC_,
                    HS_,
                    NKC_B,
                    NKP,
                    BKG,
                    NQP,
                )


@triton.jit
def _hc_seam_up_kernel(
    M,
    t_ptr,
    rs_ptr,
    r_ptr,
    wn_ptr,
    wu_ptr,
    out_ptr,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    L: tl.constexpr,
    BM: tl.constexpr,
    BJ: tl.constexpr,
    LA: tl.constexpr,
    LB: tl.constexpr,
):
    """A program owns BJ output columns: the W_up rows of all HC branches for them ([HC*BJ, L], loaded at once and
    transposed) go through one dot pair with t ([BM, LA] + [BM, LB], LA + LB = L); the gated mean sums the branch
    axis."""
    tl.static_assert(LA + LB == L)
    D: tl.constexpr = HC_ * HS_
    N: tl.constexpr = HC_ * BJ
    pid = tl.program_id(0)
    offs_m = tl.program_id(1) * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    j = pid * BJ + tl.arange(0, BJ)
    offs_c = tl.arange(0, HC_)
    cols = tl.reshape(
        offs_c[:, None] * HS_ + j[None, :], (N,)
    )  # branch-major: [c0 j.., c1 j.., ...]
    col_branch = tl.arange(0, N) // BJ
    ra = tl.arange(0, LA)
    rb = LA + tl.arange(0, LB)
    t_a = tl.load(
        t_ptr + offs_m[:, None] * L + ra[None, :], mask=mask_m[:, None], other=0.0
    )
    t_b = tl.load(
        t_ptr + offs_m[:, None] * L + rb[None, :], mask=mask_m[:, None], other=0.0
    )
    if BM <= 32:
        # Loaded as W_up^T ([LA, N], [LB, N]), the dots' B operands: a row-major load needs a layout conversion
        # through LDS before tl.trans, emitted right after the load, so each part's load waited for the previous
        # one's. The 64-row tile keeps the row-major load: transposed, it takes more LDS traffic than it saves.
        wu_a = tl.load(wu_ptr + cols[None, :] * L + ra[:, None])
        wu_b = tl.load(wu_ptr + cols[None, :] * L + rb[:, None])
    else:
        wu_a = tl.trans(tl.load(wu_ptr + cols[:, None] * L + ra[None, :]))
        wu_b = tl.trans(tl.load(wu_ptr + cols[:, None] * L + rb[None, :]))
    xr = tl.load(
        r_ptr + offs_m[:, None] * D + cols[None, :], mask=mask_m[:, None], other=0.0
    ).to(tl.float32)
    rs = tl.load(
        rs_ptr + offs_m[:, None] * HC_ + col_branch[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    wn = tl.load(wn_ptr + cols).to(tl.float32)
    acc = tl.dot(t_a, wu_a)
    acc = tl.dot(t_b, wu_b, acc)
    xn = ((xr * rs) * (1.0 + wn)[None, :]).to(out_ptr.dtype.element_ty).to(tl.float32)
    prod = tl.reshape(tl.sigmoid(acc) * xn, (BM, HC_, BJ))
    out = tl.sum(prod, axis=1) * (1.0 / HC_)
    tl.store(
        out_ptr + offs_m[:, None] * HS_ + j[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


_workspaces: dict = {}


def _workspace(device: torch.device):
    # Sized for MAX_ROWS at the first call of any width, so no later call allocates it. SGLang's CUDA graph
    # runners run each shape eagerly before capturing it, so that first call is not inside a capture.
    ws = _workspaces.get(device)
    if ws is None:
        nkc = HC * HS // KA_KB
        ws = dict(
            p=torch.empty(
                (nkc * MAX_ROWS * LOWRANK,), dtype=torch.float32, device=device
            ),
            ss=torch.empty((nkc * MAX_ROWS,), dtype=torch.float32, device=device),
            t=torch.empty((MAX_ROWS * LOWRANK,), dtype=torch.bfloat16, device=device),
            rs=torch.empty((MAX_ROWS * HC,), dtype=torch.float32, device=device),
        )
        _workspaces[device] = ws
    return ws


def hc_seam_supported(rows: int, x: torch.Tensor) -> bool:
    return (
        0 < rows <= MAX_ROWS
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == HC * HS
        and x.is_contiguous()
    )


def hc_gated_combine(
    y: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """R' = bf16(R + gate[c] * y) with a precomputed fp32 gate [M, HC]."""
    rows = residual.shape[0]
    assert y.shape == (rows, HS) and gate.shape == (rows, HC)
    out = torch.empty_like(residual)
    total = residual.numel()
    _hc_combine_apply_kernel[(triton.cdiv(total, KC_BLOCK),)](
        y, residual, gate, out, total, HC_=HC, HS_=HS, BLOCK=KC_BLOCK, num_warps=4
    )
    return out


def hc_seam(
    y: Optional[torch.Tensor],
    residual: torch.Tensor,
    gate_prev: Optional[torch.Tensor],
    norm_w: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject_next: Optional[torch.Tensor],
    eps: float,
):
    """Combine (when ``y`` is given) + norm + low-rank gated mix (+ the next combine's gate).

    Returns (residual_new, mixed [M, HS] bf16, gate_new fp32 [M, HC] or None). Inputs must be contiguous bf16;
    ``gate_prev`` fp32 [M, HC]. At most MAX_ROWS rows.
    """
    rows = residual.shape[0]
    if rows > MAX_ROWS:
        raise ValueError(f"hc_seam: {rows} rows, the workspace holds 1..{MAX_ROWS}")
    if y is not None:
        assert y.shape == (rows, HS) and gate_prev.shape == (rows, HC)
    bm = max(16, triton.next_power_of_2(rows)) if rows <= _ROW_TILE else _ROW_TILE
    row_tiles = triton.cdiv(rows, bm)
    device = residual.device
    ws = _workspace(device)
    has_combine = y is not None
    has_gate = w_inject_next is not None
    r_new = torch.empty_like(residual) if has_combine else residual
    mixed = torch.empty((rows, HS), dtype=residual.dtype, device=device)
    gate_new = (
        torch.empty((rows, HC), dtype=torch.float32, device=device)
        if has_gate
        else None
    )
    nkc = HC * HS // KA_KB
    _hc_seam_down_kernel[(nkc, triton.cdiv(LOWRANK, KA_BN), row_tiles)](
        rows,
        y if has_combine else residual,
        residual,
        gate_prev if has_combine else ws["rs"],
        r_new,
        norm_w,
        w_down,
        ws["p"],
        ws["ss"],
        HC_=HC,
        HS_=HS,
        L=LOWRANK,
        BM=bm,
        KB=KA_KB,
        BK=KA_BK,
        BN=KA_BN,
        HAS_COMBINE=has_combine,
        num_warps=KA_WARPS,
    )
    # Gate programs: blocks of _XCDS rows x HC gates (a row's programs _XCDS ids apart), or one per row.
    row_gates = has_gate and rows >= KF_ROW_GATES
    if row_gates:
        n_gate = rows
    elif has_gate:
        n_gate = triton.cdiv(rows, _XCDS) * _XCDS * HC
    else:
        n_gate = 0
    _hc_seam_reduce_kernel[(triton.cdiv(rows * LOWRANK, KF_BLOCK) + n_gate,)](
        rows,
        ws["p"],
        ws["ss"],
        r_new,
        norm_w,
        w_inject_next if has_gate else norm_w,
        ws["t"],
        ws["rs"],
        gate_new if has_gate else ws["rs"],
        eps,
        HC_=HC,
        HS_=HS,
        L=LOWRANK,
        KB=KA_KB,
        NKP=triton.next_power_of_2(HS // KA_KB),
        BLOCK=KF_BLOCK,
        BKG=KF_BKG,
        NQP=triton.next_power_of_2(HS // KF_BKG),
        XCDS=_XCDS,
        HAS_GATE=has_gate,
        ROW_GATES=row_gates,
        num_warps=4,
    )
    _hc_seam_up_kernel[(HS // KB_BJ, row_tiles)](
        rows,
        ws["t"],
        ws["rs"],
        r_new,
        norm_w,
        w_up,
        mixed,
        HC_=HC,
        HS_=HS,
        L=LOWRANK,
        BM=bm,
        BJ=KB_BJ,
        LA=KB_LA,
        LB=LOWRANK - KB_LA,
        num_warps=KB_WARPS_BM32 if bm == 32 else KB_WARPS,
    )
    return r_new, mixed, gate_new
