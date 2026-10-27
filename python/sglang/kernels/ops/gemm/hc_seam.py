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
  K_f  fixed-order (tree) reduction of the partials: rsqrt and t; plus one program per (row, branch) that computes
       that entry of a' from the exact bf16 xn, as the stock combine does.
  K_b  up projection of all HC branches in one dot pair, sigmoid, exact bf16 xn, gated mean.
``hc_gated_combine`` (K_c) is the combine alone, for a boundary whose mix runs later.

K_a and K_b tile rows on their slowest grid axis: up to _ROW_TILE rows a call runs one BM = next_pow2(rows) tile,
above that tiles of _ROW_TILE rows, so every row goes through the arithmetic of a single-tile call.
The per-device workspace is reused by every seam; seams on one stream run in order, so it needs no clearing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HC = 4
HS = 2560
LOWRANK = 320
MAX_ROWS = 256  # rows per call (workspace size)
_ROW_TILE = 64

# Fixed launch configuration, chosen by a tile sweep on gfx950. K_f and K_b read each operand in a few wide loads;
# chains of small loads measured slower.
KA_KB = 256  # row chunk per K_a program (divides HS)
KA_BK = 64
KA_BN = 64
KA_WARPS = 4
KF_BM = 8
KF_BL = 16
KF_BKG = 512
KB_BJ = 16
KB_LA = 256  # LOWRANK = KB_LA + (LOWRANK - KB_LA); both parts powers of two
KB_WARPS = 4
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
    y_ptr,
    r_ptr,
    a_ptr,
    rout_ptr,
    wn_ptr,
    wd_ptr,
    p_ptr,
    ss_ptr,
    M,
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
def _hc_seam_reduce_kernel(
    p_ptr,
    ss_ptr,
    r_ptr,
    wn_ptr,
    wi_ptr,
    t_ptr,
    rs_ptr,
    aout_ptr,
    M,
    eps,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    L: tl.constexpr,
    KB: tl.constexpr,
    BM: tl.constexpr,
    BL: tl.constexpr,
    BKG: tl.constexpr,
    NKP: tl.constexpr,
    NQP: tl.constexpr,
):
    """Programs (pid_m, pid_l < NLT) reduce the down-projection partials of a [BM, BL] tile; programs with
    pid_l >= NLT (gate grid only) each compute one of a row's next combine gates from the exact bf16 xn.
    Each branch's split-K chunks are read in one masked load and summed as a tree, so a program issues a few
    independent loads instead of a chain of small ones."""
    D: tl.constexpr = HC_ * HS_
    NKC_B: tl.constexpr = HS_ // KB  # split-K chunks per branch (<= NKP)
    NQ: tl.constexpr = HS_ // BKG  # gate chunks per branch (<= NQP)
    NLT: tl.constexpr = (L + BL - 1) // BL
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)
    offs_j = tl.arange(0, NKP)
    mask_j = offs_j < NKC_B
    if pid_l < NLT:
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_m = offs_m < M
        offs_l = pid_l * BL + tl.arange(0, BL)
        mask_l = offs_l < L
        first_l = pid_l == 0
        t_raw = tl.zeros((BM, BL), dtype=tl.float32)
        for c in tl.static_range(HC_):
            kc = c * NKC_B + offs_j
            ssb = tl.sum(
                tl.load(
                    ss_ptr + kc[:, None] * M + offs_m[None, :],
                    mask=mask_j[:, None] & mask_m[None, :],
                    other=0.0,
                ),
                axis=0,
            )
            ub = tl.sum(
                tl.load(
                    p_ptr
                    + kc[:, None, None] * (M * L)
                    + offs_m[None, :, None] * L
                    + offs_l[None, None, :],
                    mask=mask_j[:, None, None]
                    & mask_m[None, :, None]
                    & mask_l[None, None, :],
                    other=0.0,
                ),
                axis=0,
            )
            rs = tl.math.rsqrt(ssb / HS_ + eps)
            tl.store(rs_ptr + offs_m * HC_ + c, rs, mask=mask_m & first_l)
            t_raw += rs[:, None] * ub
        a = t_raw * (1.0 / HC_)
        t = a * tl.sigmoid(a)
        tl.store(
            t_ptr + offs_m[:, None] * L + offs_l[None, :],
            t.to(t_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_l[None, :],
        )
    else:
        # One program per (row, output gate): the row's exact bf16 xn dotted with one W_inject row.
        q = pid_l - NLT
        m = pid_m * BM + q // HC_
        cp = q % HC_
        if m < M:
            offs_q = tl.arange(0, NQP)
            mask_q = offs_q < NQ
            offs_k = tl.arange(0, BKG)
            accv = tl.zeros((NQP, BKG), dtype=tl.float32)
            for c in tl.static_range(HC_):
                ssb = tl.sum(
                    tl.load(
                        ss_ptr + (c * NKC_B + offs_j) * M + m, mask=mask_j, other=0.0
                    ),
                    axis=0,
                )
                rs = tl.math.rsqrt(ssb / HS_ + eps)
                k = c * HS_ + offs_q[:, None] * BKG + offs_k[None, :]
                xr = tl.load(r_ptr + m * D + k, mask=mask_q[:, None], other=0.0).to(
                    tl.float32
                )
                wn = tl.load(wn_ptr + k, mask=mask_q[:, None], other=0.0).to(tl.float32)
                xn = ((xr * rs) * (1.0 + wn)).to(r_ptr.dtype.element_ty).to(tl.float32)
                wi = tl.load(wi_ptr + cp * D + k, mask=mask_q[:, None], other=0.0).to(
                    tl.float32
                )
                accv += xn * wi
            g = tl.sum(tl.sum(accv, axis=1), axis=0)
            tl.store(aout_ptr + m * HC_ + cp, 2.0 / (1.0 + tl.exp(-g * (1.0 / HC_))))


@triton.jit
def _hc_seam_up_kernel(
    t_ptr,
    rs_ptr,
    r_ptr,
    wn_ptr,
    wu_ptr,
    out_ptr,
    M,
    HC_: tl.constexpr,
    HS_: tl.constexpr,
    L: tl.constexpr,
    BM: tl.constexpr,
    BJ: tl.constexpr,
    LA: tl.constexpr,
    LB: tl.constexpr,
):
    """A program owns BJ output columns: the W_up rows of all HC branches for them ([HC*BJ, L], loaded at once) go
    through one dot pair with t ([BM, LA] + [BM, LB], LA + LB = L); the gated mean sums the branch axis."""
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
    wu_a = tl.load(wu_ptr + cols[:, None] * L + ra[None, :])
    wu_b = tl.load(wu_ptr + cols[:, None] * L + rb[None, :])
    xr = tl.load(
        r_ptr + offs_m[:, None] * D + cols[None, :], mask=mask_m[:, None], other=0.0
    ).to(tl.float32)
    rs = tl.load(
        rs_ptr + offs_m[:, None] * HC_ + col_branch[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    wn = tl.load(wn_ptr + cols).to(tl.float32)
    acc = tl.dot(t_a, tl.trans(wu_a))
    acc = tl.dot(t_b, tl.trans(wu_b), acc)
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
    out = torch.empty_like(residual)
    total = residual.numel()
    _hc_combine_apply_kernel[(triton.cdiv(total, KC_BLOCK),)](
        y, residual, gate, out, total, HC_=HC, HS_=HS, BLOCK=KC_BLOCK, num_warps=4
    )
    return out


def hc_seam(
    y,
    residual: torch.Tensor,
    gate_prev,
    norm_w: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject_next,
    eps: float,
):
    """Combine (when ``y`` is given) + norm + low-rank gated mix (+ the next combine's gate).

    Returns (residual_new, mixed [M, HS] bf16, gate_new fp32 [M, HC] or None). Inputs must be contiguous bf16;
    ``gate_prev`` fp32 [M, HC]. At most MAX_ROWS rows.
    """
    rows = residual.shape[0]
    if rows > MAX_ROWS:
        raise ValueError(f"hc_seam: {rows} rows, the workspace holds 1..{MAX_ROWS}")
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
        y if has_combine else residual,
        residual,
        gate_prev if has_combine else ws["rs"],
        r_new,
        norm_w,
        w_down,
        ws["p"],
        ws["ss"],
        rows,
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
    nlt = triton.cdiv(LOWRANK, KF_BL)
    _hc_seam_reduce_kernel[
        (triton.cdiv(rows, KF_BM), nlt + (KF_BM * HC if has_gate else 0))
    ](
        ws["p"],
        ws["ss"],
        r_new,
        norm_w,
        w_inject_next if has_gate else norm_w,
        ws["t"],
        ws["rs"],
        gate_new if has_gate else ws["rs"],
        rows,
        eps,
        HC_=HC,
        HS_=HS,
        L=LOWRANK,
        KB=KA_KB,
        BM=KF_BM,
        BL=KF_BL,
        BKG=KF_BKG,
        NKP=triton.next_power_of_2(HS // KA_KB),
        NQP=triton.next_power_of_2(HS // KF_BKG),
        num_warps=4,
    )
    _hc_seam_up_kernel[(HS // KB_BJ, row_tiles)](
        ws["t"],
        ws["rs"],
        r_new,
        norm_w,
        w_up,
        mixed,
        rows,
        HC_=HC,
        HS_=HS,
        L=LOWRANK,
        BM=bm,
        BJ=KB_BJ,
        LA=KB_LA,
        LB=LOWRANK - KB_LA,
        num_warps=KB_WARPS,
    )
    return r_new, mixed, gate_new
