"""Kimi-K3 MLA absorbed W_UK / W_UV for decode / target-verify batches (ROCm).

At these sizes the hipBLASLt batched GEMMs launch too few workgroups, so two
fused Triton launches replace them:

* ``k3_mla_absorb_q_cat_cache``: ``rocm_absorb_q_bmm`` + ``k3_mla_cat_cache``.
* ``k3_mla_absorb_v_gate``: ``rocm_absorb_v_bmm`` + the K3 output gate.

fp32 accumulation over the full K of one head with one rounding to bf16, like
the BMM (only summation order differs); the gate keeps the double rounding of
the unfused sigmoid + mul.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _k3_absorb_q_cat_cache_kernel(
    qn_ptr,  # [M, H, DK] strided, unit last stride
    qp_ptr,  # [M, H, DP]
    w_ptr,  # [H, DK, DN] (any strides)
    kn_ptr,  # [M, DN]
    kp_ptr,  # [M, DP]
    loc_ptr,  # [M]
    kv_ptr,  # [rows, DN + DP]
    qo_ptr,  # [M, H, DN + DP]
    M,
    s_qn_m,
    s_qn_h,
    s_qp_m,
    s_qp_h,
    s_w_h,
    s_w_k,
    s_w_n,
    s_kn_m,
    s_kp_m,
    s_kv,
    s_qo_m,
    s_qo_h,
    H: tl.constexpr,
    DK: tl.constexpr,
    DN: tl.constexpr,
    DP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    NB: tl.constexpr = DN // BN
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    if pid < H * NB:
        h = pid // NB
        nb = pid % NB
        offs_k = tl.arange(0, DK)
        offs_n = nb * BN + tl.arange(0, BN)
        q = tl.load(
            qn_ptr + offs_m[:, None] * s_qn_m + h * s_qn_h + offs_k[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptr + h * s_w_h + offs_k[:, None] * s_w_k + offs_n[None, :] * s_w_n
        )
        acc = tl.dot(q, w)
        base = qo_ptr + offs_m[:, None] * s_qo_m + h * s_qo_h
        tl.store(
            base + offs_n[None, :], acc.to(qo_ptr.dtype.element_ty), mask=mmask[:, None]
        )
        if nb == 0:
            offs_p = tl.arange(0, DP)
            xp = tl.load(
                qp_ptr + offs_m[:, None] * s_qp_m + h * s_qp_h + offs_p[None, :],
                mask=mmask[:, None],
            )
            tl.store(base + DN + offs_p[None, :], xp, mask=mmask[:, None])
    else:
        offs_c = tl.arange(0, DN)
        offs_e = tl.arange(0, DP)
        loc = tl.load(loc_ptr + offs_m, mask=mmask, other=0).to(tl.int64)
        kn = tl.load(
            kn_ptr + offs_m[:, None] * s_kn_m + offs_c[None, :], mask=mmask[:, None]
        )
        kp = tl.load(
            kp_ptr + offs_m[:, None] * s_kp_m + offs_e[None, :], mask=mmask[:, None]
        )
        dst = kv_ptr + loc[:, None] * s_kv
        tl.store(
            dst + offs_c[None, :], kn.to(kv_ptr.dtype.element_ty), mask=mmask[:, None]
        )
        tl.store(
            dst + DN + offs_e[None, :],
            kp.to(kv_ptr.dtype.element_ty),
            mask=mmask[:, None],
        )


@triton.jit
def _k3_absorb_v_gate_kernel(
    x_ptr,  # [M, H, DK] strided, unit last stride
    w_ptr,  # [H, DK, DV] (any strides)
    g_ptr,  # [M, H * DV] (unused when HAS_GATE False)
    o_ptr,  # [M, H * DV]
    M,
    s_x_m,
    s_x_h,
    s_w_h,
    s_w_k,
    s_w_n,
    s_g_m,
    s_o_m,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid = tl.program_id(0)
    NB: tl.constexpr = DV // BN
    h = pid // NB
    nb = pid % NB
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    offs_n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, DK, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(
            x_ptr + offs_m[:, None] * s_x_m + h * s_x_h + offs_k[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        w = tl.load(
            w_ptr + h * s_w_h + offs_k[:, None] * s_w_k + offs_n[None, :] * s_w_n
        )
        acc = tl.dot(x, w, acc)
    y = acc.to(o_ptr.dtype.element_ty)
    col = h * DV + offs_n
    if HAS_GATE:
        g = tl.load(
            g_ptr + offs_m[:, None] * s_g_m + col[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        s = (1.0 / (1.0 + tl.exp(-g.to(tl.float32)))).to(o_ptr.dtype.element_ty)
        y = (y.to(tl.float32) * s.to(tl.float32)).to(o_ptr.dtype.element_ty)
    tl.store(o_ptr + offs_m[:, None] * s_o_m + col[None, :], y, mask=mmask[:, None])


MAX_M = 64


def _bm(m: int) -> int:
    return max(16, triton.next_power_of_2(m))


def covered_q(q_nope, q_pe, w_kc, k_nope, k_pe, kv_buffer, loc) -> bool:
    if q_nope.dim() != 3 or q_pe.dim() != 3 or w_kc.dim() != 3:
        return False
    m, h, dk = q_nope.shape
    dn = w_kc.shape[2]
    dp = q_pe.shape[-1]
    if not (0 < m <= MAX_M) or q_pe.shape[:2] != (m, h) or w_kc.shape[:2] != (h, dk):
        return False
    if any(x & (x - 1) for x in (dk, dn, dp)) or dn % 64:
        return False
    if (
        q_nope.dtype != torch.bfloat16
        or q_pe.dtype != torch.bfloat16
        or w_kc.dtype != torch.bfloat16
    ):
        return False
    if k_nope.dtype != torch.bfloat16 or k_pe.dtype != torch.bfloat16:
        return False
    if k_nope.shape[0] != m or k_pe.shape[0] != m or loc.shape[0] != m:
        return False
    if k_nope.numel() != m * dn or k_pe.numel() != m * dp:
        return False
    if kv_buffer.shape[-1] != dn + dp or kv_buffer.numel() != kv_buffer.shape[0] * (
        dn + dp
    ):
        return False
    if (
        kv_buffer.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or not kv_buffer.is_contiguous()
    ):
        return False
    if (
        loc.dtype not in (torch.int32, torch.int64)
        or loc.dim() != 1
        or loc.stride(0) != 1
    ):
        return False
    if any(t.stride(-1) != 1 for t in (q_nope, q_pe, k_nope, k_pe)):
        return False
    if k_nope.reshape(m, dn).stride(-1) != 1 or k_pe.reshape(m, dp).stride(-1) != 1:
        return False
    return True


def k3_mla_absorb_q_cat_cache(
    q_nope, q_pe, w_kc, k_nope, k_pe, kv_buffer, loc, bn: int = 32
):
    """q_nope [M, H, DK], q_pe [M, H, DP], w_kc [H, DK, DN] (W_UK, any
    strides), k_nope/k_pe [M, (1,) DN/DP], kv_buffer [rows, (1,) DN + DP], loc
    [M]. Writes kv_buffer[loc[m]] = cast([k_nope | k_pe]) and returns q =
    [q_nope @ W_UK | q_pe] as a new contiguous [M, H, DN + DP]. Caller checks
    covered_q()."""
    m, h, dk = q_nope.shape
    dn = w_kc.shape[2]
    dp = q_pe.shape[-1]
    kn = k_nope.reshape(m, dn)
    kp = k_pe.reshape(m, dp)
    kv = kv_buffer.view(kv_buffer.shape[0], dn + dp)
    q_out = torch.empty((m, h, dn + dp), dtype=torch.bfloat16, device=q_nope.device)
    _k3_absorb_q_cat_cache_kernel[(h * (dn // bn) + 1,)](
        q_nope,
        q_pe,
        w_kc,
        kn,
        kp,
        loc,
        kv,
        q_out,
        m,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        w_kc.stride(0),
        w_kc.stride(1),
        w_kc.stride(2),
        kn.stride(0),
        kp.stride(0),
        kv.stride(0),
        q_out.stride(0),
        q_out.stride(1),
        H=h,
        DK=dk,
        DN=dn,
        DP=dp,
        BM=_bm(m),
        BN=bn,
        num_warps=4,
    )
    return q_out


def covered_v(attn_output, w_vc, gate) -> bool:
    if attn_output.dim() != 3 or w_vc.dim() != 3:
        return False
    m, h, dk = attn_output.shape
    dv = w_vc.shape[2]
    if not (0 < m <= MAX_M) or w_vc.shape[:2] != (h, dk):
        return False
    if dk % 128 or dv % 32 or dv & (dv - 1):
        return False
    if attn_output.dtype != torch.bfloat16 or w_vc.dtype != torch.bfloat16:
        return False
    if attn_output.stride(-1) != 1:
        return False
    if gate is not None:
        if (
            gate.dtype != torch.bfloat16
            or gate.shape != (m, h * dv)
            or gate.stride(-1) != 1
        ):
            return False
    return True


def k3_mla_absorb_v_gate(attn_output, w_vc, gate=None, bn: int = 16, bk: int = 512):
    """attn_output [M, H, DK] (strided ok), w_vc [H, DK, DV] (W_UV, any
    strides), gate [M, H * DV] or None. Returns [M, H * DV] bf16 =
    bf16(attn @ W_UV) (* bf16(sigmoid(gate)), double-rounded). Caller checks
    covered_v()."""
    m, h, dk = attn_output.shape
    dv = w_vc.shape[2]
    out = torch.empty((m, h * dv), dtype=torch.bfloat16, device=attn_output.device)
    g = gate if gate is not None else out
    _k3_absorb_v_gate_kernel[(h * (dv // bn),)](
        attn_output,
        w_vc,
        g,
        out,
        m,
        attn_output.stride(0),
        attn_output.stride(1),
        w_vc.stride(0),
        w_vc.stride(1),
        w_vc.stride(2),
        g.stride(0),
        out.stride(0),
        DK=dk,
        DV=dv,
        BM=_bm(m),
        BN=bn,
        BK=bk,
        HAS_GATE=gate is not None,
        num_warps=4,
    )
    return out
