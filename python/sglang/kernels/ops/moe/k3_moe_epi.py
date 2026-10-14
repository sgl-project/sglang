# SPDX-License-Identifier: Apache-2.0
"""Fused prologue/epilogue kernels for the Kimi-K3 MoE sub-layer on ROCm
(decode / target-verify, M = tokens <= a few hundred).

shared_down_topk: one launch that does two independent jobs of the fused
  front's consumers ("horizontal" fusion; no cross-CTA dependency):
    * CTAs [0, n_gemm): shared-expert down GEMM with the SiTU activation in
      its prologue:  out[M, N] = bf16(situ(gate) * up) @ w_down[N, I]^T
      (replaces sglang::situ_and_mul + the hipBLASLt down GEMM)
    * CTAs [n_gemm, n_gemm + M): sigmoid + correction-bias top-k of the router
      logits, one CTA per token (same math as aiter biased_grouped_topk with one
      group).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.moe.k3_route_sort import _sort_quant_program
from sglang.kernels.ops.moe.k3_route_sort_mid import _aiter_biased_topk

# --------------------------------------------------------------------------
# shared-expert down GEMM (SiTU prologue) + router top-k
# --------------------------------------------------------------------------


@triton.jit
def _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN: tl.constexpr):
    gate = beta * libdevice.tanh(g * inv_beta) * (1.0 / (1.0 + tl.exp(-g)))
    if HAS_LIN:
        u = lin_beta * libdevice.tanh(u * inv_lin_beta)
    return gate * u


@triton.jit
def _shared_down_tile(
    pid,
    a_ptr,  # [M, 2*I] gate|up, or [M, I] activation if PRE_ACT; row stride stride_a
    stride_a,
    wd_ptr,  # [N, I] contiguous
    out_ptr,  # [M, N], row stride stride_out
    stride_out,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    N: tl.constexpr,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRE_ACT: tl.constexpr,
):
    # one (BM, BN) tile of out = act(a) @ wd^T
    NT_N: tl.constexpr = N // BN
    pid_n = pid % NT_N
    pid_m = pid // NT_N
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, I, BK):
        offs_k = k0 + tl.arange(0, BK)
        row = a_ptr + offs_m[:, None] * stride_a + offs_k[None, :]
        w = tl.load(wd_ptr + offs_n[:, None] * I + offs_k[None, :])
        if PRE_ACT:
            a = tl.load(row, mask=mm[:, None], other=0.0)
        else:
            g = tl.load(row, mask=mm[:, None], other=0.0).to(tl.float32)
            u = tl.load(row + I, mask=mm[:, None], other=0.0).to(tl.float32)
            a = _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN).to(
                wd_ptr.dtype.element_ty
            )
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mm[:, None],
    )


@triton.jit(do_not_specialize=["stride_gu"])
def _k3_situ_kernel(
    gu_ptr,  # [M, 2*I] bf16 (gate | up)
    stride_gu,
    act_ptr,  # [M, I] bf16 out
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The SiTU activation once per element,
    # instead of once per down-GEMM CTA (224 CTAs at M=8 each recomputed all
    # M x I of it; that, not the 11 MB weight stream, bounded the fused kernel)
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mk = offs < I
    g = tl.load(gu_ptr + m * stride_gu + offs, mask=mk, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + m * stride_gu + I + offs, mask=mk, other=0.0).to(tl.float32)
    a = _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN)
    tl.store(act_ptr + m * I + offs, a.to(act_ptr.dtype.element_ty), mask=mk)


@triton.jit(do_not_specialize=["M", "stride_gu", "stride_out", "stride_lg"])
def _k3_shared_topk_kernel(
    gu_ptr,  # [M, 2*I] bf16 (gate | up), row stride stride_gu
    stride_gu,
    wd_ptr,  # [N, I] bf16 contiguous
    out_ptr,  # [M, N] bf16, row stride stride_out
    stride_out,
    lg_ptr,  # [M, E] router logits, row stride stride_lg
    stride_lg,
    bias_ptr,  # [E] correction bias
    topk_ids_ptr,  # [M, TOPK] i32 out
    topk_w_ptr,  # [M, TOPK] fp32 out
    routed_scale,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    N: tl.constexpr,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DO_TOPK: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
    PRE_ACT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    n_gemm = (N // BN) * tl.cdiv(M, BM)
    if pid < n_gemm:
        _shared_down_tile(
            pid,
            gu_ptr,
            stride_gu,
            wd_ptr,
            out_ptr,
            stride_out,
            M,
            beta,
            inv_beta,
            lin_beta,
            inv_lin_beta,
            N,
            I,
            HAS_LIN,
            BM,
            BN,
            BK,
            PRE_ACT,
        )
    else:
        if DO_TOPK:
            _topk_body(
                pid - n_gemm,
                lg_ptr,
                stride_lg,
                bias_ptr,
                topk_ids_ptr,
                topk_w_ptr,
                routed_scale,
                E,
                E_POW2,
                TOPK,
                RENORM,
            )


def _shared_cfg(m: int):
    # (BM, BN, BK, num_warps, num_stages); tuned on MI355X for N=7168, I=768
    if m <= 16:
        return 16, 32, 128, 8, 2
    if m <= 32:
        return 32, 32, 256, 4, 2
    if m <= 64:
        return 64, 32, 128, 4, 2
    return 64, 64, 128, 4, 2


def _shared_preact_cfg(m: int):
    # (BM, BN, BK, num_warps, num_stages) for the pre-activated GEMM; MI355X sweep
    if m <= 16:
        return 16, 32, 256, 4, 3
    if m <= 32:
        return 32, 16, 256, 4, 2
    return 64, 32, 256, 8, 2


def shared_down_topk(
    gate_up: torch.Tensor,  # [M, 2I] bf16, row-strided ok
    w_down: torch.Tensor,  # [N, I] bf16
    out: torch.Tensor,  # [M, N] bf16
    beta: float,
    linear_beta: Optional[float],
    router_logits: Optional[torch.Tensor] = None,  # [M, E], row-strided ok
    correction_bias: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,  # [M, TOPK] i32 out
    topk_weights: Optional[torch.Tensor] = None,  # [M, TOPK] fp32 out
    renormalize: bool = True,
    routed_scale: float = 1.0,
    cfg: Optional[tuple] = None,  # (BM, BN, BK, num_warps, num_stages)
    pre_act: bool = True,
) -> None:
    """pre_act (default on): run the SiTU
    activation as its own small launch first, so the GEMM CTAs only stream
    weights (2 launches instead of 1, but ~2x faster at decode sizes). The
    activation values, GEMM tiling/reduction order and top-k code are the
    same, so outputs are bit-identical to the single-launch kernel."""
    M, two_i = gate_up.shape
    I = two_i // 2
    N = w_down.shape[0]
    assert w_down.shape[1] == I and w_down.is_contiguous()
    assert gate_up.stride(1) == 1 and out.stride(1) == 1
    lin = linear_beta is not None
    if pre_act:
        act = torch.empty((M, I), dtype=w_down.dtype, device=gate_up.device)
        blk = 256
        _k3_situ_kernel[(M, triton.cdiv(I, blk))](
            gate_up,
            gate_up.stride(0),
            act,
            float(beta),
            1.0 / float(beta),
            float(linear_beta) if lin else 1.0,
            1.0 / float(linear_beta) if lin else 1.0,
            I=I,
            HAS_LIN=lin,
            BLOCK=blk,
            num_warps=2,
        )
        gate_up = act
    bm, bn, bk, nw, ns = cfg or (_shared_preact_cfg(M) if pre_act else _shared_cfg(M))
    assert N % bn == 0 and I % bk == 0
    do_topk = router_logits is not None
    if do_topk:
        E = router_logits.shape[1]
        topk = topk_ids.shape[1]
        assert router_logits.stride(1) == 1 and E < 0xFFFF and topk & (topk - 1) == 0
    else:
        E, topk = 1, 1
        router_logits = correction_bias = topk_ids = topk_weights = gate_up
    grid = ((N // bn) * triton.cdiv(M, bm) + (M if do_topk else 0),)
    _k3_shared_topk_kernel[grid](
        gate_up,
        gate_up.stride(0),
        w_down,
        out,
        out.stride(0),
        router_logits,
        router_logits.stride(0),
        correction_bias,
        topk_ids,
        topk_weights,
        float(routed_scale),
        M,
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if lin else 1.0,
        1.0 / float(linear_beta) if lin else 1.0,
        N=N,
        I=I,
        HAS_LIN=lin,
        BM=bm,
        BN=bn,
        BK=bk,
        DO_TOPK=do_topk,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=renormalize,
        PRE_ACT=pre_act,
        num_warps=nw,
        num_stages=ns,
    )


# --------------------------------------------------------------------------
# rstd-scale + add3 tail behind an unfused up GEMM (norm weight folded into w_up)
# --------------------------------------------------------------------------


@triton.jit(
    do_not_specialize=["stride_up", "stride_x", "stride_sh", "stride_pf", "stride_out"]
)
def _k3_norm_add3_kernel(
    up_ptr,  # [M, N] fp32/bf16 = x @ (w_up * norm_w)^T
    stride_up,
    x_ptr,  # [M, K] latent (pre-norm)
    stride_x,
    sh_ptr,
    stride_sh,
    pf_ptr,
    stride_pf,
    out_ptr,
    stride_out,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PF: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mn = offs < N
    up = tl.load(up_ptr + m * stride_up + offs, mask=mn, other=0.0).to(tl.float32)
    s = tl.load(sh_ptr + m * stride_sh + offs, mask=mn, other=0.0).to(tl.float32)
    if HAS_PF:
        p = tl.load(pf_ptr + m * stride_pf + offs, mask=mn, other=0.0).to(tl.float32)
    offs_k = tl.arange(0, K_POW2)
    x = tl.load(x_ptr + m * stride_x + offs_k, mask=offs_k < K, other=0.0).to(
        tl.float32
    )
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    y = (up * rstd).to(out_ptr.dtype.element_ty)
    y = (y.to(tl.float32) + s).to(out_ptr.dtype.element_ty)
    if HAS_PF:
        y = (y.to(tl.float32) + p).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + m * stride_out + offs, y, mask=mn)


def norm_add3(
    up: torch.Tensor,  # [M, N] fp32 (preferred) or bf16
    x: torch.Tensor,  # [M, K] latent the GEMM consumed (pre-norm)
    eps: float,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    block: int = 1024,
) -> torch.Tensor:
    """out = bf16(bf16(bf16(rstd(x) * up) + shared) + prefix)"""
    M, N = up.shape
    K = x.shape[1]
    if out is None:
        out = torch.empty((M, N), dtype=shared.dtype, device=up.device)
    has_pf = prefix is not None
    _k3_norm_add3_kernel[(M, triton.cdiv(N, block))](
        up,
        up.stride(0),
        x,
        x.stride(0),
        shared,
        shared.stride(0),
        prefix if has_pf else shared,
        prefix.stride(0) if has_pf else 0,
        out,
        out.stride(0),
        float(eps),
        N=N,
        K=K,
        K_POW2=triton.next_power_of_2(K),
        BLOCK=block,
        HAS_PF=has_pf,
        num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# Regroup the decode MoE routing glue by
# dependency instead of by producer (M <= 64, pre-activated shared expert):
#
#   before:  [SiTU] -> [shared down GEMM | top-k] -> [sort + mxfp8 quant] -> stage1
#   after:   [SiTU | top-k]  ->  [shared down GEMM | sort + mxfp8 quant] -> stage1
#
# SiTU and top-k both read only the front GEMM output; the shared down GEMM
# (reads the SiTU output) and the sort/quant (reads the top-k output) both
# feed only later kernels. Same bodies, same tiling, same reduction order as
# _k3_situ_kernel / _k3_shared_topk_kernel / k3_route_sort._k3_sort_quant_kernel,
# so every output is bit-identical; one launch fewer per MoE layer.
#
# The sort runs inside the aiter MoE runner (moe_sorting_small._run_small_sort),
# so the model plans the shared GEMM for its top-k ids tensor
# (moe_sorting_small.K3SortPlan.shared_gemm); the runner's sort call takes it and
# launches the combined kernel, and the plan runs an untaken GEMM standalone.
# --------------------------------------------------------------------------


@triton.jit
def _topk_body(
    t,
    lg_ptr,
    stride_lg,
    bias_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    routed_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    offs_e = tl.arange(0, E_POW2)
    ids, wt = _aiter_biased_topk(
        lg_ptr + t * stride_lg,
        bias_ptr,
        offs_e,
        offs_e < E,
        TOPK,
        RENORM,
        routed_scale,
    )
    offs_k = tl.arange(0, TOPK)
    tl.store(topk_ids_ptr + t * TOPK + offs_k, ids)
    tl.store(topk_w_ptr + t * TOPK + offs_k, wt)


@triton.jit(do_not_specialize=["M", "stride_gu", "stride_lg"])
def _k3_situ_topk_kernel(
    gu_ptr,
    stride_gu,
    act_ptr,
    M,
    beta,
    inv_beta,
    lin_beta,
    inv_lin_beta,
    lg_ptr,
    stride_lg,
    bias_ptr,
    topk_ids_ptr,
    topk_w_ptr,
    routed_scale,
    I: tl.constexpr,
    HAS_LIN: tl.constexpr,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    RENORM: tl.constexpr,
):
    pid = tl.program_id(0)
    NB: tl.constexpr = (I + BLOCK - 1) // BLOCK
    n_situ = M * NB
    if pid < n_situ:
        # == _k3_situ_kernel, program (m, blk)
        m = pid // NB
        offs = (pid % NB) * BLOCK + tl.arange(0, BLOCK)
        mk = offs < I
        g = tl.load(gu_ptr + m * stride_gu + offs, mask=mk, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + m * stride_gu + I + offs, mask=mk, other=0.0).to(
            tl.float32
        )
        a = _situ(g, u, beta, inv_beta, lin_beta, inv_lin_beta, HAS_LIN)
        tl.store(act_ptr + m * I + offs, a.to(act_ptr.dtype.element_ty), mask=mk)
    else:
        _topk_body(
            pid - n_situ,
            lg_ptr,
            stride_lg,
            bias_ptr,
            topk_ids_ptr,
            topk_w_ptr,
            routed_scale,
            E,
            E_POW2,
            TOPK,
            RENORM,
        )


@triton.jit(do_not_specialize=["M", "stride_out", "stride_qx"])
def _k3_shared_sortq_kernel(
    act_ptr,
    wd_ptr,
    out_ptr,
    stride_out,
    topk_ids_ptr,
    topk_weights_ptr,
    sorted_ids_ptr,
    sorted_weights_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    moe_buf_ptr,
    qx_ptr,
    stride_qx,
    qout_ptr,
    qscale_ptr,
    M,
    N: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P_POW2: tl.constexpr,
    N_COLS: tl.constexpr,
    QCHUNK: tl.constexpr,
    SCALEN_PAD: tl.constexpr,
    MOE_BUF_ZERO: tl.constexpr,
):
    # sort/quant programs first: they gate stage 1, the GEMM only the AR
    n_sort = M * (N_COLS // QCHUNK)
    pid = tl.program_id(0)
    if pid >= n_sort:
        _shared_down_tile(
            pid - n_sort,
            act_ptr,
            I,
            wd_ptr,
            out_ptr,
            stride_out,
            M,
            0.0,
            0.0,
            0.0,
            0.0,
            N,
            I,
            False,
            BM,
            BN,
            BK,
            True,
        )
    else:
        _sort_quant_program(
            pid,
            topk_ids_ptr,
            topk_weights_ptr,
            sorted_ids_ptr,
            sorted_weights_ptr,
            sorted_expert_ids_ptr,
            num_valid_ids_ptr,
            moe_buf_ptr,
            qx_ptr,
            stride_qx,
            qout_ptr,
            qscale_ptr,
            M,
            TOPK,
            BLOCK_SIZE,
            P_POW2,
            N_COLS,
            QCHUNK,
            SCALEN_PAD,
            MOE_BUF_ZERO,
        )


def situ_topk(
    gate_up: torch.Tensor,  # [M, 2I]
    act: torch.Tensor,  # [M, I] out
    beta: float,
    linear_beta: Optional[float],
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    renormalize: bool = True,
    routed_scale: float = 1.0,
) -> None:
    """Launch A: SiTU pre-activation + router top-k (bit-identical to
    _k3_situ_kernel + the top-k CTAs of _k3_shared_topk_kernel)."""
    M, two_i = gate_up.shape
    I = two_i // 2
    E = router_logits.shape[1]
    topk = topk_ids.shape[1]
    assert (
        gate_up.stride(1) == 1 and router_logits.stride(1) == 1 and act.is_contiguous()
    )
    assert E < 0xFFFF and topk & (topk - 1) == 0
    lin = linear_beta is not None
    blk = 256
    _k3_situ_topk_kernel[(M * triton.cdiv(I, blk) + M,)](
        gate_up,
        gate_up.stride(0),
        act,
        M,
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if lin else 1.0,
        1.0 / float(linear_beta) if lin else 1.0,
        router_logits,
        router_logits.stride(0),
        correction_bias,
        topk_ids,
        topk_weights,
        float(routed_scale),
        I=I,
        HAS_LIN=lin,
        BLOCK=blk,
        E=E,
        E_POW2=triton.next_power_of_2(E),
        TOPK=topk,
        RENORM=renormalize,
        num_warps=4,
    )


def shared_gemm_preact(act, w_down, out) -> None:
    M, I = act.shape
    N = w_down.shape[0]
    bm, bn, bk, nw, ns = _shared_preact_cfg(M)
    _k3_shared_topk_kernel[((N // bn) * triton.cdiv(M, bm),)](
        act,
        act.stride(0),
        w_down,
        out,
        out.stride(0),
        act,
        0,
        act,
        act,
        act,
        1.0,
        M,
        1.0,
        1.0,
        1.0,
        1.0,
        N=N,
        I=I,
        HAS_LIN=False,
        BM=bm,
        BN=bn,
        BK=bk,
        DO_TOPK=False,
        E=1,
        E_POW2=1,
        TOPK=1,
        RENORM=False,
        PRE_ACT=True,
        num_warps=nw,
        num_stages=ns,
    )


def shared_gemm_sort_quant(
    pending,
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    block_size,
    mx_quant_input,
    qout,
    qscale,
) -> None:
    """Launch B: the pending shared down GEMM + k3_route_sort.sort_quant."""
    from sglang.kernels.ops.moe.k3_route_sort import SORT_QCHUNK

    act, w_down, out = pending
    M, I = act.shape
    N = w_down.shape[0]
    m, topk = topk_ids.shape
    n = mx_quant_input.shape[1]
    assert (
        m == M and act.is_contiguous() and w_down.is_contiguous() and out.stride(1) == 1
    )
    bm, bn, bk, nw, ns = _shared_preact_cfg(M)
    # wider N tiles leave fewer CTAs next to the sort programs; the K loop and
    # MFMA order are unchanged, so the output stays bit-identical
    bn = 64 if M <= 16 else bn
    n_gemm = (N // bn) * triton.cdiv(M, bm)
    _k3_shared_sortq_kernel[(n_gemm + m * (n // SORT_QCHUNK),)](
        act,
        w_down,
        out,
        out.stride(0),
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        mx_quant_input,
        mx_quant_input.stride(0),
        qout,
        qscale,
        M,
        N=N,
        I=I,
        BM=bm,
        BN=bn,
        BK=bk,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        P_POW2=triton.next_power_of_2(m * topk),
        N_COLS=n,
        QCHUNK=SORT_QCHUNK,
        SCALEN_PAD=qscale.shape[1],
        MOE_BUF_ZERO=moe_buf.numel() > 0,
        num_warps=nw,
        num_stages=ns,
    )
