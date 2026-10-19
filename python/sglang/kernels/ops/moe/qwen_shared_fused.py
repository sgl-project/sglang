"""Qwen3.5 shared expert (FP8 per-tensor gate_up / down, sigmoid gate) at decode token counts in three launches
instead of six: the served static FP8 input quant, one gate_up + SiLU*mul + quant kernel, and one down GEMM with
the sigmoid gate in its epilogue. The two GEMM kernels run on 16-48 CTAs (the cuBLASLt GEMMs take 128 CTAs at one
CTA per SM), so the chain leaves the SMs to the routed experts. Rounding points follow the served chain; FP32
accumulation orders are Triton's.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

MAX_TOKENS = 128
# PDL early launch of the down kernel pays off only when the routed front is short (measured on the decode layer graph).
_PDL_DOWN_MAX_TOKENS = 16
_FP8_MAX = tl.constexpr(448.0)
_BN_UP = 32
_BK_UP = 256
_GATE_ROWS = 8
_GATE_BK = 1024
_BN_DOWN = 128
_BK_DOWN = 256


@triton.jit
def _gate_up_silu_quant_kernel(
    xq_ptr,  # [M, K] e4m3, the served static quant of x
    x_ptr,  # [M, K] bf16
    w1_ptr,  # [2 * I, K] e4m3, gate rows then up rows
    wg_ptr,  # [K] bf16, the shared-expert gate weight
    h_ptr,  # [M, I] e4m3 out
    gate_ptr,  # [M] fp32 out, sigmoid(x . wg)
    M,
    s1_ptr,
    ws1_ptr,
    s2_ptr,
    K: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GR: tl.constexpr,
    GK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Programs [0, I / BN) own BN gate columns and the matching up columns; the rest compute the gate for GR rows.
    pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if pid < I // BN:
        rm = tl.arange(0, BM)
        rn = pid * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        mmask = rm < M
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            xq = tl.load(
                xq_ptr + rm[:, None] * K + (k0 + rk)[None, :],
                mask=mmask[:, None],
                other=0.0,
            )
            wg = tl.load(w1_ptr + rn[:, None] * K + (k0 + rk)[None, :])
            wu = tl.load(w1_ptr + (I + rn)[:, None] * K + (k0 + rk)[None, :])
            acc_g = tl.dot(xq, tl.trans(wg), acc_g)
            acc_u = tl.dot(xq, tl.trans(wu), acc_u)
        sc = tl.load(s1_ptr) * tl.load(ws1_ptr)
        g = (acc_g * sc).to(tl.bfloat16).to(tl.float32)
        u = (acc_u * sc).to(tl.bfloat16).to(tl.float32)
        # SiLU as the served act_and_mul kernel computes it: x / (1 + expf(-x)), then * up, BF16.
        a = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))
        h = (a * u).to(tl.bfloat16).to(tl.float32)
        hq = tl.clamp(h * (1.0 / tl.load(s2_ptr)), -_FP8_MAX, _FP8_MAX).to(
            tl.float8e4nv
        )
        tl.store(h_ptr + rm[:, None] * I + rn[None, :], hq, mask=mmask[:, None])
    else:
        rr = (pid - I // BN) * GR + tl.arange(0, GR)
        rmask = rr < M
        rkg = tl.arange(0, GK)
        gsum = tl.zeros((GR,), dtype=tl.float32)
        for kg in range(0, K, GK):
            xg = tl.load(
                x_ptr + rr[:, None] * K + (kg + rkg)[None, :],
                mask=rmask[:, None],
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(wg_ptr + kg + rkg).to(tl.float32)
            gsum += tl.sum(xg * wv[None, :], axis=1)
        tl.store(gate_ptr + rr, tl.sigmoid(gsum), mask=rmask)


@triton.jit
def _down_gate_kernel(
    h_ptr,  # [M, I] e4m3
    w2_ptr,  # [H, I] e4m3
    gate_ptr,  # [M] fp32
    out_ptr,  # [M, H] bf16
    M,
    s2_ptr,
    ws2_ptr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = pid * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = rm < M
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, I, BK):
        hb = tl.load(
            h_ptr + rm[:, None] * I + (k0 + rk)[None, :], mask=mmask[:, None], other=0.0
        )
        wt = tl.load(w2_ptr + rn[:, None] * I + (k0 + rk)[None, :])
        acc = tl.dot(hb, tl.trans(wt), acc)
    y = (acc * (tl.load(s2_ptr) * tl.load(ws2_ptr))).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(gate_ptr + rm, mask=mmask, other=0.0)
    out = (gate[:, None] * y).to(tl.bfloat16)
    tl.store(out_ptr + rm[:, None] * H + rn[None, :], out, mask=mmask[:, None])


def shared_expert_fused(
    x: torch.Tensor,
    w1_nk: torch.Tensor,
    w2_nk: torch.Tensor,
    gate_weight: torch.Tensor,
    in_scale1: torch.Tensor,
    w_scale1: torch.Tensor,
    in_scale2: torch.Tensor,
    w_scale2: torch.Tensor,
) -> torch.Tensor:
    """sigmoid(x @ gate_weight) * down(SiLU(gate) * up), BF16 [M, H].

    x [M, H] bf16 (M <= 128); w1_nk [2I, H] and w2_nk [H, I] e4m3, row-major; gate_weight [H] bf16;
    static per-tensor scales [1] fp32 (input scale, weight scale) for each GEMM.
    """
    m, hidden = x.shape
    inter = w2_nk.shape[1]
    bm = max(16, triton.next_power_of_2(m))
    xq, _ = static_quant_fp8(x, in_scale1)
    h = torch.empty((m, inter), dtype=torch.float8_e4m3fn, device=x.device)
    gate = torch.empty((m,), dtype=torch.float32, device=x.device)
    out = torch.empty((m, hidden), dtype=torch.bfloat16, device=x.device)
    grid_up = inter // _BN_UP + triton.cdiv(m, _GATE_ROWS)
    _gate_up_silu_quant_kernel[(grid_up,)](
        xq,
        x,
        w1_nk,
        gate_weight,
        h,
        gate,
        m,
        in_scale1,
        w_scale1,
        in_scale2,
        K=hidden,
        I=inter,
        BM=bm,
        BN=_BN_UP,
        BK=_BK_UP,
        GR=_GATE_ROWS,
        GK=_GATE_BK,
        USE_PDL=True,
        num_warps=4,
        num_stages=4,
        launch_pdl=True,
    )
    pdl_down = m <= _PDL_DOWN_MAX_TOKENS
    _down_gate_kernel[(hidden // _BN_DOWN,)](
        h,
        w2_nk,
        gate,
        out,
        m,
        in_scale2,
        w_scale2,
        H=hidden,
        I=inter,
        BM=bm,
        BN=_BN_DOWN,
        BK=_BK_DOWN,
        USE_PDL=pdl_down,
        num_warps=4,
        num_stages=2,
        launch_pdl=pdl_down,
    )
    return out
