"""W8A16 fused MoE for a few decode tokens with 128x128 block-FP8 expert weights.

At M <= 4 the Triton fused MoE spends most of its time outside weight reads:
align/sort, two activation quantizations, a grouped GEMM tiled for M >= 16 and a
separate top-k sum. Here every (token, slot) pair maps directly to its expert,
so the whole block is two launches over bf16 activations:

1. up: each CTA owns BLOCK_N gate rows and the matching up rows of one slot over
   the full K and writes silu(gate) * up in fp32.
2. down: each CTA owns BLOCK_N output rows of one token and reduces all top-k
   slots in a fixed order, so the result is deterministic.

The per-block weight scales are applied to the activation, which is never
quantized; this is slightly more accurate than the W8A8 path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_GROUP = 128
MAX_TOKENS = 4


@triton.jit
def _small_m_moe_up_kernel(
    x_ptr,
    ids_ptr,
    w_ptr,
    s_ptr,
    act_ptr,
    stride_xm,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    eid = tl.load(ids_ptr + slot).to(tl.int64)
    x_row = x_ptr + (slot // TOPK) * stride_xm
    w = w_ptr + eid * stride_we
    s = s_ptr + eid * stride_se
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # BLOCK_N divides GROUP and I, so the gate and up rows each share one scale row.
    s_gate = s + ((pid_n * BLOCK_N) // GROUP) * stride_sn
    s_up = s + ((I + pid_n * BLOCK_N) // GROUP) * stride_sn
    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in tl.static_range(0, H, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        xk = tl.load(x_row + rk).to(tl.float32)
        xg = xk * tl.load(s_gate + (rk // GROUP) * stride_sk)
        xu = xk * tl.load(s_up + (rk // GROUP) * stride_sk)
        wg = tl.load(w + rn[:, None] * stride_wn + rk[None, :]).to(tl.float32)
        wu = tl.load(w + (rn[:, None] + I) * stride_wn + rk[None, :]).to(tl.float32)
        acc_g += tl.sum(wg * xg[None, :], axis=1)
        acc_u += tl.sum(wu * xu[None, :], axis=1)
    tl.store(act_ptr + slot * I + rn, acc_g / (1.0 + tl.exp(-acc_g)) * acc_u)


@triton.jit
def _small_m_moe_down_kernel(
    act_ptr,
    ids_ptr,
    wts_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    stride_ym,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    routed_scaling_factor,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tok = tl.program_id(0)
    pid_n = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    s_row = ((pid_n * BLOCK_N) // GROUP) * stride_sn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        slot = tok * TOPK + j
        eid = tl.load(ids_ptr + slot).to(tl.int64)
        w = w_ptr + eid * stride_we
        s = s_ptr + eid * stride_se + s_row
        part = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in tl.static_range(0, I, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(act_ptr + slot * I + rk) * tl.load(s + (rk // GROUP) * stride_sk)
            wk = tl.load(w + rn[:, None] * stride_wn + rk[None, :]).to(tl.float32)
            part += tl.sum(wk * a[None, :], axis=1)
        acc += part * tl.load(wts_ptr + slot).to(tl.float32)
    acc = acc * routed_scaling_factor
    tl.store(y_ptr + tok * stride_ym + rn, acc.to(y_ptr.dtype.element_ty))


def _fit_block_k(block_k: int, k: int) -> int:
    while k % block_k:
        block_k //= 2
    return block_k


# (BLOCK_N, BLOCK_K, num_warps) per token count, from a cold-L2 sweep on H100
# SXM for Qwen3.6-35B-A3B experts (H=2048, I=512, top-8 + fused shared expert).
_UP_CONFIGS = {1: (4, 512, 2), 2: (8, 256, 4), 3: (16, 256, 4), 4: (8, 1024, 4)}
_DOWN_CONFIGS = {1: (8, 512, 8), 2: (16, 512, 8), 3: (8, 1024, 4), 4: (32, 1024, 8)}


def _up_config(m: int, inter: int) -> tuple[int, int, int]:
    return _UP_CONFIGS[m]


def _down_config(m: int, hidden: int) -> tuple[int, int, int]:
    return _DOWN_CONFIGS[m]


def use_small_m_block_fp8_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
) -> bool:
    if not (
        hidden_states.dim() == 2
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.stride(1) == 1
        and w13.dtype == torch.float8_e4m3fn
        and w2.dtype == torch.float8_e4m3fn
        and w13.stride(2) == 1
        and w2.stride(2) == 1
        and w13_scale.dtype == torch.float32
        and w2_scale.dtype == torch.float32
        and topk_ids.is_contiguous()
    ):
        return False
    m, hidden = hidden_states.shape
    inter = w2.shape[2]
    return (
        1 <= m <= MAX_TOKENS
        and w13.shape[1] == 2 * inter
        and w13.shape[2] == hidden
        and hidden % _GROUP == 0
        and inter % _GROUP == 0
    )


def small_m_block_fp8_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """sum_j topk_weights[:, j] * expert_{topk_ids[:, j]}(hidden_states), SiLU-gated.

    w13: [E, 2 * I, H] fp8 (gate rows then up rows), w2: [E, H, I] fp8; the
    scales hold one fp32 per 128x128 block with any strides. All topk_ids must
    be valid local experts.
    """
    m, hidden = hidden_states.shape
    inter = w2.shape[2]
    topk = topk_ids.shape[1]
    weights = topk_weights.contiguous()
    act = torch.empty((m * topk, inter), dtype=torch.float32, device=w13.device)
    if out is None:
        out = torch.empty_like(hidden_states)

    bn, bk, nw = _up_config(m, inter)
    _small_m_moe_up_kernel[(m * topk, inter // bn)](
        hidden_states,
        topk_ids,
        w13,
        w13_scale,
        act,
        hidden_states.stride(0),
        w13.stride(0),
        w13.stride(1),
        w13_scale.stride(0),
        w13_scale.stride(1),
        w13_scale.stride(2),
        H=hidden,
        I=inter,
        TOPK=topk,
        GROUP=_GROUP,
        BLOCK_N=bn,
        BLOCK_K=_fit_block_k(bk, hidden),
        num_warps=nw,
    )
    bn, bk, nw = _down_config(m, hidden)
    _small_m_moe_down_kernel[(m, hidden // bn)](
        act,
        topk_ids,
        weights,
        w2,
        w2_scale,
        out,
        out.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        1.0 if routed_scaling_factor is None else routed_scaling_factor,
        H=hidden,
        I=inter,
        TOPK=topk,
        GROUP=_GROUP,
        BLOCK_N=bn,
        BLOCK_K=_fit_block_k(bk, inter),
        num_warps=nw,
    )
    return out
