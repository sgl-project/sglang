"""Kimi-K3 MLA prefill/extend MHA (d_qk 192, d_v 128) on AITER's gfx950 FP8
persistent-scheduled ASM kernel (``mla_prefill_ps_asm_fwd``) + ``mla_reduce_v1``.

Unlike the bf16 opus varlen kernel it splits KV across all CUs, which matters
for short extends over a long prefix. Q/K/V are cast to E4M3 with unit scales
(saturating at +-448), as TRT-LLM's FP8 ragged prefill.

mla_reduce_v1 has no 12-head instance: the [rows, 12, 128] buffers are viewed
as [3 * rows, 4, 128] and the reduce row maps scaled by 3 (exact, the reduction
is per (row, head, dim)).

The ASM kernel takes no strides: K8 [N, H, 192] and V8 [N, H, 128] must be
dense, rows ordered per request exactly like kv_indptr (prefix then extend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.k3_mla_prefix_nocat import _gather_dequant

_TILE_Q = 256
_FP8_MAX = 448.0
# The ASM kernel addresses K/V with 32-bit byte offsets.
_MAX_KV_BYTES = 2**31 - 1


@dataclass
class K3Fp8PrefillPlan:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    work_indptr: torch.Tensor
    work_info: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map3: torch.Tensor
    reduce_partial_map3: torch.Tensor
    num_partial_rows: int
    max_q_len: int
    one: torch.Tensor


def supported(num_heads: int, qk_head_dim: int, v_head_dim: int, total_kv: int) -> bool:
    return (
        qk_head_dim == 192
        and v_head_dim == 128
        and num_heads % 4 == 0
        and total_kv * num_heads * qk_head_dim < _MAX_KV_BYTES
    )


def make_plan(
    qo_lens: List[int], kv_lens: List[int], num_heads: int, device: torch.device
) -> K3Fp8PrefillPlan:
    """Host-side PS metadata for one extend batch (shared by all layers).
    qo_lens / kv_lens: per-request new tokens / total KV (prefix + new)."""
    import aiter

    bs = len(qo_lens)
    qo_cpu = torch.zeros(bs + 1, dtype=torch.int32)
    kv_cpu = torch.zeros(bs + 1, dtype=torch.int32)
    qo_cpu[1:] = torch.cumsum(torch.tensor(qo_lens, dtype=torch.int32), 0)
    kv_cpu[1:] = torch.cumsum(torch.tensor(kv_lens, dtype=torch.int32), 0)
    seq_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32)
    max_q = max(qo_lens)
    total_q = int(qo_cpu[-1])
    total_kv = int(kv_cpu[-1])
    info = aiter.get_ps_metadata_info_v1(
        batch_size=bs,
        num_head_k=num_heads,
        max_qlen=max_q,
        qlen_granularity=_TILE_Q,
        total_qlen=total_q,
    )
    bufs = [torch.empty(sz, dtype=dt, device=device) for sz, dt in info]
    aiter.get_ps_metadata_v1(
        qo_cpu,
        kv_cpu,
        seq_lens_cpu,
        1,
        num_heads,
        *bufs,
        qhead_granularity=1,
        qlen_granularity=_TILE_Q,
        kvlen_granularity=128,
        block_size=1,
        is_causal=True,
    )
    _, work_indptr, work_info, red_indptr, red_final, red_partial = bufs
    g = num_heads // 4
    red_final3 = red_final * g
    red_partial3 = torch.where(red_partial >= 0, red_partial * g, red_partial)
    return K3Fp8PrefillPlan(
        qo_indptr=qo_cpu.to(device, non_blocking=True),
        kv_indptr=kv_cpu.to(device, non_blocking=True),
        kv_page_indices=torch.arange(total_kv, dtype=torch.int32, device=device),
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=red_indptr,
        reduce_final_map3=red_final3,
        reduce_partial_map3=red_partial3,
        num_partial_rows=red_partial.size(0) * _TILE_Q,
        max_q_len=max_q,
        one=torch.ones((), dtype=torch.float32, device=device),
    )


# --------------------------------------------------------------------------- casts
@triton.jit
def _cast3d_kernel(
    X,
    Y,
    R,
    H: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    sx_r,
    sx_h,
    BLOCK_R: tl.constexpr,
    FP8MAX: tl.constexpr,
):
    """x [R, H, D] (strided, unit last stride) -> dense fp8 y [R, H, D], saturating."""
    pid = tl.program_id(0)
    rows = (pid * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
    rm = rows < R
    d = tl.arange(0, DP)
    dm = d < D
    for h in tl.static_range(H):
        x = tl.load(
            X + rows[:, None] * sx_r + h * sx_h + d[None, :],
            mask=rm[:, None] & dm[None, :],
            other=0.0,
        )
        x = tl.clamp(x.to(tl.float32), -FP8MAX, FP8MAX)
        tl.store(
            Y + (rows[:, None] * H + h) * D + d[None, :],
            x.to(Y.dtype.element_ty),
            mask=rm[:, None] & dm[None, :],
        )


def cast_fp8(x: torch.Tensor, fp8_dtype) -> torch.Tensor:
    R, H, D = x.shape
    assert x.stride(2) == 1
    y = torch.empty((R, H, D), dtype=fp8_dtype, device=x.device)
    if R == 0:
        return y
    block_r = 16
    _cast3d_kernel[(triton.cdiv(R, block_r),)](
        x,
        y,
        R,
        H=H,
        D=D,
        DP=triton.next_power_of_2(D),
        sx_r=x.stride(0),
        sx_h=x.stride(1),
        BLOCK_R=block_r,
        FP8MAX=_FP8_MAX,
        num_warps=4,
    )
    return y


def gather_latent(
    cache2d: torch.Tensor, kv_indices: torch.Tensor, lora: int, dtype
) -> torch.Tensor:
    """cache[idx, :lora] (fp8, unit scale) -> bf16 [N, lora] (exact)."""
    out = torch.empty((kv_indices.shape[0], lora), dtype=dtype, device=cache2d.device)
    _gather_dequant(cache2d, kv_indices, out, None, lora, cache2d.shape[1] - lora)
    return out


@triton.jit
def _kvb_to_fp8_kernel(
    KV,
    CACHE,
    IDX,
    K8,
    V8,
    N,
    s_kv,
    s_cache,
    H: tl.constexpr,
    DN: tl.constexpr,
    DV: tl.constexpr,
    LORA: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8MAX: tl.constexpr,
):
    """kv_b output kv [N, H*(DN+DV)] bf16 (per head [k_nope | v]) + k_pe from the
    fp8 latent cache -> dense K8 [N, H, DN+ROPE], V8 [N, H, DV] (e4m3, unit scale)."""
    pid = tl.program_id(0)
    rows = (pid * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rm = rows < N
    idx = tl.load(IDX + rows, mask=rm, other=0).to(tl.int64)
    offs_r = tl.arange(0, ROPE)[None, :]
    pe = tl.load(
        CACHE + idx[:, None] * s_cache + LORA + offs_r, mask=rm[:, None], other=0.0
    )
    pe = tl.clamp(pe.to(tl.float32), -FP8MAX, FP8MAX).to(K8.dtype.element_ty)
    offs_n = tl.arange(0, DN)[None, :]
    offs_v = tl.arange(0, DV)[None, :]
    DK: tl.constexpr = DN + ROPE
    for h in tl.static_range(H):
        base = KV + rows[:, None] * s_kv + h * (DN + DV)
        kn = tl.load(base + offs_n, mask=rm[:, None], other=0.0).to(tl.float32)
        vv = tl.load(base + DN + offs_v, mask=rm[:, None], other=0.0).to(tl.float32)
        kn = tl.clamp(kn, -FP8MAX, FP8MAX).to(K8.dtype.element_ty)
        vv = tl.clamp(vv, -FP8MAX, FP8MAX).to(V8.dtype.element_ty)
        kb = K8 + (rows[:, None] * H + h) * DK
        tl.store(kb + offs_n, kn, mask=rm[:, None])
        tl.store(kb + DN + offs_r, pe, mask=rm[:, None])
        tl.store(V8 + (rows[:, None] * H + h) * DV + offs_v, vv, mask=rm[:, None])


def kvb_to_fp8(
    kv: torch.Tensor,
    cache2d: torch.Tensor,
    kv_indices: torch.Tensor,
    num_heads: int,
    nope: int,
    v_dim: int,
    lora: int,
    rope: int,
    fp8_dtype,
):
    n = kv.shape[0]
    assert (
        kv.dim() == 2
        and kv.stride(1) == 1
        and kv.shape[1] == num_heads * (nope + v_dim)
    )
    k8 = torch.empty((n, num_heads, nope + rope), dtype=fp8_dtype, device=kv.device)
    v8 = torch.empty((n, num_heads, v_dim), dtype=fp8_dtype, device=kv.device)
    if n:
        block_n = 8
        _kvb_to_fp8_kernel[(triton.cdiv(n, block_n),)](
            kv,
            cache2d,
            kv_indices,
            k8,
            v8,
            n,
            kv.stride(0),
            cache2d.stride(0),
            H=num_heads,
            DN=nope,
            DV=v_dim,
            LORA=lora,
            ROPE=rope,
            BLOCK_N=block_n,
            FP8MAX=_FP8_MAX,
            num_warps=4,
        )
    return k8, v8


# --------------------------------------------------------------------------- attention
def attention(
    q8: torch.Tensor,
    k8: torch.Tensor,
    v8: torch.Tensor,
    plan: K3Fp8PrefillPlan,
    softmax_scale: float,
    out_dtype=torch.bfloat16,
) -> torch.Tensor:
    """q8 [Tq, H, 192], k8 [Tk, H, 192], v8 [Tk, H, 128] dense e4m3 (unit scale)."""
    import aiter

    T, H, _ = q8.shape
    DV = v8.shape[-1]
    dev = q8.device
    R = plan.num_partial_rows
    logits = torch.empty((R, H, DV), dtype=torch.float32, device=dev)
    lse = torch.empty((R, H), dtype=torch.float32, device=dev)
    out = torch.empty((T, H, DV), dtype=out_dtype, device=dev)
    flse = torch.empty((T, H), dtype=torch.float32, device=dev)
    aiter.mla_prefill_ps_asm_fwd(
        q8,
        k8,
        v8,
        plan.qo_indptr,
        plan.kv_indptr,
        plan.kv_page_indices,
        plan.work_indptr,
        plan.work_info,
        plan.max_q_len,
        softmax_scale,
        True,
        logits,
        lse,
        out,
        plan.one,
        plan.one,
        plan.one,
    )
    g = H // 4
    aiter.mla_reduce_v1(
        logits.view(-1, 4, DV),
        lse.view(-1, 4),
        plan.reduce_indptr,
        plan.reduce_final_map3,
        plan.reduce_partial_map3,
        _TILE_Q * g,
        0,
        out.view(-1, 4, DV),
        flse.view(-1, 4),
    )
    return out
