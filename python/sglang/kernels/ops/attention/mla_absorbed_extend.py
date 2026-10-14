"""Absorbed MLA cached extend over an FP8 latent cache (gfx950).

A short extend after a long cached prefix attends the paged latent cache
directly with absorbed queries ``[T, H, kv_lora_rank + rope]`` instead of
expanding every prefix token to per-head K/V through kv_b_proj. Both costs grow
with the prefix and this one also with the new tokens, so it serves only short
extends (SGLANG_ROCM_MLA_ABSORBED_EXTEND_MAX_TOKENS). Each program packs BLOCK_Q
query positions times BLOCK_H heads into the MFMA M dimension so one KV tile
feeds all of them, over one split of the keys; a second kernel combines the
splits by their LSE.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

# BLOCK_Q x BLOCK_H = 64 rows keep the [64, 512] fp32 accumulator in registers on gfx950 at
# 4 warps (402 VGPRs incl. AGPRs, no spills); 128 rows spill. Measured at a 50K prefix.
_BLOCK_Q = 4
_BLOCK_H = 16
# One program covers every head of its query block.
MAX_HEADS = _BLOCK_H
_BLOCK_N = 32
# Programs to aim for across the batch, and the minimum keys per split.
_TARGET_PROGRAMS = 512
_MIN_SPLIT_KEYS = 512
_MAX_SPLITS = 64


class AbsorbedExtendPlan(NamedTuple):
    """Per-batch work list: the request and first query of each BLOCK_Q block."""

    block_req: torch.Tensor
    block_q0: torch.Tensor
    num_splits: int
    total_q: int


def plan_absorbed_extend(
    extend_lens_cpu, kv_lens_cpu, device: torch.device
) -> AbsorbedExtendPlan:
    """The work list of mla_absorbed_extend from the per-request host lengths."""
    reqs, q0s = [], []
    for b, q_len in enumerate(extend_lens_cpu):
        for q0 in range(0, int(q_len), _BLOCK_Q):
            reqs.append(b)
            q0s.append(q0)
    max_kv = max(int(k) for k in kv_lens_cpu)
    num_splits = max(
        1,
        min(
            _MAX_SPLITS,
            _TARGET_PROGRAMS // max(len(reqs), 1),
            triton.cdiv(max_kv, _MIN_SPLIT_KEYS),
        ),
    )
    work = torch.tensor([reqs, q0s], dtype=torch.int32).to(device, non_blocking=True)
    return AbsorbedExtendPlan(
        work[0], work[1], num_splits, int(sum(int(x) for x in extend_lens_cpu))
    )


@triton.jit
def _absorbed_extend_kernel(
    Q,
    KV,
    KV_INDICES,
    KV_INDPTR,
    QO_INDPTR,
    BLOCK_REQ,
    BLOCK_Q0,
    PART_O,
    PART_LSE,
    sm_scale,
    kv_scale,
    stride_qt,
    stride_qh,
    stride_kv,
    stride_ps,
    stride_pt,
    stride_ph,
    stride_ls,
    stride_lt,
    H: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    b = tl.load(BLOCK_REQ + pid)
    q0 = tl.load(BLOCK_Q0 + pid)
    qo_start = tl.load(QO_INDPTR + b)
    q_len = tl.load(QO_INDPTR + b + 1) - qo_start
    kv_start = tl.load(KV_INDPTR + b)
    prefix = tl.load(KV_INDPTR + b + 1) - kv_start - q_len

    M: tl.constexpr = BLOCK_Q * BLOCK_H
    rows = tl.arange(0, M)
    qi = q0 + rows // BLOCK_H
    hh = rows % BLOCK_H
    row_ok = (qi < q_len) & (hh < H)
    qpos = prefix + qi
    # This block's queries see keys [0, kv_hi); this program takes one split.
    kv_hi = prefix + tl.minimum(q0 + BLOCK_Q, q_len)
    split = tl.cdiv(tl.cdiv(kv_hi, NUM_SPLITS), BLOCK_N) * BLOCK_N
    lo = sid * split
    hi = tl.minimum(lo + split, kv_hi)

    offs_l = tl.arange(0, LATENT)
    offs_r = tl.arange(0, ROPE)
    offs_n = tl.arange(0, BLOCK_N)
    q_row = (
        Q + (qo_start + qi).to(tl.int64)[:, None] * stride_qt + hh[:, None] * stride_qh
    )
    q_nope = tl.load(q_row + offs_l[None, :], mask=row_ok[:, None], other=0.0)
    q_rope = tl.load(q_row + LATENT + offs_r[None, :], mask=row_ok[:, None], other=0.0)

    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, LATENT], tl.float32)
    for start in range(lo, hi, BLOCK_N):
        k_ok = start + offs_n < hi
        idx = tl.load(KV_INDICES + kv_start + start + offs_n, mask=k_ok, other=0)
        k_row = KV + idx.to(tl.int64)[:, None] * stride_kv
        k_nope = tl.load(k_row + offs_l[None, :], mask=k_ok[:, None], other=0.0)
        k_rope = tl.load(
            k_row + LATENT + offs_r[None, :], mask=k_ok[:, None], other=0.0
        )
        s = tl.dot(q_nope, tl.trans(k_nope)) + tl.dot(q_rope, tl.trans(k_rope))
        s = s * (sm_scale * kv_scale)
        seen = (start + offs_n)[None, :] <= qpos[:, None]
        s = tl.where(seen & k_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # Rows that have seen no key yet keep m = -inf; shift by 0 there.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_nope.to(tl.bfloat16))
        m_i = m_new

    has = l_i > 0.0
    out = acc * (kv_scale / tl.where(has, l_i, 1.0))[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    t = (qo_start + qi).to(tl.int64)
    po = PART_O + sid * stride_ps + t[:, None] * stride_pt + hh[:, None] * stride_ph
    tl.store(po + offs_l[None, :], out, mask=row_ok[:, None])
    tl.store(PART_LSE + sid * stride_ls + t * stride_lt + hh, lse, mask=row_ok)


@triton.jit
def _combine_kernel(
    PART_O,
    PART_LSE,
    O,
    stride_ps,
    stride_pt,
    stride_ph,
    stride_ls,
    stride_lt,
    stride_ot,
    stride_oh,
    NUM_SPLITS: tl.constexpr,
    SPLITS_P2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_l = tl.program_id(2) * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_s = tl.arange(0, SPLITS_P2)
    lse = tl.load(
        PART_LSE + offs_s * stride_ls + t * stride_lt + h,
        mask=offs_s < NUM_SPLITS,
        other=float("-inf"),
    )
    w = tl.exp(lse - tl.max(lse, 0))
    w = w / tl.sum(w, 0)
    part = tl.load(
        PART_O
        + offs_s[:, None] * stride_ps
        + t * stride_pt
        + h * stride_ph
        + offs_l[None, :],
        mask=(w > 0.0)[:, None],
        other=0.0,
    )
    acc = tl.sum(part * w[:, None], 0)
    tl.store(O + t * stride_ot + h * stride_oh + offs_l, acc.to(O.dtype.element_ty))


def mla_absorbed_extend(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    qo_indptr: torch.Tensor,
    plan: AbsorbedExtendPlan,
    sm_scale: float,
    kv_lora_rank: int,
    kv_scale: float = 1.0,
) -> torch.Tensor:
    """o [T, H, kv_lora_rank] (bf16) for absorbed q [T, H, kv_lora_rank + rope]
    (fp8 e4m3, unit scale) over kv_cache [slots, kv_lora_rank + rope] (fp8 e4m3
    times kv_scale) whose kv_indices segments (kv_indptr) hold each request's
    prefix and then its extend tokens; causal on global positions. H <= MAX_HEADS."""
    total_q, num_heads, head_dim = q.shape
    rope = head_dim - kv_lora_rank
    splits = plan.num_splits
    part_o = q.new_empty(
        (splits, total_q, num_heads, kv_lora_rank), dtype=torch.float32
    )
    part_lse = q.new_empty((splits, total_q, num_heads), dtype=torch.float32)
    _absorbed_extend_kernel[(plan.block_req.numel(), splits)](
        q,
        kv_cache,
        kv_indices,
        kv_indptr,
        qo_indptr,
        plan.block_req,
        plan.block_q0,
        part_o,
        part_lse,
        sm_scale,
        kv_scale,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_lse.stride(0),
        part_lse.stride(1),
        H=num_heads,
        LATENT=kv_lora_rank,
        ROPE=rope,
        NUM_SPLITS=splits,
        BLOCK_Q=_BLOCK_Q,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    o = q.new_empty((total_q, num_heads, kv_lora_rank), dtype=torch.bfloat16)
    block_l = 128
    _combine_kernel[(total_q, num_heads, kv_lora_rank // block_l)](
        part_o,
        part_lse,
        o,
        part_o.stride(0),
        part_o.stride(1),
        part_o.stride(2),
        part_lse.stride(0),
        part_lse.stride(1),
        o.stride(0),
        o.stride(1),
        NUM_SPLITS=splits,
        SPLITS_P2=triton.next_power_of_2(splits),
        BLOCK_L=block_l,
        num_warps=4,
    )
    return o
