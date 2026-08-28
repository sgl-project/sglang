"""Native FP8 sparse MLA prefill for NVIDIA Blackwell (SM100).

This is the SM100 counterpart of ``sparse_mla_q8kv8_prefill_sm90``.  The
kernel keeps Q and KV in E4M3 for both tensor-core matrix products while the
online-softmax state and output accumulator stay in FP32:

* QK: E4M3 x E4M3 -> FP32
* PV: scaled E4M3 probabilities x E4M3 values -> FP32

The probability tile is multiplied by ``P_FP8_SCALE`` before its E4M3 cast
and the dot result is divided by the same value.  This preserves small
probabilities at the large top-k values used by DeepSeek-V4-Flash without
changing the represented attention result.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950408889634)
_P_FP8_SCALE = tl.constexpr(256.0)


@triton.jit
def _q8kv8_load_kv_tile(
    kv,
    token_ids,
    valid,
    offs_d,
    d_qk: tl.constexpr,
):
    ptrs = kv + token_ids[:, None] * d_qk + offs_d[None, :]
    return tl.load(ptrs, mask=valid[:, None], other=0.0)


@triton.jit
def _q8kv8_sparse_prefill_kernel(
    q,
    kv,
    indices,
    q_scale,
    kv_scale,
    attn_sink,
    topk_length,
    out,
    max_logits,
    lse,
    s_kv,
    h_q: tl.constexpr,
    d_qk: tl.constexpr,
    topk,
    sm_scale,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_idx = tl.program_id(0)
    h_block = tl.program_id(1)

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < h_q
    offs_d = tl.arange(0, 64)

    q_base = q + q_idx * h_q * d_qk + offs_h[:, None] * d_qk

    m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    if d_qk == 512:
        acc = tl.zeros([BLOCK_H, 512], tl.float32)
    else:
        acc0 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc1 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc2 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc3 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc4 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc5 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc6 = tl.zeros([BLOCK_H, 64], tl.float32)
        acc7 = tl.zeros([BLOCK_H, 64], tl.float32)

    qk_scale = tl.load(q_scale) * tl.load(kv_scale) * sm_scale
    valid_topk = tl.load(topk_length + q_idx) if HAVE_TOPK_LENGTH else topk

    for start_n in range(0, topk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        token_ids = tl.load(
            indices + q_idx * topk + offs_n,
            mask=offs_n < topk,
            other=-1,
        )
        valid = (offs_n < valid_topk) & (token_ids >= 0) & (token_ids < s_kv)
        safe_token_ids = tl.where(valid, token_ids, 0)

        if d_qk == 512:
            offs_d_full = tl.arange(0, 512)
            q_full = tl.load(
                q_base + offs_d_full[None, :],
                mask=h_mask[:, None],
                other=0.0,
            )
            kv_full = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d_full, d_qk)
            scores = tl.dot(q_full, tl.trans(kv_full), out_dtype=tl.float32)
        else:
            scores = tl.zeros([BLOCK_H, BLOCK_N], tl.float32)
            q0 = tl.load(
                q_base + (offs_d + 0)[None, :], mask=h_mask[:, None], other=0.0
            )
            k0 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 0, d_qk)
            scores += tl.dot(q0, tl.trans(k0), out_dtype=tl.float32)
            q1 = tl.load(
                q_base + (offs_d + 64)[None, :], mask=h_mask[:, None], other=0.0
            )
            k1 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 64, d_qk)
            scores += tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
            q2 = tl.load(
                q_base + (offs_d + 128)[None, :], mask=h_mask[:, None], other=0.0
            )
            k2 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 128, d_qk)
            scores += tl.dot(q2, tl.trans(k2), out_dtype=tl.float32)
            q3 = tl.load(
                q_base + (offs_d + 192)[None, :], mask=h_mask[:, None], other=0.0
            )
            k3 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 192, d_qk)
            scores += tl.dot(q3, tl.trans(k3), out_dtype=tl.float32)
            q4 = tl.load(
                q_base + (offs_d + 256)[None, :], mask=h_mask[:, None], other=0.0
            )
            k4 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 256, d_qk)
            scores += tl.dot(q4, tl.trans(k4), out_dtype=tl.float32)
            q5 = tl.load(
                q_base + (offs_d + 320)[None, :], mask=h_mask[:, None], other=0.0
            )
            k5 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 320, d_qk)
            scores += tl.dot(q5, tl.trans(k5), out_dtype=tl.float32)
            q6 = tl.load(
                q_base + (offs_d + 384)[None, :], mask=h_mask[:, None], other=0.0
            )
            k6 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 384, d_qk)
            scores += tl.dot(q6, tl.trans(k6), out_dtype=tl.float32)
            q7 = tl.load(
                q_base + (offs_d + 448)[None, :], mask=h_mask[:, None], other=0.0
            )
            k7 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 448, d_qk)
            scores += tl.dot(q7, tl.trans(k7), out_dtype=tl.float32)
            q8 = tl.load(
                q_base + (offs_d + 512)[None, :], mask=h_mask[:, None], other=0.0
            )
            k8 = _q8kv8_load_kv_tile(kv, safe_token_ids, valid, offs_d + 512, d_qk)
            scores += tl.dot(q8, tl.trans(k8), out_dtype=tl.float32)

        scores *= qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        block_max = tl.where(block_max == float("-inf"), -1.0e30, block_max)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.where(
            m_i == float("-inf"),
            0.0,
            tl.math.exp2((m_i - m_new) * _LOG2E),
        )
        p = tl.where(
            valid[None, :] & h_mask[:, None],
            tl.math.exp2((scores - m_new[:, None]) * _LOG2E),
            0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Scaling before the E4M3 cast avoids losing probabilities around
        # 1/topk while keeping the largest possible value (256) representable.
        p_fp8 = (p * _P_FP8_SCALE).to(tl.float8e4nv)
        pv_scale = 1.0 / _P_FP8_SCALE

        if d_qk == 512:
            acc = (
                acc * alpha[:, None]
                + tl.dot(p_fp8, kv_full, out_dtype=tl.float32) * pv_scale
            )
        else:
            acc0 = (
                acc0 * alpha[:, None]
                + tl.dot(p_fp8, k0, out_dtype=tl.float32) * pv_scale
            )
            acc1 = (
                acc1 * alpha[:, None]
                + tl.dot(p_fp8, k1, out_dtype=tl.float32) * pv_scale
            )
            acc2 = (
                acc2 * alpha[:, None]
                + tl.dot(p_fp8, k2, out_dtype=tl.float32) * pv_scale
            )
            acc3 = (
                acc3 * alpha[:, None]
                + tl.dot(p_fp8, k3, out_dtype=tl.float32) * pv_scale
            )
            acc4 = (
                acc4 * alpha[:, None]
                + tl.dot(p_fp8, k4, out_dtype=tl.float32) * pv_scale
            )
            acc5 = (
                acc5 * alpha[:, None]
                + tl.dot(p_fp8, k5, out_dtype=tl.float32) * pv_scale
            )
            acc6 = (
                acc6 * alpha[:, None]
                + tl.dot(p_fp8, k6, out_dtype=tl.float32) * pv_scale
            )
            acc7 = (
                acc7 * alpha[:, None]
                + tl.dot(p_fp8, k7, out_dtype=tl.float32) * pv_scale
            )
        m_i = m_new

    denominator = l_i
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink + offs_h, mask=h_mask, other=float("-inf"))
        denominator += tl.math.exp2((sink - m_i) * _LOG2E)

    nonempty = l_i > 0.0
    inv_denom = tl.where(denominator > 0.0, 1.0 / denominator, 0.0)
    value_scale = tl.load(kv_scale) * inv_denom

    out_base = out + q_idx * h_q * 512 + offs_h[:, None] * 512
    if d_qk == 512:
        tl.store(
            out_base + tl.arange(0, 512)[None, :],
            acc * value_scale[:, None],
            mask=h_mask[:, None],
        )
    else:
        tl.store(
            out_base + (offs_d + 0)[None, :],
            acc0 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 64)[None, :],
            acc1 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 128)[None, :],
            acc2 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 192)[None, :],
            acc3 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 256)[None, :],
            acc4 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 320)[None, :],
            acc5 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 384)[None, :],
            acc6 * value_scale[:, None],
            mask=h_mask[:, None],
        )
        tl.store(
            out_base + (offs_d + 448)[None, :],
            acc7 * value_scale[:, None],
            mask=h_mask[:, None],
        )

    meta_ptrs = q_idx * h_q + offs_h
    tl.store(max_logits + meta_ptrs, tl.where(nonempty, m_i, -1.0e30), mask=h_mask)
    tl.store(
        lse + meta_ptrs,
        tl.where(nonempty, m_i + tl.log(l_i), float("-inf")),
        mask=h_mask,
    )


def sparse_mla_q8kv8_prefill_fwd_sm100(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """Launch the validated SM100 implementation into caller-owned buffers."""
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    # The production path uses the dedicated CUDA/tcgen05 kernel. Keep the
    # Triton implementation below as a bring-up fallback for unpadded head
    # counts while the CUDA kernel intentionally matches FlashMLA's head64
    # specialization.
    if h_q % 32 == 0 and os.environ.get("SGLANG_SM100_Q8KV8_TRITON", "0") != "1":
        from sglang.kernels.ops.attention.sparse_mla_q8kv8_prefill_sm100_cuda import (
            sparse_mla_q8kv8_prefill_fwd_sm100_cuda,
        )

        sparse_mla_q8kv8_prefill_fwd_sm100_cuda(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
            max_logits=max_logits,
            lse=lse,
        )
        return
    # Unlike the SM90 kernel, SM100 does not require TP-local Q heads to be
    # padded to 64.  DeepSeek-V4-Flash TP8 has only 8 active heads per rank;
    # using the smallest legal 32-row tcgen05 tile avoids half of the redundant
    # work of the SM90-compatible 64-row shape.
    if d_qk == 512:
        # Environment overrides are intentionally limited to tile-selection
        # knobs so B200 experiments can compare compiler specializations
        # without editing kernel math between runs. Invalid combinations fail
        # during compilation instead of silently changing semantics.
        block_h = int(os.environ.get("SGLANG_SM100_Q8KV8_BLOCK_H", "32"))
        block_n = int(os.environ.get("SGLANG_SM100_Q8KV8_BLOCK_N", "256"))
        num_warps = int(os.environ.get("SGLANG_SM100_Q8KV8_NUM_WARPS", "8"))
        num_stages = int(os.environ.get("SGLANG_SM100_Q8KV8_NUM_STAGES", "1"))
        if block_h not in (8, 16, 32, 64) or block_n not in (64, 128, 256):
            raise ValueError(
                "SM100 Q8KV8 tile overrides require BLOCK_H in {8,16,32,64} and "
                f"BLOCK_N in {{64,128,256}}, got {block_h}/{block_n}"
            )
    else:
        block_h = 32
        block_n = 64
        num_warps = 8
        num_stages = 2
    grid = (s_q, triton.cdiv(h_q, block_h))

    # Optional pointers are represented by any valid CUDA pointer when their
    # corresponding constexpr flag is false; the kernel will not dereference
    # them in that specialization.
    sink_arg = attn_sink if attn_sink is not None else max_logits
    length_arg = topk_length if topk_length is not None else indices
    _q8kv8_sparse_prefill_kernel[grid](
        q,
        kv,
        indices,
        q_scale,
        kv_scale,
        sink_arg,
        length_arg,
        out,
        max_logits,
        lse,
        s_kv,
        h_q=h_q,
        d_qk=d_qk,
        topk=topk,
        sm_scale=sm_scale,
        HAVE_ATTN_SINK=attn_sink is not None,
        HAVE_TOPK_LENGTH=topk_length is not None,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
