# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

# Copied from https://code.alipay.com/pia/PainlessInferenceAcceleration/blob/v0.0.6/flood/flood/ops/seg_la.py

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


# arg `meta` of `seg_la_fwd` is SegLaMeta
@dataclass
class SegLaMeta:
    batch_size: int  # batch size, num of requests
    max_q_length: int  # max(seq_lens)
    q_offsets: torch.Tensor  # [bs+1], query_start_locations,
    s_offsets: torch.Tensor  # [bs], slot_ids
    q_lengths: torch.Tensor  # [bs], query length
    s_scales: torch.Tensor  # [bs], prefill = 0, decode = 1
    s_offsets_stride: int = 0
    q_offsets_stride: int = 0
    s_scales_stride: int = 0
    decay_scales_stride: int = 0
    mask: Optional[torch.Tensor] = None  # Currently not supported


# fused
@triton.jit
def seg_la_kernel(
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    DECOUPLE: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    # s_scale is 0 (prefill) or 1 (decode)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SPLIT_DIM)

    if s_offset == -1:
        return

    q_ptrs = (
        Q
        + q_offset * stride_q
        + hid * HEAD_DIM
        + (offs_b[:, None] * stride_q + offs_d[None, :])
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + (offs_b[:, None] * stride_k + offs_d[None, :])
    )
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + sid * SPLIT_DIM
        + (offs_b[:, None] * stride_v + offs_s[None, :])
    )
    out_ptrs = (
        Out
        + q_offset * stride_o
        + hid * HEAD_DIM
        + sid * SPLIT_DIM
        + (offs_b[:, None] * stride_o + offs_s[None, :])
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + sid * SPLIT_DIM
        + (offs_d[:, None] * HEAD_DIM + offs_s[None, :])
    )
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if BLOCK > 1:
        for n in range(0, q_length, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            if EVEN:
                q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
                k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
                v = tl.load(v_ptrs + n * stride_k).to(tl.float32)
            else:
                q = tl.load(
                    q_ptrs + n * stride_q,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)
                k = tl.trans(
                    tl.load(
                        k_ptrs + n * stride_k,
                        mask=(n + offs_b)[:, None] < q_length,
                        other=0.0,
                    )
                ).to(tl.float32)
                v = tl.load(
                    v_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                ).to(tl.float32)

            if DECOUPLE:
                # only work with small scales
                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                b_offs = b - 1 - offs_b

                edb = tl.exp(decay_scale * b_offs)
                decays = tl.where(b_offs >= 0, edb, 0)
                inv_decays = tl.where(b_offs >= 0, 1 / edb, 0)

                q = q * inv_decays[:, None]
                k = k * decays[None, :]
                qk = tl.dot(q, k) * softmax_scale
                qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
                o = tl.dot(qk, v)

                block_decay = tl.exp(decay_scale * b)
                block_decay_plus = block_decay * softmax_scale
                o = tl.dot(q, state) * block_decay_plus + o

                state = state * block_decay + tl.dot(k, v)
            else:

                qk = tl.dot(q, k) * softmax_scale
                decays = tl.exp(decay_scale * (offs_b[:, None] - offs_b[None, :]))
                decays = tl.where(offs_b[None, :] <= offs_b[:, None], decays, 0.0)
                qk *= decays
                o = tl.dot(qk, v)

                decay_arr = tl.exp(decay_scale * (offs_b[:, None] + 1)) * softmax_scale
                o = tl.dot(q * decay_arr, state, acc=o)

                if EVEN:
                    b = BLOCK
                else:
                    b = min(BLOCK, q_length - n)
                b_offs = b - 1 - offs_b
                b_offs = tl.where(b_offs >= 0, b_offs, 10000)
                decays = tl.exp(decay_scale * b_offs)
                block_decay = tl.exp(decay_scale * b)
                state = state * block_decay + tl.dot(k * decays[None, :], v)

            if EVEN:
                tl.store(out_ptrs + n * stride_o, o.to(Out.dtype.element_ty))
            else:
                tl.store(
                    out_ptrs + n * stride_o,
                    o.to(Out.dtype.element_ty),
                    mask=(n + offs_b)[:, None] < q_length,
                )

        tl.store(s_ptrs, state.to(S.dtype.element_ty))

    else:
        q = tl.trans(tl.load(q_ptrs)).to(tl.float32) * softmax_scale
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        state = state * tl.exp(decay_scale) + k * v

        o = tl.sum(q * state, axis=0, keep_dims=True)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))

        tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for prefilling
@triton.jit
def seg_la_p_kernel(
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    if s_offset == -1:
        return

    q_ptrs = (
        Q
        + q_offset * stride_q
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + q_offset * HEAD_DIM * H
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    for n in range(0, q_length, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        if EVEN:
            q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
            k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
            v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
            b = BLOCK
            b_offs = b - 1 - offs_b
            decays = tl.exp(decay_scale * b_offs)
            inv_decays = 1 / decays
        else:
            q = tl.load(
                q_ptrs + n * stride_q, mask=(n + offs_b)[:, None] < q_length, other=0.0
            ).to(tl.float32)
            k = tl.trans(
                tl.load(
                    k_ptrs + n * stride_k,
                    mask=(n + offs_b)[:, None] < q_length,
                    other=0.0,
                )
            ).to(tl.float32)
            v = tl.load(
                v_ptrs + n * stride_v, mask=(n + offs_b)[:, None] < q_length, other=0.0
            ).to(tl.float32)
            b = min(BLOCK, q_length - n)
            b_offs = b - 1 - offs_b
            block_decays = tl.exp(decay_scale * b_offs)
            decays = tl.where(b_offs >= 0, block_decays, 0)
            inv_decays = tl.where(b_offs >= 0, 1 / block_decays, 0)

        q = q * inv_decays[:, None]
        k = k * decays[None, :]
        qk = tl.dot(q, k) * softmax_scale
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
        o = tl.dot(qk, v)

        block_decay = tl.exp(decay_scale * b)
        o = tl.dot(q, state) * block_decay * softmax_scale + o

        state = state * block_decay + tl.dot(k, v)

        if EVEN:
            tl.store(out_ptrs + n * H * HEAD_DIM, o.to(Out.dtype.element_ty))
        else:
            tl.store(
                out_ptrs + n * H * HEAD_DIM,
                o.to(Out.dtype.element_ty),
                mask=(n + offs_b)[:, None] < q_length,
            )

    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for speculative
@triton.jit
def seg_la_s_kernel(
    Q,
    K,
    V,
    S,
    Out,
    Mask,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    q_offsets,
    q_lengths,
    s_scales,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_scale = tl.load(s_scales + bid)
    q_length = tl.load(q_lengths + bid)
    q_offset = tl.load(q_offsets + bid)
    s_offset = tl.load(s_offsets + bid)
    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    if s_offset == -1:
        return

    q_ptrs = (
        Q
        + q_offset * stride_q
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + q_offset * stride_k
        + hid * HEAD_DIM
        + kid * K_SPLIT_DIM
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + q_offset * stride_v
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + q_offset * HEAD_DIM * H
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_b[:, None] * H * HEAD_DIM + offs_v[None, :])
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs, mask=s_scale > 0).to(tl.float32)

    if EVEN:
        q = tl.load(q_ptrs).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs)).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)
        mask = tl.load(
            Mask
            + bid * BLOCK * BLOCK
            + tl.arange(0, BLOCK)[:, None] * BLOCK
            + tl.arange(0, BLOCK)[None, :]
        ).to(tl.int32)
        positions = tl.sum(mask, 1) - 1
        max_pos = tl.max(positions)
        b_offs = max_pos - positions
    else:
        q = tl.load(q_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        k = tl.trans(tl.load(k_ptrs, mask=offs_b[:, None] < q_length)).to(tl.float32)
        v = tl.load(v_ptrs, mask=offs_b[:, None] < q_length).to(tl.float32)
        mask = tl.load(
            Mask
            + bid * q_length * q_length
            + tl.arange(0, BLOCK)[:, None] * q_length
            + tl.arange(0, BLOCK)[None, :],
            mask=(tl.arange(0, BLOCK)[:, None] < q_length)
            & (tl.arange(0, BLOCK)[None, :] < q_length),
        ).to(tl.int32)
        positions = tl.sum(mask, 1) - 1
        max_pos = tl.max(positions)
        b_offs = max_pos - positions

    decays = tl.exp(decay_scale * b_offs)
    inv_decays = 1 / decays

    q = q * inv_decays[:, None]
    k = k * decays[None, :]
    qk = tl.dot(q, k) * softmax_scale
    qk = qk * mask.to(tl.float32)
    o = tl.dot(qk, v)

    block_decay = tl.exp(decay_scale * (max_pos + 1))
    o = tl.dot(q, state) * block_decay * softmax_scale + o

    if EVEN:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    else:
        tl.store(out_ptrs, o.to(Out.dtype.element_ty), mask=offs_b[:, None] < q_length)


# used for decode
@triton.jit
def seg_la_d_kernel(
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_o,
    s_offsets,
    decay_scales,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    # s_scale is 0 (first prefill chunk) or 1 (next prefill chunk)
    s_offset = tl.load(s_offsets + bid)
    if s_offset == -1:
        return

    decay_scale = -tl.load(decay_scales + hid)

    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    q_ptrs = Q + bid * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + bid * H * HEAD_DIM
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_v)
    )
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs).to(tl.float32)

    k = tl.load(k_ptrs).to(tl.float32)
    v = tl.load(v_ptrs).to(tl.float32)
    q = tl.load(q_ptrs).to(tl.float32) * softmax_scale

    state = state * tl.exp(decay_scale) + k[:, None] * v
    o = tl.sum(q[:, None] * state, axis=0)

    tl.store(out_ptrs, o.to(Out.dtype.element_ty))
    tl.store(s_ptrs, state.to(S.dtype.element_ty))


# used for MTP with only spec-topk=1.
@triton.jit
def seg_la_mtp_kernel(
    Q,
    K,
    V,
    S,
    CACHES,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_c,
    stride_o,
    s_offsets,
    cache_indices,
    decay_scales,
    step,
    HEAD_DIM: tl.constexpr,
    K_SPLIT_DIM: tl.constexpr,
    V_SPLIT_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = HEAD_DIM // V_SPLIT_DIM
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    s_offset = tl.load(s_offsets + bid)
    if s_offset == -1:
        return

    decay_scale = tl.exp(-tl.load(decay_scales + hid))

    offs_k = tl.arange(0, K_SPLIT_DIM)
    offs_v = tl.arange(0, V_SPLIT_DIM)

    # (length, qo_heads, d)
    q_ptrs = Q + bid * step * stride_q + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    k_ptrs = K + bid * step * stride_k + hid * HEAD_DIM + kid * K_SPLIT_DIM + (offs_k)
    v_ptrs = V + bid * step * stride_v + hid * HEAD_DIM + vid * V_SPLIT_DIM + (offs_v)
    # (num_dim_block, length, qo_heads, d)
    out_ptrs = (
        Out
        + kid * stride_o
        + bid * step * H * HEAD_DIM
        + hid * HEAD_DIM
        + vid * V_SPLIT_DIM
        + (offs_v)
    )
    # (bs, qo_heads, d, d)
    s_ptrs = (
        S
        + s_offset * stride_s
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )
    state = tl.load(s_ptrs).to(tl.float32)
    # (bs, step, kv_heads, d, d)
    cache_indices = tl.load(cache_indices + bid)
    c_ptrs = (
        CACHES
        + cache_indices * stride_c
        + hid * HEAD_DIM * HEAD_DIM
        + kid * HEAD_DIM * K_SPLIT_DIM
        + vid * V_SPLIT_DIM
        + (offs_k[:, None] * HEAD_DIM + offs_v[None, :])
    )

    for i in range(step):
        q = tl.load(q_ptrs).to(tl.float32) * softmax_scale
        k = tl.load(k_ptrs).to(tl.float32)
        v = tl.load(v_ptrs).to(tl.float32)

        state = state * decay_scale + k[:, None] * v
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(out_ptrs, o.to(Out.dtype.element_ty))
        tl.store(c_ptrs, state.to(CACHES.dtype.element_ty))
        q_ptrs += stride_q
        k_ptrs += stride_k
        v_ptrs += stride_v
        out_ptrs += H * HEAD_DIM
        c_ptrs += H * HEAD_DIM * HEAD_DIM


# (k_dim_block, length, qo_heads, d)
@triton.jit
def seg_la_sum_kernel(T, O, DIM: tl.constexpr, NUM_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    length = tl.num_programs(0)
    x = tl.zeros((DIM,), dtype=tl.float32)
    for i in range(NUM_BLOCK):
        x += tl.load(T + i * length * DIM + pid * DIM + tl.arange(0, DIM)).to(
            tl.float32
        )
    tl.store(O + pid * DIM + tl.arange(0, DIM), x)


def seg_la_fwd(
    q,
    k,
    v,
    s,
    decay_scales,
    meta,
    caches=None,
    cache_indices=None,
    softmax_scale=None,
    decouple=False,
):
    length, qo_heads, HEAD_DIM = q.shape
    _, kv_heads, _ = k.shape
    bs = meta.batch_size
    if softmax_scale is None:
        softmax_scale = HEAD_DIM ** (-0.5)

    # MAX_LENGTH = meta.max_q_length
    MAX_LENGTH = triton.cdiv(length, bs)

    assert qo_heads == kv_heads, "seg_la does NOT support GQA currently"

    if MAX_LENGTH > 1:
        # prefill with partitioning q/k/v
        # BLOCK should <= 64 with decouple
        K_SPLIT_DIM = 32
        V_SPLIT_DIM = 32 if bs <= 2 else 64

        num_warps = 2  # 2
        num_stages = 3  # 3

        k_dim_block = HEAD_DIM // K_SPLIT_DIM
        v_dim_block = HEAD_DIM // V_SPLIT_DIM
        tmp = torch.empty(
            (k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
        )
        grid = (bs, kv_heads, k_dim_block * v_dim_block)

        if caches is not None:
            # mtp
            EVEN = False
            BLOCK = 32
            step = length // bs

            seg_la_mtp_kernel[grid](
                q,
                k,
                v,
                s,
                caches,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                caches.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                cache_indices,
                decay_scales,
                step,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        elif meta.mask is not None:
            # spec
            ms = meta.mask.size(-1)
            BLOCK = (ms + 15) // 16 * 16
            EVEN = BLOCK == ms

            seg_la_s_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                meta.mask,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        else:
            # prefill
            BLOCK = 32
            EVEN = MAX_LENGTH % BLOCK == 0 if bs == 1 else False

            seg_la_p_kernel[grid](
                q,
                k,
                v,
                s,
                tmp,
                softmax_scale,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                s.stride(0),
                tmp.stride(0),
                meta.s_offsets,
                meta.q_offsets,
                meta.q_lengths,
                meta.s_scales,
                decay_scales,
                HEAD_DIM=HEAD_DIM,
                K_SPLIT_DIM=K_SPLIT_DIM,
                V_SPLIT_DIM=V_SPLIT_DIM,
                BLOCK=BLOCK,
                EVEN=EVEN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        if k_dim_block > 1:
            if length < 2048:
                o = tmp.sum(0)
            else:
                o = torch.empty(
                    (length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
                )
                seg_la_sum_kernel[(length,)](
                    tmp,
                    o,
                    DIM=qo_heads * HEAD_DIM,
                    NUM_BLOCK=k_dim_block,
                    num_warps=2,
                    num_stages=3,
                )
        else:
            o = tmp[0]

    else:
        # decode with partitioning q/k/v
        if bs <= 128:
            K_SPLIT_DIM = 128  # 128
            V_SPLIT_DIM = 32  # 32
            num_warps = 2  # 2
            num_stages = 2  # 3
        else:
            K_SPLIT_DIM = 128  # 128
            V_SPLIT_DIM = 64  # 32
            num_warps = 2  # 2
            num_stages = 3  # 3
        k_dim_block = HEAD_DIM // K_SPLIT_DIM
        v_dim_block = HEAD_DIM // V_SPLIT_DIM
        tmp = torch.empty(
            (k_dim_block, length, qo_heads, HEAD_DIM), device=q.device, dtype=q.dtype
        )
        grid = (bs, kv_heads, k_dim_block * v_dim_block)

        seg_la_d_kernel[grid](
            q,
            k,
            v,
            s,
            tmp,
            softmax_scale,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            s.stride(0),
            tmp.stride(0),
            meta.s_offsets,
            decay_scales,
            HEAD_DIM=HEAD_DIM,
            K_SPLIT_DIM=K_SPLIT_DIM,
            V_SPLIT_DIM=V_SPLIT_DIM,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if k_dim_block > 1:
            o = tmp.sum(0)
        else:
            o = tmp[0]

    # if fallback:
    #     # prefill/decode with partitioning v only
    #     o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    #     if MAX_LENGTH == 1:
    #         # decode
    #         BLOCK = 1
    #         EVEN = False
    #         SPLIT_DIM = 32
    #         num_warps = 8
    #         num_stages = 2
    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)
    #     else:
    #         # prefill
    #         if decouple:
    #             BLOCK = 64
    #             SPLIT_DIM = 16
    #         else:
    #             BLOCK = HEAD_DIM
    #             SPLIT_DIM = 32
    #         # EVEN = all([x % BLOCK == 0 for x in meta.qls])
    #         EVEN = False
    #         num_warps = 8
    #         num_stages = 2
    #         # prop = torch.cuda.get_device_properties(q.device.index)
    #         # arch = prop.major * 10 + prop.minor
    #         # if arch not in (80, 90):
    #         #     num_stages = 1

    #         num_dim_block = HEAD_DIM // SPLIT_DIM
    #         grid = (batch, kv_heads, num_dim_block)

    #     seg_la_kernel[grid](
    #         q,
    #         k,
    #         v,
    #         s,
    #         o,
    #         softmax_scale,
    #         q.stride(0),
    #         k.stride(0),
    #         v.stride(0),
    #         s.stride(0),
    #         o.stride(0),
    #         meta.s_offsets,
    #         meta.q_offsets,
    #         meta.q_lengths,
    #         meta.s_scales,
    #         decay_scales,
    #         HEAD_DIM=HEAD_DIM,
    #         SPLIT_DIM=SPLIT_DIM,
    #         BLOCK=BLOCK,
    #         EVEN=EVEN,
    #         DECOUPLE=decouple,
    #         num_warps=num_warps,
    #         num_stages=num_stages
    #     )
    return o
