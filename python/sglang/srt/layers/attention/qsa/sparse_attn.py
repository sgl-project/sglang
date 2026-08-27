"""Validated sparse GQA operators migrated from the QSA reference branch.

``qsa_sparse_decode_triton`` is the package-independent decode path: it maps
logical QSA selections through ``req_to_token`` and reads the paged KV pool
directly, avoiding both the packed scratch copy and a flash-attn dependency.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _qsa_sparse_decode(
    q,
    k,
    v,
    out,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    sr_m: tl.constexpr,
    sr_n: tl.constexpr,
    si_m: tl.constexpr,
    si_n: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    req = tl.load(req_indices + row)
    seq_len = tl.load(seq_lens + row)

    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + row * sq_m
        + (kv_head * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, topk, BLOCK_N):
        cols = start + offs_n
        logical = tl.load(
            indices + row * si_m + cols * si_n,
            mask=cols < topk,
            other=-1,
        )
        valid = (cols < topk) & (logical >= 0) & (logical < seq_len)
        slots = tl.load(
            req_to_token + req * sr_m + tl.where(valid, logical, 0) * sr_n,
            mask=valid,
            other=0,
        )
        keys = tl.load(
            k + slots[None, :] * sk_n + kv_head * sk_h + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v + slots[:, None] * sv_n + kv_head * sv_h + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(
            valid[None, :],
            tl.dot(q_values, keys) * scale * 1.4426950408889634,
            -float("inf"),
        )
        has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
        block_max = tl.max(scores, axis=1)
        next_max = tl.where(has_values, tl.maximum(max_value, block_max), max_value)
        alpha = tl.where(has_values, tl.math.exp2(max_value - next_max), 1.0)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / normalizer[:, None],
        0.0,
    )
    tl.store(
        out
        + row * so_m
        + (kv_head * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


@triton.jit
def _qsa_sparse_decode_splitk(
    q,
    k,
    v,
    partial_out,
    partial_lse,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    sp_r: tl.constexpr,
    sp_k: tl.constexpr,
    sp_s: tl.constexpr,
    sp_h: tl.constexpr,
    sp_d: tl.constexpr,
    sl_r: tl.constexpr,
    sl_k: tl.constexpr,
    sl_s: tl.constexpr,
    sl_h: tl.constexpr,
    sr_m: tl.constexpr,
    sr_n: tl.constexpr,
    si_m: tl.constexpr,
    si_n: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    req = tl.load(req_indices + row)
    seq_len = tl.load(seq_lens + row)

    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + row * sq_m
        + (kv_head * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = q_values.to(tl.float32) * scale * 1.4426950408889634

    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    split_start = split * TOKENS_PER_SPLIT
    for offset in range(0, TOKENS_PER_SPLIT, BLOCK_N):
        cols = split_start + offset + offs_n
        logical = tl.load(
            indices + row * si_m + cols * si_n,
            mask=cols < topk,
            other=-1,
        )
        valid = (cols < topk) & (logical >= 0) & (logical < seq_len)
        slots = tl.load(
            req_to_token + req * sr_m + tl.where(valid, logical, 0) * sr_n,
            mask=valid,
            other=0,
        )
        keys = tl.load(
            k + slots[None, :] * sk_n + kv_head * sk_h + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v + slots[:, None] * sv_n + kv_head * sv_h + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(
            valid[None, :],
            tl.dot(q_values, keys.to(tl.float32), input_precision="tf32"),
            -float("inf"),
        )
        has_values = tl.sum(valid.to(tl.int32), axis=0) > 0
        block_max = tl.max(scores, axis=1)
        next_max = tl.where(has_values, tl.maximum(max_value, block_max), max_value)
        alpha = tl.where(has_values, tl.math.exp2(max_value - next_max), 1.0)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    partial = tl.where(
        normalizer[:, None] > 0,
        accumulator / normalizer[:, None],
        0.0,
    )
    lse = tl.where(
        normalizer > 0,
        max_value + tl.math.log2(normalizer),
        -float("inf"),
    )
    partial_offset = (
        row * sp_r
        + kv_head * sp_k
        + split * sp_s
        + offs_h[:, None] * sp_h
        + offs_d[None, :] * sp_d
    )
    tl.store(
        partial_out + partial_offset,
        partial,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )
    tl.store(
        partial_lse + row * sl_r + kv_head * sl_k + split * sl_s + offs_h * sl_h,
        lse,
        mask=offs_h < GROUP_SIZE,
    )


@triton.jit
def _qsa_sparse_decode_combine(
    partial_out,
    partial_lse,
    out,
    sp_r: tl.constexpr,
    sp_k: tl.constexpr,
    sp_s: tl.constexpr,
    sp_h: tl.constexpr,
    sp_d: tl.constexpr,
    sl_r: tl.constexpr,
    sl_k: tl.constexpr,
    sl_s: tl.constexpr,
    sl_h: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_mask = offs_h < GROUP_SIZE

    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    lse_base = row * sl_r + kv_head * sl_k + offs_h * sl_h
    for split in range(NUM_SPLITS):
        lse = tl.load(
            partial_lse + lse_base + split * sl_s,
            mask=head_mask,
            other=-float("inf"),
        )
        max_value = tl.maximum(max_value, lse)

    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    partial_base = (
        row * sp_r + kv_head * sp_k + offs_h[:, None] * sp_h + offs_d[None, :] * sp_d
    )
    for split in range(NUM_SPLITS):
        lse = tl.load(
            partial_lse + lse_base + split * sl_s,
            mask=head_mask,
            other=-float("inf"),
        )
        weight = tl.where(lse > -float("inf"), tl.math.exp2(lse - max_value), 0.0)
        partial = tl.load(
            partial_out + partial_base + split * sp_s,
            mask=head_mask[:, None],
            other=0.0,
        )
        accumulator += weight[:, None] * partial
        normalizer += weight

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / normalizer[:, None],
        0.0,
    )
    tl.store(
        out
        + row * so_m
        + (kv_head * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=head_mask[:, None],
    )


def qsa_sparse_decode_triton(
    q,
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    scale,
):
    """Sparse GQA decode over logical token indices and the live paged KV pool.

    All rows, including CUDA-graph padding rows, are launched uniformly. A row
    with no valid selected token returns zero without dereferencing the request
    table or KV cache through an invalid logical index.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must be rank-3 tensors")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("QSA Triton decode requires BF16 Q/K/V tensors")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("QSA Triton decode requires CUDA Q/K/V tensors")
    rows, num_q_heads, head_dim = q.shape
    if head_dim not in (128, 256):
        raise ValueError(
            f"QSA Triton decode supports head_dim 128 or 256, got {head_dim}"
        )
    if k.shape != v.shape or k.shape[2] != head_dim:
        raise ValueError("QSA Triton decode requires matching K/V cache shapes")
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("QSA query heads must be divisible by KV heads")
    if indices.ndim != 2 or indices.shape[0] != rows:
        raise ValueError("QSA decode indices must be [query_rows, topk]")
    if req_to_token.ndim != 2:
        raise ValueError("QSA req_to_token must be rank 2")
    if req_indices.numel() != rows or seq_lens.numel() != rows:
        raise ValueError(
            "QSA request indices and sequence lengths must match query rows"
        )
    if not all(
        tensor.is_cuda for tensor in (req_to_token, req_indices, indices, seq_lens)
    ):
        raise ValueError("QSA decode metadata must be CUDA tensors")

    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n = 64
    out = torch.empty_like(q)
    num_splits = 8 if rows <= 1 else 4 if rows <= 16 else 1
    if num_splits > 1:
        partial_size = rows * num_kv_heads * num_splits * group_size * head_dim
        lse_size = rows * num_kv_heads * num_splits * group_size
        workspace = torch.empty(
            partial_size + lse_size, dtype=torch.float32, device=q.device
        )
        partial_out = workspace[:partial_size].view(
            rows, num_kv_heads, num_splits, group_size, head_dim
        )
        partial_lse = workspace[partial_size:].view(
            rows, num_kv_heads, num_splits, group_size
        )
        tokens_per_split = (
            triton.cdiv(triton.cdiv(indices.shape[1], num_splits), block_n) * block_n
        )
        _qsa_sparse_decode_splitk[(rows, num_kv_heads, num_splits)](
            q,
            k,
            v,
            partial_out,
            partial_lse,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            scale,
            indices.shape[1],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_out.stride(4),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            partial_lse.stride(3),
            req_to_token.stride(0),
            req_to_token.stride(1),
            indices.stride(0),
            indices.stride(1),
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            TOKENS_PER_SPLIT=tokens_per_split,
            num_warps=8,
            num_stages=2,
        )
        _qsa_sparse_decode_combine[(rows, num_kv_heads)](
            partial_out,
            partial_lse,
            out,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_out.stride(4),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            partial_lse.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
            num_warps=4,
            num_stages=2,
        )
        return out

    _qsa_sparse_decode[(rows, num_kv_heads)](
        q,
        k,
        v,
        out,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        scale,
        indices.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        req_to_token.stride(0),
        req_to_token.stride(1),
        indices.stride(0),
        indices.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # Count one request per program. The previous implementation formed a
    # [next_power_of_2(topk), next_power_of_2(batch)] tensor in one program;
    # topk=2051 and batch=512 therefore exceeded Triton's 1M-element limit.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, for consumers that need per-row lengths but
    not the packed cu_seqlens prefix sum (trtllm paged decode packs rows at
    a fixed page-aligned stride instead)."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk
):
    _, heads, dim = k.shape
    block_topk = 16
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )


__all__ = [
    "qsa_sparse_decode_triton",
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
]
