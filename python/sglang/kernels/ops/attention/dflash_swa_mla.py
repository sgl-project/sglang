# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Absorbed-MLA value-column split with native sequential softmax arithmetic.

Each CTA owns 64, 128, or 256 of the 512 output columns and repeats the
full native M16/N64 QK and softmax sweep. Prefix and current-token loops remain
separate. Each output column has the same sequential accumulator recurrence;
there is no partial-result merge. This aims to preserve native BF16 arithmetic,
but compiler-dependent dot/FMA lowering still requires numerical qualification.

``num_splits`` partitions value columns, not keys: 8 means 64 columns per CTA,
4 means 128, and 2 means 256. No numerical scratch is required. The opt-in
backend route is restricted to measured GB300 batch/head combinations.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_mla_value_split(
    Q,
    K,
    V,
    KB,
    VB,
    QO,
    KV,
    INDICES,
    OUT,
    qts,
    qhs,
    kts,
    vts,
    kbts,
    vbts,
    ots,
    ohs,
    sm_scale,
    k_scale,
    v_scale,
    BLOCK_DV: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    value_slice = tl.program_id(2)
    q_start = tl.load(QO + batch)
    q_len = tl.load(QO + batch + 1) - q_start
    if q_len == 0:
        return
    kv_start = tl.load(KV + batch)
    prefix = tl.load(KV + batch + 1) - kv_start
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 64)
    dims = tl.arange(0, 512)
    rope = 512 + tl.arange(0, 64)
    value_dims = value_slice * BLOCK_DV + tl.arange(0, BLOCK_DV)
    row_valid = (rows < q_len) & (rows < 8)
    q_base = (q_start + rows) * qts + head * qhs
    q = tl.load(Q + q_base[:, None] + dims[None, :], row_valid[:, None], 0.0)
    qr = tl.load(Q + q_base[:, None] + rope[None, :], row_valid[:, None], 0.0)
    acc = tl.zeros((16, BLOCK_DV), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    maximum = tl.zeros((16,), tl.float32) - float("inf")

    # Match native prefix tiles and recurrence exactly; no KV partitioning.
    for start_n in range(0, prefix, 64):
        start_n = tl.multiple_of(start_n, 64)
        columns = start_n + cols
        valid = columns < prefix
        legal = row_valid[:, None] & valid[None, :]
        legal &= prefix + rows[:, None] <= columns[None, :] + 4095
        skip = tl.max(tl.max(legal.to(tl.int32), axis=1), axis=0) == 0
        if not skip:
            slots = tl.load(INDICES + kv_start + columns, valid, 0)
            k = tl.load(KB + slots[None, :] * kbts + dims[:, None], valid[None, :], 0.0)
            scores = tl.dot(q.to(k.dtype), k)
            kr = tl.load(
                KB + slots[None, :] * kbts + rope[:, None], valid[None, :], 0.0
            )
            scores += tl.dot(qr.to(kr.dtype), kr)
            scores *= sm_scale * k_scale
            scores = tl.where(legal, scores, -float("inf"))
            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1.0e20, row_max)
            next_maximum = tl.maximum(row_max_fixed, maximum)
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum[:, None])
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            values = tl.load(
                VB + slots[:, None] * vbts + value_dims[None, :], valid[:, None], 0.0
            )
            dot = tl.dot(probabilities.to(values.dtype), values)
            acc = acc * correction[:, None] + dot * v_scale
            maximum = next_maximum

    # Native starts a fresh 64-column current-token tile even after a short
    # prefix tail. This is essential to preserve the BF16 P-cast basis.
    for start_n in range(0, q_len, 64):
        start_n = tl.multiple_of(start_n, 64)
        columns = start_n + cols
        valid = columns < q_len
        legal = row_valid[:, None] & valid[None, :]
        legal &= rows[:, None] >= columns[None, :]
        legal &= rows[:, None] <= columns[None, :] + 4095
        skip = tl.max(tl.max(legal.to(tl.int32), axis=1), axis=0) == 0
        if not skip:
            k = tl.load(
                K + (q_start + columns[None, :]) * kts + dims[:, None],
                valid[None, :],
                0.0,
            )
            scores = tl.dot(q, k, out_dtype=tl.float32)
            kr = tl.load(
                K + (q_start + columns[None, :]) * kts + rope[:, None],
                valid[None, :],
                0.0,
            )
            scores += tl.dot(qr, kr)
            scores *= sm_scale
            scores = tl.where(legal, scores, -float("inf"))
            row_max = tl.max(scores, axis=1)
            row_max_fixed = tl.where(row_max == -float("inf"), -1.0e20, row_max)
            next_maximum = tl.maximum(row_max_fixed, maximum)
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum[:, None])
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            values = tl.load(
                V + (q_start + columns[:, None]) * vts + value_dims[None, :],
                valid[:, None],
                0.0,
            )
            dot = tl.dot(probabilities.to(values.dtype), values)
            acc = acc * correction[:, None] + dot
            maximum = next_maximum

    result = acc / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
    tl.store(
        OUT + (q_start + rows[:, None]) * ots + head * ohs + value_dims[None, :],
        result,
        row_valid[:, None],
    )


def dflash_swa_mla_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    sm_scale,
    sliding_window_size=4095,
    num_splits=8,
    page_size=1,
    max_len_extend=8,
):
    """Write causal BF16 absorbed-MLA SWA attention to ``o_extend``.

    The caller owns valid CSR metadata: each query sequence has at most eight
    rows; prefix indices address physical slots and already select the suffix
    window. Device metadata is never copied to the host or synchronized here.
    Buffer strides support both contiguous V and an aliased latent V view of K.
    As in native Triton, page_size describes a flat3-D physical-token-slot view.
    Unsupported shapes/dtypes fail instead of silently changing semantics.
    """
    if max_len_extend != 8 or sliding_window_size != 4095:
        raise ValueError("experimental path requires block8 and window_left4095")
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    if page_size < 1 or page_size & (page_size - 1):
        raise ValueError("page_size must be a positive power of two")
    if num_splits not in (2, 4, 8):
        raise ValueError("num_splits must be 2 (256), 4 (128), or 8 (64 value columns)")
    if qo_indptr.ndim != 1 or kv_indptr.shape != qo_indptr.shape:
        raise ValueError("query and prefix indptr must have the same1-D shape")
    batch = qo_indptr.numel() - 1
    if batch < 1 or q_extend.ndim != 3 or q_extend.shape[1] < 1:
        raise ValueError("expected positive batch/heads and3-D Q")
    tokens, heads, width = q_extend.shape
    if width != 576 or tokens > batch * 8:
        raise ValueError("expected absorbed Q width576 and at most8 rows per sequence")
    expected = ((tokens, 1, 576), (tokens, 1, 512), (tokens, heads, 512))
    if tuple(x.shape for x in (k_extend, v_extend, o_extend)) != expected:
        raise ValueError("expected one KV head, QK576 and V/output512")
    if k_buffer.ndim != 3 or tuple(k_buffer.shape[1:]) != (1, 576):
        raise ValueError("K cache must be [physical slots, 1, 576]")
    if tuple(v_buffer.shape) != (k_buffer.shape[0], 1, 512):
        raise ValueError("V cache must be [physical slots, 1, 512]")
    for tensor in (q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer):
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or tensor.device != q_extend.device
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all Q/K/V/output tensors must be same-device CUDA BF16, contiguous in the last dimension"
            )
    for tensor in (qo_indptr, kv_indptr, kv_indices):
        if (
            tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.device != q_extend.device
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "CSR metadata must be contiguous same-device int32/int64 vectors"
            )
    _swa_mla_value_split[(batch, heads, num_splits)](
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        o_extend,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        v_extend.stride(0),
        k_buffer.stride(0),
        v_buffer.stride(0),
        o_extend.stride(0),
        o_extend.stride(1),
        sm_scale,
        1.0,
        1.0,
        BLOCK_DV=512 // num_splits,
        num_warps=8,
        num_stages=1,
    )
    return o_extend
