# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Paged KV readers. Batch, head and token work stays on the GPU.

Selection uses fixed-capacity tensors and GPU sequence lengths, so a captured
decode graph can replay across changing sequence lengths and request slots.
"""

import math

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.flashloop_quantization import QuantizedView, load_rows


@triton.jit
def _scores(
    Q,
    K,
    MIN,
    SCALE_KV,
    TAIL,
    FLAGS,
    REQ_MAP,
    REQ_IDS,
    LENS,
    SELECTED,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    KH_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    SELECT: tl.constexpr,
    BLOCK: tl.constexpr,
    QUANT: tl.constexpr,
    CAP: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    batch, head = row // H, row % H
    rank = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    length = tl.load(LENS + batch)
    logical = (
        tl.load(SELECTED + row * WIDTH + rank, rank < WIDTH, 0) if SELECT else rank
    )
    valid = (rank < WIDTH) & (logical < length)
    request = tl.load(REQ_IDS + batch)
    slot = tl.load(REQ_MAP + request * MAP_STRIDE + logical, valid, 0)
    if QUANT:
        keys = load_rows(
            K,
            MIN,
            SCALE_KV,
            TAIL,
            FLAGS,
            slot,
            logical,
            request,
            length,
            head,
            dims,
            valid,
            CAP,
            H,
            D,
            R,
            LOOP,
            True,
        )
    else:
        keys = tl.load(
            K + slot[:, None] * K_STRIDE + head * KH_STRIDE + dims[None, :],
            valid[:, None],
            0,
        ).to(tl.float32)
    query = tl.load(Q + row * D + dims).to(tl.float32)
    scores = tl.sum(keys * query[None, :], 1) * SCALE
    tl.store(
        OUT + row * WIDTH + rank, tl.where(valid, scores, -float("inf")), rank < WIDTH
    )


@triton.jit
def _weighted_values(
    V,
    MIN,
    SCALE_KV,
    TAIL,
    FLAGS,
    REQ_MAP,
    REQ_IDS,
    LENS,
    SELECTED,
    PROBS,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    VH_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    SELECT: tl.constexpr,
    QUANT: tl.constexpr,
    CAP: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head = row // H, row % H
    dims = tl.arange(0, D)
    request = tl.load(REQ_IDS + batch)
    length = tl.load(LENS + batch)
    rank = split * BLOCK + tl.arange(0, BLOCK)
    logical = (
        tl.load(SELECTED + row * WIDTH + rank, rank < WIDTH, 0) if SELECT else rank
    )
    prob = tl.load(PROBS + row * WIDTH + rank, rank < WIDTH, 0)
    valid = (rank < WIDTH) & (logical < length) & (prob != 0)
    slot = tl.load(REQ_MAP + request * MAP_STRIDE + logical, valid, 0)
    if QUANT:
        values = load_rows(
            V,
            MIN,
            SCALE_KV,
            TAIL,
            FLAGS,
            slot,
            logical,
            request,
            length,
            head,
            dims,
            valid,
            CAP,
            H,
            D,
            R,
            LOOP,
            False,
        )
    else:
        values = tl.load(
            V + slot[:, None] * V_STRIDE + head * VH_STRIDE + dims[None, :],
            valid[:, None],
            0,
        ).to(tl.float32)
    acc = tl.sum(values * prob[:, None], 0)
    tl.store(OUT + (row * tl.cdiv(WIDTH, BLOCK) + split) * D + dims, acc)


def paged_scores(query, keys, req_map, req_ids, lengths, width, selected=None):
    query = query.contiguous()
    batch, heads, dims = query.shape
    output = torch.empty(
        (batch, heads, width), device=query.device, dtype=torch.float32
    )
    quant = isinstance(keys, QuantizedView)
    pointer, lo, scale, tail, flags, cap, requests, loop, _ = (
        keys.args() if quant else (keys, keys, keys, keys, keys, 0, 0, 0, True)
    )
    ks, kh = (0, 0) if quant else keys.stride()[:2]
    _scores[(batch * heads, triton.cdiv(width, 32))](
        query,
        pointer,
        lo,
        scale,
        tail,
        flags,
        req_map,
        req_ids,
        lengths,
        selected if selected is not None else query,
        output,
        heads,
        dims,
        width,
        req_map.stride(0),
        ks,
        kh,
        dims**-0.5,
        selected is not None,
        32,
        quant,
        cap,
        requests,
        loop,
        enable_fp_fusion=False,
    )
    return output


def paged_weighted_values(values, req_map, req_ids, lengths, selected, probabilities):
    batch, heads, width = probabilities.shape
    dims = values.shape[-1]
    splits = triton.cdiv(width, 32)
    partial = torch.empty(
        (batch, heads, splits, dims), device=values.device, dtype=torch.float32
    )
    quant = isinstance(values, QuantizedView)
    pointer, lo, scale, tail, flags, cap, requests, loop, _ = (
        values.args()
        if quant
        else (values, values, values, values, values, 0, 0, 0, False)
    )
    vs, vh = (0, 0) if quant else values.stride()[:2]
    _weighted_values[(batch * heads, splits)](
        pointer,
        lo,
        scale,
        tail,
        flags,
        req_map,
        req_ids,
        lengths,
        selected if selected is not None else pointer,
        probabilities.contiguous(),
        partial,
        heads,
        dims,
        width,
        req_map.stride(0),
        vs,
        vh,
        32,
        selected is not None,
        quant,
        cap,
        requests,
        loop,
        enable_fp_fusion=False,
    )
    return partial.sum(dim=2)


def select_source(
    query,
    keys,
    values,
    req_map,
    req_ids,
    lengths,
    capacity,
    fraction,
    return_output=False,
):
    scores = paged_scores(query, keys, req_map, req_ids, lengths, capacity)
    width = max(1, math.ceil(capacity * fraction))
    _, selected = torch.topk(scores, width, dim=-1, sorted=True)
    # Retain exactly ceil(actual_length * fraction), even though graph capacity
    # is fixed. No .item(), .tolist(), per-request Python loop or CPU sync.
    budget = torch.ceil(lengths.float() * fraction).to(torch.long).clamp(min=1)
    valid = (
        torch.arange(width, device=query.device)[None, None, :] < budget[:, None, None]
    )
    valid = valid & (selected < lengths[:, None, None])
    full_probabilities = torch.softmax(scores, dim=-1)
    # SGLang may pad graph batches with zero-length requests.
    full_probabilities = torch.where(
        lengths[:, None, None] > 0, full_probabilities, 0.0
    )
    probabilities = full_probabilities.gather(-1, selected)
    probabilities = torch.where(valid, probabilities, 0.0)
    mass = probabilities.sum(-1, keepdim=True)
    selected_output = paged_weighted_values(
        values, req_map, req_ids, lengths, selected, probabilities
    )
    result = selected, valid, mass, selected_output
    if return_output:
        full_output = paged_weighted_values(
            values, req_map, req_ids, lengths, None, full_probabilities
        )
        result = (*result, full_output.to(query.dtype))
    return result


def correct_attention(query, keys, values, req_map, req_ids, lengths, source):
    selected, valid, mass, selected_output, source_output = source
    scores = paged_scores(
        query, keys, req_map, req_ids, lengths, selected.shape[-1], selected
    )
    scores = scores.masked_fill(~valid, -torch.inf)
    probabilities = torch.where(valid, torch.softmax(scores, dim=-1) * mass, 0.0)
    new_output = paged_weighted_values(
        values, req_map, req_ids, lengths, selected, probabilities
    )
    return (source_output.float() - selected_output + new_output).to(query.dtype)


def dense_paged_attention(query, keys, values, req_map, req_ids, lengths, capacity):
    scores = paged_scores(query, keys, req_map, req_ids, lengths, capacity)
    probabilities = torch.where(
        lengths[:, None, None] > 0, torch.softmax(scores, dim=-1), 0.0
    )
    return paged_weighted_values(
        values, req_map, req_ids, lengths, None, probabilities
    ).to(query.dtype)
