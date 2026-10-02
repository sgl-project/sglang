# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FlashLoop contributors
"""Nested active-token selection and compact causal attention over paged KV."""

import math

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.flashloop_quantization import QuantizedView, load_rows


def select_rows(current, previous, lengths, fraction, eligible=None):
    score = torch.linalg.vector_norm(current.float() - previous.float(), dim=-1)
    score /= torch.linalg.vector_norm(current.float(), dim=-1).clamp_min(1e-12)
    if eligible is not None:
        mask = torch.zeros(current.shape[0], device=current.device, dtype=torch.bool)
        mask[eligible] = True
        score = score.masked_fill(~mask, -torch.inf)
    indices, counts, starts = [], [], []
    offset, active_offset = 0, 0
    for length in lengths:  # request metadata is already on the CPU during prefill
        count = max(1, math.ceil(length * fraction))
        local = score[offset : offset + length].clone()
        local[-1] = torch.inf  # the last token must reach the LM head
        selected = torch.topk(local, count).indices.sort().values + offset
        indices.append(selected)
        counts.append(count)
        starts.append(active_offset)
        active_offset += count
        offset += length
    return (
        torch.cat(indices),
        torch.tensor(starts, dtype=torch.int32, device=current.device),
        torch.tensor(counts, dtype=torch.int32, device=current.device),
        max(counts),
    )


@triton.jit
def _prefill(
    Q,
    K,
    KM,
    KC,
    KT,
    KF,
    V,
    VM,
    VC,
    VT,
    VF,
    POS,
    MAP,
    REQ,
    LENGTHS,
    STARTS,
    COUNTS,
    OUT,
    H: tl.constexpr,
    D: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    MS: tl.constexpr,
    QM: tl.constexpr,
    KN: tl.constexpr,
    QUANT: tl.constexpr,
    CAP: tl.constexpr,
    R: tl.constexpr,
    LOOP: tl.constexpr,
):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = block * QM + tl.arange(0, QM)
    count = tl.load(COUNTS + batch)
    start = tl.load(STARTS + batch)
    channels = tl.arange(0, D)
    positions = tl.load(POS + start + rows, rows < count, 0)
    query = tl.load(
        Q + (start + rows[:, None]) * H * D + head * D + channels[None, :],
        rows[:, None] < count,
        0,
    )
    request = tl.load(REQ + batch)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((QM,), -float("inf"), tl.float32)
    denominator = tl.full((QM,), 0, tl.float32)
    accumulator = tl.full((QM, D), 0, tl.float32)
    for block_k in range(tl.cdiv(tl.minimum(length, tl.max(positions, 0) + 1), KN)):
        columns = block_k * KN + tl.arange(0, KN)
        slots = tl.load(MAP + request * MS + columns, columns < length, 0)
        if QUANT:
            keys = tl.trans(
                load_rows(
                    K,
                    KM,
                    KC,
                    KT,
                    KF,
                    slots,
                    columns,
                    request,
                    length,
                    head,
                    channels,
                    columns < length,
                    CAP,
                    H,
                    D,
                    R,
                    LOOP,
                    True,
                )
            ).to(query.dtype)
        else:
            keys = tl.load(
                K + slots[None, :] * KS + head * KH + channels[:, None],
                columns[None, :] < length,
                0,
            )
        logits = tl.dot(query, keys).to(tl.float32) * (D**-0.5)
        logits = tl.where(
            (columns[None, :] <= positions[:, None]) & (columns[None, :] < length),
            logits,
            -float("inf"),
        )
        next_max = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.exp(maximum - next_max)
        p = tl.exp(logits - next_max[:, None])
        denominator = denominator * correction + tl.sum(p, 1)
        if QUANT:
            values = load_rows(
                V,
                VM,
                VC,
                VT,
                VF,
                slots,
                columns,
                request,
                length,
                head,
                channels,
                columns < length,
                CAP,
                H,
                D,
                R,
                LOOP,
                False,
            ).to(query.dtype)
        else:
            values = tl.load(
                V + slots[:, None] * VS + head * VH + channels[None, :],
                columns[:, None] < length,
                0,
            )
        accumulator = accumulator * correction[:, None] + tl.dot(
            p.to(values.dtype), values
        )
        maximum = next_max
    result = accumulator / denominator[:, None]
    tl.store(
        OUT + (start + rows[:, None]) * H * D + head * D + channels[None, :],
        result,
        rows[:, None] < count,
    )


def sparse_paged_prefill(
    q, keys, values, positions, req_map, req_ids, lengths, starts, counts, max_count
):
    q = q.contiguous()
    output = torch.empty_like(q)
    heads, dims = q.shape[1:]
    quant = isinstance(keys, QuantizedView)
    if quant:
        kp, km, kc, kt, kf, cap, requests, loop, _ = keys.args()
        vp, vm, vc, vt, vf, *_ = values.args()
        ks = kh = vs = vh = 0
    else:
        kp = km = kc = kt = kf = keys
        vp = vm = vc = vt = vf = values
        ks, kh = keys.stride()[:2]
        vs, vh = values.stride()[:2]
        cap = requests = loop = 0
    _prefill[(triton.cdiv(max_count, 16), heads, req_ids.numel())](
        q,
        kp,
        km,
        kc,
        kt,
        kf,
        vp,
        vm,
        vc,
        vt,
        vf,
        positions,
        req_map,
        req_ids,
        lengths,
        starts,
        counts,
        output,
        heads,
        dims,
        ks,
        kh,
        vs,
        vh,
        req_map.stride(0),
        16,
        32 if quant else 64,
        quant,
        cap,
        requests,
        loop,
        num_stages=1 if quant else 3,
    )
    return output
