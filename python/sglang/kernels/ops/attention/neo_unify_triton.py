# SPDX-License-Identifier: Apache-2.0
# Adapted from ModelTC/LightLLM at fb7838ce86217a7c86625d71b15b72f2f05e0ac2:
# lightllm/models/neo_chat_moe/triton_kernel/context_attention_fwd_neo.py
# Continuous KV layout replaces LightLLM's request-to-token indirection.

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    K,
    V,
    E,
    O,
    QB: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    EB: tl.constexpr,
    ES: tl.constexpr,
    QLEN: tl.constexpr,
    KLEN: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    IMAGE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BD: tl.constexpr,
):
    b, h, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = block * M + tl.arange(0, M)
    cols = tl.arange(0, N)
    ds = tl.arange(0, BD)
    q = tl.load(
        Q + b * QB + rows[:, None] * QS + h * QH + ds[None, :] * QD,
        (rows[:, None] < QLEN) & (ds[None, :] < D),
        other=0,
    )
    end = tl.full((M,), 0, tl.int32)
    if IMAGE:
        end = tl.load(E + b * EB + rows * ES, rows < QLEN, other=0)
    stop = KLEN
    if CAUSAL:
        stop = tl.minimum(
            KLEN, tl.maximum(KLEN - QLEN + (block + 1) * M, tl.max(end, 0))
        )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.full((M,), 0.0, tl.float32)
    acc = tl.full((M, BD), 0.0, tl.float32)
    for start in range(0, stop, N):
        kp = start + cols
        k = tl.load(
            K + b * KB + kp[None, :] * KS + (h // GROUP) * KH + ds[:, None] * KD,
            (kp[None, :] < KLEN) & (ds[:, None] < D),
            other=0,
        )
        scores = tl.dot(q, k) * SCALE
        allow = kp[None, :] < KLEN
        if CAUSAL:
            allow = allow & (
                (kp[None, :] <= (KLEN - QLEN + rows[:, None]))
                | (kp[None, :] < end[:, None])
            )
        scores = tl.where(allow, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        p = tl.exp(scores - new_max[:, None])
        alpha = tl.exp(maximum - new_max)
        denominator = denominator * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            V + b * VB + kp[:, None] * VS + (h // GROUP) * VH + ds[None, :] * VD,
            (kp[:, None] < KLEN) & (ds[None, :] < D),
            other=0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        maximum = new_max
    result = acc / denominator[:, None]
    tl.store(
        O + ((b * QLEN + rows[:, None]) * HEADS + h) * D + ds[None, :],
        result,
        (rows[:, None] < QLEN) & (ds[None, :] < D),
    )


def neo_unify_attention_triton(q, k, v, image_token_end, causal, scale):
    b, s, h, d = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    end_strides = image_token_end.stride() if image_token_end is not None else (0, 0)
    _attention[(b, h, triton.cdiv(s, 64))](
        q,
        k,
        v,
        image_token_end,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *end_strides,
        s,
        k.shape[1],
        h,
        h // k.shape[2],
        d,
        scale,
        causal,
        image_token_end is not None,
        64,
        64,
        triton.next_power_of_2(d),
        num_warps=4,
        num_stages=2,
    )
    return out
