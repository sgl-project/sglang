# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Pack a CSR prefix and current K/V into capture-stable dense storage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kv(
    K,
    V,
    KB,
    VB,
    QO,
    KI,
    IDS,
    KD,
    VD,
    CU,
    H: tl.constexpr,
    D: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    KBS: tl.constexpr,
    KBH: tl.constexpr,
    VBS: tl.constexpr,
    VBH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    qs, qe = tl.load(QO + row), tl.load(QO + row + 1)
    ps, pe = tl.load(KI + row), tl.load(KI + row + 1)
    prefix = pe - ps
    length = prefix + qe - qs
    start = ps + qs
    offsets = tl.program_id(1) * TILE + tl.arange(0, TILE)
    token, head, channel = offsets // (H * D), offsets // D % H, offsets % D
    slot = tl.load(IDS + ps + token, token < prefix, other=0)
    kp = tl.load(KB + slot * KBS + head * KBH + channel, token < prefix, other=0)
    vp = tl.load(VB + slot * VBS + head * VBH + channel, token < prefix, other=0)
    current = qs + token - prefix
    kc = tl.load(
        K + current * KS + head * KH + channel,
        (token >= prefix) & (token < length),
        other=0,
    )
    vc = tl.load(
        V + current * VS + head * VH + channel,
        (token >= prefix) & (token < length),
        other=0,
    )
    tl.store(
        KD + start * H * D + offsets, tl.where(token < prefix, kp, kc), token < length
    )
    tl.store(
        VD + start * H * D + offsets, tl.where(token < prefix, vp, vc), token < length
    )
    if tl.program_id(1) == 0:
        tl.store(CU + row, start)
        if row == tl.num_programs(0) - 1:
            tl.store(CU + row + 1, pe + qe)


def pack_prefix_current(k, v, kb, vb, qo, ki, ids, kd, vd, cu, max_length):
    h, d = k.shape[1:]
    _pack_kv[(qo.numel() - 1, triton.cdiv(max_length * h * d, 1024))](
        k,
        v,
        kb,
        vb,
        qo,
        ki,
        ids,
        kd,
        vd,
        cu,
        h,
        d,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kb.stride(0),
        kb.stride(1),
        vb.stride(0),
        vb.stride(1),
        TILE=1024,
    )


class DenseKVWorkspace:
    def __init__(self, capacity, heads, dim, batch_size, device, dtype):
        # TMA loads can include the masked tail beyond the packed sequences.
        self.key = torch.zeros((capacity, heads, dim), device=device, dtype=dtype)
        self.value = torch.zeros_like(self.key)
        self.cu_seqlens = torch.empty(batch_size + 1, device=device, dtype=torch.int32)
        self.indices = torch.arange(capacity, device=device, dtype=torch.int64)
        self.window_start = torch.zeros(batch_size, device=device, dtype=torch.int32)
