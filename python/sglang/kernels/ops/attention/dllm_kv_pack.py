# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Pack a CSR prefix and current K/V into capture-stable dense storage."""

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
    KPS: tl.constexpr,
    VPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    qs = tl.load(QO + row).to(tl.int64)
    qe = tl.load(QO + row + 1).to(tl.int64)
    ps = tl.load(KI + row).to(tl.int64)
    pe = tl.load(KI + row + 1).to(tl.int64)
    prefix = pe - ps
    length = prefix + qe - qs
    start = ps + qs
    offsets = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
    token, head, channel = offsets // (H * D), offsets // D % H, offsets % D
    slot = tl.load(IDS + ps + token, token < prefix, other=0).to(tl.int64)
    if PAGE_SIZE > 1:
        ko = (slot // PAGE_SIZE) * KPS + (slot % PAGE_SIZE) * KBS
        vo = (slot // PAGE_SIZE) * VPS + (slot % PAGE_SIZE) * VBS
    else:
        ko = slot * KBS
        vo = slot * VBS
    kp = tl.load(KB + ko + head * KBH + channel, token < prefix, other=0)
    vp = tl.load(VB + vo + head * VBH + channel, token < prefix, other=0)
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


def pack_prefix_current(
    k, v, kb, vb, qo, ki, ids, kd, vd, cu, max_length, *, page_size=1
):
    h, d = k.shape[1:]
    if page_size > 1:
        if (
            kb.ndim != 4
            or vb.ndim != 4
            or kb.shape[1:] != (h, page_size, d)
            or vb.shape[1:] != (h, page_size, d)
            or kb.stride(-1) != 1
            or vb.stride(-1) != 1
        ):
            raise ValueError("Paged prefix caches must have HND layout")
        kbs, vbs = kb.stride(2), vb.stride(2)
    else:
        kbs, vbs = kb.stride(0), vb.stride(0)
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
        kbs,
        kb.stride(1),
        vbs,
        vb.stride(1),
        kb.stride(0),
        vb.stride(0),
        page_size,
        TILE=1024,
    )
