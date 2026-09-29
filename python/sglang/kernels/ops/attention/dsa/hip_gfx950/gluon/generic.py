"""Generic gfx950 Gluon schedule for full indexer preparation."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)


@gluon.jit
def _projection_slab(
    X,
    W,
    acc,
    am,
    bn,
    start,
    M: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    BK: gl.constexpr,
):
    a_layout: gl.constexpr = gl.DotOperandLayout(0, acc.type.layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, acc.type.layout, 8)
    ak = start + gl.arange(0, BK, gl.SliceLayout(0, a_layout))
    bk = start + gl.arange(0, BK, gl.SliceLayout(1, b_layout))
    a_mask = True
    if M % acc.type.shape[0] != 0:
        a_mask = am[:, None] < M
    a_offsets = am[:, None] * SX + ak[None, :]
    b_offsets = bn[None, :] * SW + bk[:, None]
    a = gl.amd.cdna4.buffer_load(X, a_offsets, a_mask, 0)
    b = gl.amd.cdna4.buffer_load(W, b_offsets)
    return gl.amd.cdna4.mfma(a, b, acc)


@gluon.jit
def _key_gate_tile(
    X,
    W,
    Y,
    M: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    OUT_N: gl.constexpr,
    BM: gl.constexpr,
    tile_n,
    split,
):
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    am = gl.program_id(1) * 16 + gl.arange(0, BM, gl.SliceLayout(1, a_layout))
    bn = tile_n * 16 + gl.arange(0, 16, gl.SliceLayout(0, b_layout))
    acc = gl.zeros((BM, 16), gl.float32, mma_layout)
    acc = _projection_slab(X, W, acc, am, bn, split * 384, M, SX, SW, 256)
    acc = _projection_slab(X, W, acc, am, bn, split * 384 + 256, M, SX, SW, 128)
    m = gl.program_id(1) * 16 + gl.arange(0, BM, gl.SliceLayout(1, mma_layout))
    n = tile_n * 16 + gl.arange(0, 16, gl.SliceLayout(0, mma_layout))
    mask = True
    if M % BM != 0:
        mask = m[:, None] < M
    if OUT_N == 128:
        dst = ((split // 4 * M + m[:, None]) * 128 + n[None, :]) * 4 + split % 4
    else:
        dst = (split * M + m[:, None]) * OUT_N + n[None, :]
    gl.store(Y + dst, acc, mask)


@gluon.jit
def _stage_m1_query(A, B, X, W, ao, bo, mask):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(A, X, ao, mask, 0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(B, W, bo)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _pipelined_m1_query(
    X,
    W,
    Y,
    SX: gl.constexpr,
    SW: gl.constexpr,
    OUT_N: gl.constexpr,
    K: gl.constexpr,
    tile,
):
    BK: gl.constexpr = 512
    STAGES: gl.constexpr = 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [64, 1], [1, 1], [0, 1])
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    a_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [1, BK], [1, 0]
    )
    b_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BK, 8], [0, 1]
    )
    a0 = gl.allocate_shared_memory(X.dtype.element_ty, [1, BK], a_shared)
    a1 = gl.allocate_shared_memory(X.dtype.element_ty, [1, BK], a_shared)
    b0 = gl.allocate_shared_memory(W.dtype.element_ty, [BK, 8], b_shared)
    b1 = gl.allocate_shared_memory(W.dtype.element_ty, [BK, 8], b_shared)
    am = gl.arange(0, 1, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bn = gl.arange(0, 8, gl.SliceLayout(0, bl))
    W = W + tile * 8 * SW
    ao = am[:, None] * SX + ak[None, :]
    bo = bn[None, :] * SW + bk[:, None]
    mask = am[:, None] < 1
    _stage_m1_query(a0, b0, X, W, ao, bo, mask)
    _stage_m1_query(a1, b1, X, W, ao + BK, bo + BK, mask)
    acc = gl.zeros((1, 8), gl.float32, mma)
    for step in gl.static_range(K // BK):
        gl.amd.cdna4.async_copy.wait_group(
            STAGES - 1 if step < K // BK - STAGES + 1 else K // BK - step - 1
        )
        if step % STAGES == 0:
            a = a0.load(a_dot)
            b = b0.load(b_dot)
            if step + STAGES < K // BK:
                _stage_m1_query(
                    a0,
                    b0,
                    X,
                    W,
                    ao + (step + STAGES) * BK,
                    bo + (step + STAGES) * BK,
                    mask,
                )
        else:
            a = a1.load(a_dot)
            b = b1.load(b_dot)
            if step + STAGES < K // BK:
                _stage_m1_query(
                    a1,
                    b1,
                    X,
                    W,
                    ao + (step + STAGES) * BK,
                    bo + (step + STAGES) * BK,
                    mask,
                )
        acc = gl.amd.cdna4.mfma(a, b, acc)
    m = gl.arange(0, 1, gl.SliceLayout(1, mma))
    n = tile * 8 + gl.arange(0, 8, gl.SliceLayout(0, mma))
    gl.store(Y + m[:, None] * OUT_N + n[None, :], acc, m[:, None] < 1)


@gluon.jit
def _pipelined_query(
    X,
    W,
    Y,
    M: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    OUT_N: gl.constexpr,
    K: gl.constexpr,
    tile,
):
    BM: gl.constexpr = _artifact_next_power_of_2(M)
    BK: gl.constexpr = 256
    STAGES: gl.constexpr = 2 if M > 8 else 4 if M == 2 else 3
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    ac: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [1, 1], [1, 0])
    bc: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, 1], [0, 1])
    ash: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BM, BK], [1, 0]
    )
    bsh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BK, 8], [0, 1]
    )
    am = gl.arange(0, BM, gl.SliceLayout(1, ac))
    ak = gl.arange(0, BK, gl.SliceLayout(0, ac))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bc))
    bn = gl.arange(0, 8, gl.SliceLayout(0, bc))
    ao = am[:, None] * SX + ak[None, :]
    bo = bn[None, :] * SW + bk[:, None]
    W = W + tile * 8 * SW
    a_buffers = ()
    b_buffers = ()
    for slot in gl.static_range(STAGES):
        a_buffers += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], ash),)
        b_buffers += (gl.allocate_shared_memory(gl.bfloat16, [BK, 8], bsh),)
    for step in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_buffers[step], X, ao + step * BK, am[:, None] < M, 0
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b_buffers[step], W, bo + step * BK
        )
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((BM, 8), gl.float32, mma)
    for step in gl.static_range(K // BK):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, K // BK - step - 1))
        a = a_buffers[step % STAGES].load(ad)
        b = b_buffers[step % STAGES].load(bd)
        if step + STAGES < K // BK:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                a_buffers[step % STAGES],
                X,
                ao + (step + STAGES) * BK,
                am[:, None] < M,
                0,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_buffers[step % STAGES], W, bo + (step + STAGES) * BK
            )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma(a, b, acc)
    m = gl.arange(0, BM, gl.SliceLayout(1, mma))
    n = tile * 8 + gl.arange(0, 8, gl.SliceLayout(0, mma))
    gl.store(Y + m[:, None] * OUT_N + n[None, :], acc, m[:, None] < M)


@gluon.jit
def _projections(
    X,
    QL,
    WQ,
    WK,
    WG,
    Q,
    KG,
    M: gl.constexpr,
    LATENT: gl.constexpr,
    HEADS: gl.constexpr,
    SX: gl.constexpr,
    SQL: gl.constexpr,
    SWQ: gl.constexpr,
    SWK: gl.constexpr,
    SWG: gl.constexpr,
    SPLITS: gl.constexpr,
):
    tile = gl.program_id(0)
    BM: gl.constexpr = 1 if M == 1 else 16
    q_tiles: gl.constexpr = gl.cdiv(HEADS * 128, 8)
    g_columns: gl.constexpr = gl.cdiv(HEADS, 16)
    if tile < q_tiles:
        if M == 1:
            _pipelined_m1_query(QL, WQ, Q, SQL, SWQ, HEADS * 128, LATENT, tile)
        else:
            _pipelined_query(QL, WQ, Q, M, SQL, SWQ, HEADS * 128, LATENT, tile)
    else:
        t = tile - q_tiles
        tile_n = t % (8 + g_columns)
        split = t // (8 + g_columns)
        if tile_n < 8:
            _key_gate_tile(X, WK, KG, M, SX, SWK, 128, BM, tile_n, split)
        else:
            _key_gate_tile(
                X,
                WG,
                KG + SPLITS * M * 128,
                M,
                SX,
                SWG,
                HEADS,
                BM,
                tile_n - 8,
                split,
            )


@gluon.jit
def _merge_projection(P, m, n, M: gl.constexpr, N: gl.constexpr, SPLITS: gl.constexpr):
    if SPLITS > 1:
        if M == 1:
            layout: gl.constexpr = gl.BlockedLayout([4, 1], [1, 64], [1, 1], [0, 1])
        else:
            layout: gl.constexpr = gl.BlockedLayout([4, 2], [1, 64], [1, 1], [0, 1])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    return _merge_projection_indices(P, m, n + d, M, N, SPLITS)


@gluon.jit
def _merge_projection_indices(
    P, m, indices, M: gl.constexpr, N: gl.constexpr, SPLITS: gl.constexpr
):
    if SPLITS > 1:
        if M == 1:
            layout: gl.constexpr = gl.BlockedLayout([4, 1], [1, 64], [1, 1], [0, 1])
        else:
            layout: gl.constexpr = gl.BlockedLayout([4, 2], [1, 64], [1, 1], [0, 1])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [1, 1], [1, 0])
    s = gl.arange(0, _artifact_next_power_of_2(SPLITS), gl.SliceLayout(1, layout))
    indices = gl.convert_layout(indices, gl.SliceLayout(0, layout))
    if SPLITS > 1:
        offsets = ((s[:, None] // 4 * M + m) * 128 + indices[None, :]) * 4 + s[
            :, None
        ] % 4
    else:
        offsets = (s[:, None] * M + m) * N + indices[None, :]
    if M >= 16 or (M == 2 and SPLITS == 12):
        partial = gl.amd.cdna4.buffer_load(P, offsets, s[:, None] < SPLITS, 0)
    else:
        partial = gl.load(P + offsets, s[:, None] < SPLITS, 0)
    x = gl.sum(partial, 0).to(gl.bfloat16).to(gl.float32)
    return gl.convert_layout(x, gl.BlockedLayout([2], [64], [1], [0]))


@gluon.jit
def _rotate(
    x,
    CosSin,
    Positions,
    m,
    SC: gl.constexpr,
    ROPE: gl.constexpr,
    TABLE_BUFFER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    pos = gl.load(Positions + m).to(gl.uint32)
    even, odd = gl.split(gl.reshape(x, (64, 2)))
    pair = gl.arange(0, 64, even.type.layout)
    half: gl.constexpr = ROPE // 2
    if TABLE_BUFFER:
        c = gl.amd.cdna4.buffer_load(CosSin, pos * SC + pair % half).to(gl.float32)
        s = gl.amd.cdna4.buffer_load(CosSin, pos * SC + half + pair % half).to(
            gl.float32
        )
    else:
        c = gl.load(CosSin + pos * SC + pair % half).to(gl.float32)
        s = gl.load(CosSin + pos * SC + half + pair % half).to(gl.float32)
    rotated_even = gl.where(pair < half, even * c + -odd * s, even)
    rotated_odd = gl.where(pair < half, odd * c + even * s, odd)
    rotated_even = rotated_even.to(gl.bfloat16).to(gl.float32)
    rotated_odd = rotated_odd.to(gl.bfloat16).to(gl.float32)
    rotated = gl.join(rotated_even, rotated_odd)
    return gl.convert_layout(gl.reshape(rotated, (128,)), layout)


@gluon.jit
def _rotate_neox(
    x,
    partner,
    CosSin,
    Positions,
    m,
    SC: gl.constexpr,
    ROPE: gl.constexpr,
    TABLE_BUFFER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    d = gl.arange(0, 128, layout)
    half: gl.constexpr = ROPE // 2
    freq = d % half
    pos = gl.load(Positions + m).to(gl.uint32)
    if TABLE_BUFFER:
        cosine = gl.amd.cdna4.buffer_load(CosSin, pos * SC + freq).to(gl.float32)
        sine = gl.amd.cdna4.buffer_load(CosSin, pos * SC + half + freq).to(gl.float32)
    else:
        cosine = gl.load(CosSin + pos * SC + freq).to(gl.float32)
        sine = gl.load(CosSin + pos * SC + half + freq).to(gl.float32)
    rotated = gl.where(
        d < half,
        x * cosine - partner * sine,
        gl.where(d < ROPE, x * cosine + partner * sine, x),
    )
    return rotated.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _scale_and_inverse(x, M: gl.constexpr):
    if M == 1:
        magnitude = gl.maximum(gl.abs(x), 0.0001)
        amax = gl.max(magnitude.to(gl.uint32, bitcast=True), 0).to(
            gl.float32, bitcast=True
        )
    else:
        magnitude = gl.abs(x).to(gl.uint32, bitcast=True)
        magnitude = gl.where(magnitude > 2139095040, 0, magnitude)
        amax = gl.maximum(gl.max(magnitude, 0).to(gl.float32, bitcast=True), 0.0001)
    exponent = gl.ceil(gl.log2(amax / 448.0))
    return (gl.exp2(exponent), gl.exp2(-exponent))


@gluon.jit
def _key(
    K,
    Gamma,
    Beta,
    CosSin,
    Positions,
    Slots,
    Cache,
    SC: gl.constexpr,
    eps,
    PAGE: gl.constexpr,
    M: gl.constexpr,
    SPLITS: gl.constexpr,
    KN: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    m = 0 if M == 1 else gl.program_id(0)
    slot = gl.load(Slots + m)
    d = gl.arange(0, 128, gl.BlockedLayout([2], [64], [1], [0]))
    raw = _merge_projection(K, m, 0, M, KN, SPLITS)
    mean = gl.sum(raw, 0) / 128
    delta = raw - mean
    inv = gl.rsqrt(gl.sum(delta * delta, 0) / 128 + eps)
    x = delta * inv * gl.load(Gamma + d).to(gl.float32) + gl.load(Beta + d).to(
        gl.float32
    )
    x = x.to(gl.bfloat16).to(gl.float32)
    if NEOX:
        half: gl.constexpr = ROPE // 2
        partner_d = gl.where(d < half, d + half, gl.where(d < ROPE, d - half, d))
        partner_raw = _merge_projection_indices(K, m, partner_d, M, KN, SPLITS)
        partner = (
            (
                (partner_raw - mean) * inv * gl.load(Gamma + partner_d).to(gl.float32)
                + gl.load(Beta + partner_d).to(gl.float32)
            )
            .to(gl.bfloat16)
            .to(gl.float32)
        )
        x = _rotate_neox(
            x,
            partner,
            CosSin,
            Positions,
            m,
            SC,
            ROPE,
            M == 1 and SPLITS == 16,
        )
    else:
        x = _rotate(x, CosSin, Positions, m, SC, ROPE, M == 1 and SPLITS == 16)
    scale, inv_scale = _scale_and_inverse(x, M)
    quant = (x * inv_scale).to(gl.float8e4nv)
    d = gl.arange(0, 128, x.type.layout)
    valid = slot >= 0
    address_slot = slot.to(gl.uint32)
    page = address_slot // PAGE
    offset = address_slot % PAGE
    dst = (
        page * PAGE * 132
        + offset // 16 * 2048
        + d // 16 * 256
        + offset % 16 * 16
        + d % 16
    )
    gl.store(Cache + dst, quant.to(gl.uint8, bitcast=True), valid)
    scale_ptr = (
        gl.cast(Cache, gl.pointer_type(gl.float32))
        + page * PAGE * 33
        + PAGE * 32
        + offset
    )
    gl.store(scale_ptr, scale, valid)


@gluon.jit
def _query(
    Q,
    Gate,
    CosSin,
    Positions,
    Out,
    Weights,
    SC: gl.constexpr,
    HEADS: gl.constexpr,
    M: gl.constexpr,
    KS: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    m = 0 if M == 1 else gl.program_id(0)
    h = gl.program_id(1) - 1
    splits = gl.arange(
        0, _artifact_next_power_of_2(KS), gl.BlockedLayout([1], [64], [1], [0])
    )
    gate_offsets = (splits * M + m) * HEADS + h
    if M >= 16:
        gate_partials = gl.amd.cdna4.buffer_load(Gate, gate_offsets, splits < KS, 0)
    else:
        gate_partials = gl.load(Gate + gate_offsets, splits < KS, 0)
    gate_value = gl.sum(gate_partials, 0).to(gl.bfloat16).to(gl.float32)
    gate = (gate_value * HEADS ** (-0.5)).to(gl.bfloat16).to(gl.float32)
    q = _merge_projection(Q, m, h * 128, M, HEADS * 128, 1)
    if NEOX:
        d = gl.arange(0, 128, q.type.layout)
        half: gl.constexpr = ROPE // 2
        partner_d = gl.where(d < half, d + half, gl.where(d < ROPE, d - half, d))
        partner = _merge_projection_indices(
            Q, m, h * 128 + partner_d, M, HEADS * 128, 1
        )
        q = _rotate_neox(q, partner, CosSin, Positions, m, SC, ROPE, M == 1)
    else:
        q = _rotate(q, CosSin, Positions, m, SC, ROPE, M == 1)
    scale, inv_scale = _scale_and_inverse(q, M)
    d = gl.arange(0, 128, q.type.layout)
    gl.store(Out + (m * HEADS + h) * 128 + d, (q * inv_scale).to(gl.float8e4nv))
    gl.store(Weights + m * HEADS + h, gate * scale * 0.08838834764831845)


@gluon.jit
def _finish(
    Q,
    KG,
    Gamma,
    Beta,
    CosSin,
    Positions,
    Slots,
    Cache,
    Out,
    Weights,
    SC: gl.constexpr,
    eps,
    PAGE: gl.constexpr,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    if gl.program_id(1) == 0:
        _key(
            KG,
            Gamma,
            Beta,
            CosSin,
            Positions,
            Slots,
            Cache,
            SC,
            eps,
            PAGE,
            M,
            SPLITS,
            128,
            ROPE,
            NEOX,
        )
    else:
        _query(
            Q,
            KG + SPLITS * M * 128,
            CosSin,
            Positions,
            Out,
            Weights,
            SC,
            HEADS,
            M,
            SPLITS,
            ROPE,
            NEOX,
        )


def indexer_prepare(
    x,
    q_latent,
    wq,
    wk,
    wgate,
    k_gamma,
    k_beta,
    cos_sin,
    positions,
    slots,
    cache: torch.Tensor,
    *,
    eps: float = 1e-06,
    rope_dim: int = 64,
    is_neox_style: bool = False,
):
    m, hidden = x.shape
    heads = wgate.shape[0]
    splits = triton.cdiv(hidden, 384)
    kg = torch.empty((splits, m, 128 + heads), device=x.device, dtype=torch.float32)
    q = torch.empty((m, heads * 128), device=x.device, dtype=torch.float32)
    out = torch.empty((m, heads, 128), device=x.device, dtype=torch.float8_e4m3fn)
    weights = torch.empty((m, heads), device=x.device, dtype=torch.float32)
    tiles = triton.cdiv(heads * 128, 8) + splits * (8 + triton.cdiv(heads, 16))
    _projections[tiles, triton.cdiv(m, 16)](
        x,
        q_latent,
        wq,
        wk,
        wgate,
        q,
        kg,
        m,
        q_latent.shape[1],
        heads,
        x.stride(0),
        q_latent.stride(0),
        wq.stride(0),
        wk.stride(0),
        wgate.stride(0),
        splits,
        num_warps=1,
    )
    _finish[m, heads + 1](
        q,
        kg,
        k_gamma,
        k_beta,
        cos_sin,
        positions,
        slots,
        cache,
        out,
        weights,
        cos_sin.stride(0),
        eps,
        cache.shape[1],
        m,
        heads,
        splits,
        rope_dim,
        is_neox_style,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return (out, weights)
