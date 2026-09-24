import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.runtime.jit import constexpr_function


@constexpr_function
def _lane_exchange_assembly(bit, pack):
    controls = (
        ("quad_perm:[1,0,3,2]",),
        ("quad_perm:[2,3,0,1]",),
        ("quad_perm:[3,2,1,0]", "row_half_mirror"),
        ("row_ror:8",),
    )[bit]
    lines = []
    for stage, control in enumerate(controls):
        lines.append("s_nop 1")
        for i in range(pack):
            source = i + pack if stage == 0 else i
            lines.append(
                f"v_mov_b32 ${i}, ${source} {control} row_mask:0xf bank_mask:0xf"
            )
    return ("\n".join(lines), ",".join(["=&v"] * pack + ["v"] * pack))


@gluon.jit
def _exchange_lanes(x, BIT: gl.constexpr, PACK: gl.constexpr):
    code: gl.constexpr = _lane_exchange_assembly(BIT, PACK)
    return gl.inline_asm_elementwise(
        code[0], code[1], (x,), dtype=gl.float32, is_pure=True, pack=PACK
    )


@gluon.jit
def _exchange_registers(x, BIT: gl.constexpr):
    paired = x.reshape((x.numel // (2 << BIT), 2, 1 << BIT)).permute((0, 2, 1))
    low, high = gl.split(paired)
    swapped = gl.join(high, low).permute((0, 2, 1)).reshape(x.shape)
    return gl.convert_layout(swapped, x.type.layout, assert_trivial=True)


@gluon.jit
def _project_async_tile(
    X,
    W,
    WG,
    Y,
    Gate,
    row,
    column,
    split,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SWG: gl.constexpr,
    SPLITS: gl.constexpr,
    BN: gl.constexpr,
    STAGES: gl.constexpr,
    TRANSPOSED: gl.constexpr,
    WITH_GATE: gl.constexpr,
    BM: gl.constexpr = 32,
    EXPLICIT_SYNC: gl.constexpr = False,
    MAX_PHASE: gl.constexpr = 8,
):
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=TRANSPOSED, warps_per_cta=[2, 2]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    per_phase: gl.constexpr = 16 // MAX_PHASE
    shared_a = gl.allocate_shared_memory(
        X.dtype.element_ty,
        [STAGES, BM, 128],
        gl.SwizzledSharedLayout(8, per_phase, MAX_PHASE, [1, 0]),
    )
    shared_b = gl.allocate_shared_memory(
        W.dtype.element_ty,
        [STAGES, 128, BN],
        gl.SwizzledSharedLayout(8, per_phase, MAX_PHASE, [0, 1]),
    )
    if WITH_GATE:
        shared_g = gl.allocate_shared_memory(
            WG.dtype.element_ty,
            [STAGES, 128, BN],
            gl.SwizzledSharedLayout(8, per_phase, MAX_PHASE, [0, 1]),
        )
        gate = gl.zeros((BM, BN), gl.float32, matrix)
    rows = gl.arange(0, BM, gl.SliceLayout(1, load_a))
    cols = gl.arange(0, BN, gl.SliceLayout(0, load_b))
    ka = gl.arange(0, 128, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, 128, gl.SliceLayout(1, load_b))
    chunk: gl.constexpr = K // SPLITS
    blocks: gl.constexpr = chunk // 128
    gl.static_assert(M % BM == 0 and K % (SPLITS * 128) == 0 and (blocks >= STAGES))
    a_base = X + row * BM * SX + split * chunk
    b_base = W + column * BN * SW + split * chunk
    a_offsets = rows[:, None] * SX + ka[None, :]
    b_offsets = cols[None, :] * SW + kb[:, None]
    if WITH_GATE:
        g_base = WG + split * chunk
        g_offsets = cols[None, :] * SWG + kb[:, None]
    for stage in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            shared_a.index(stage), a_base + stage * 128, a_offsets
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            shared_b.index(stage), b_base + stage * 128, b_offsets
        )
        if WITH_GATE:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                shared_g.index(stage), g_base + stage * 128, g_offsets
            )
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((BM, BN), gl.float32, matrix)
    for block in gl.static_range(blocks):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, blocks - block - 1))
        if EXPLICIT_SYNC:
            gl.barrier()
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            shared_a.index(block % STAGES), dot_a
        )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            shared_b.index(block % STAGES), dot_b
        )
        if WITH_GATE:
            g = gl.amd.cdna4.async_copy.load_shared_relaxed(
                shared_g.index(block % STAGES), dot_b
            )
        if block + STAGES < blocks:
            if EXPLICIT_SYNC:
                gl.inline_asm_elementwise(
                    "s_waitcnt lgkmcnt(0)",
                    "=v",
                    (),
                    dtype=gl.int32,
                    is_pure=False,
                    pack=1,
                )
            gl.barrier()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                shared_a.index(block % STAGES),
                a_base + (block + STAGES) * 128,
                a_offsets,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                shared_b.index(block % STAGES),
                b_base + (block + STAGES) * 128,
                b_offsets,
            )
            if WITH_GATE:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    shared_g.index(block % STAGES),
                    g_base + (block + STAGES) * 128,
                    g_offsets,
                )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma(a, b, acc)
        if WITH_GATE:
            gate = gl.amd.cdna4.mfma(a, g, gate)
    rows_out = row * BM + gl.arange(0, BM, gl.SliceLayout(1, matrix))
    cols_out = column * BN + gl.arange(0, BN, gl.SliceLayout(0, matrix))
    gl.store(Y + (rows_out[:, None] * SPLITS + split) * N + cols_out[None, :], acc)
    if WITH_GATE:
        gl.store(
            Gate + split * M * 32 + rows_out[:, None] * 32 + cols_out[None, :], gate
        )


@gluon.jit
def _projections_async(
    X,
    QLatent,
    WK,
    WQ,
    WGate,
    KeyParts,
    Query,
    GateParts,
    M: gl.constexpr,
    H: gl.constexpr,
    QK: gl.constexpr,
    SX: gl.constexpr,
    SQL: gl.constexpr,
    SWK: gl.constexpr,
    SWQ: gl.constexpr,
    SWG: gl.constexpr,
):
    program = gl.program_id(0).to(gl.uint32)
    BM: gl.constexpr = 32
    query_stages: gl.constexpr = 3
    key_stages: gl.constexpr = 2
    key_phases: gl.constexpr = 16
    stripe: gl.constexpr = 16
    query_rows: gl.constexpr = M // BM
    query_programs: gl.constexpr = query_rows * 64
    if program < query_programs:
        col = program % stripe * (64 // stripe) + program // (stripe * query_rows)
        row = program // stripe % query_rows
        _project_async_tile(
            QLatent,
            WQ,
            None,
            Query,
            None,
            row,
            col,
            0,
            M,
            4096,
            QK,
            SQL,
            SWQ,
            0,
            1,
            64,
            query_stages,
            False,
            False,
            BM,
            True,
            16,
        )
    else:
        program -= query_programs
        split = program % 8
        if M == 128:
            row = program // 8 % (M // 32)
            col = program // (8 * (M // 32))
        else:
            col = program // 8 % 4
            row = program // 32
        if col == 0:
            _project_async_tile(
                X,
                WK,
                WGate,
                KeyParts,
                GateParts,
                row,
                col,
                split,
                M,
                128,
                H,
                SX,
                SWK,
                SWG,
                8,
                32,
                key_stages,
                False,
                True,
                32,
                True,
                key_phases,
            )
        else:
            _project_async_tile(
                X,
                WK,
                None,
                KeyParts,
                None,
                row,
                col,
                split,
                M,
                128,
                H,
                SX,
                SWK,
                0,
                8,
                32,
                key_stages,
                False,
                False,
                32,
                True,
                key_phases,
            )


@gluon.jit
def _merge_key(Parts, row, d, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    split = gl.arange(0, SPLITS, gl.SliceLayout(1, layout))
    d = gl.convert_layout(d, gl.SliceLayout(0, layout))
    partials = gl.load(Parts + row * SPLITS * 128 + split[:, None] * 128 + d[None, :])
    total = gl.convert_layout(
        gl.sum(partials, 0), gl.BlockedLayout([1], [64], [1], [0])
    )
    return total.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _rotate(
    x,
    d,
    CosSin,
    Positions,
    row,
    SC: gl.constexpr,
    ROPE: gl.constexpr,
    CHANNELS: gl.constexpr = 1,
):
    GROUPED: gl.constexpr = len(x.shape) == 2
    PACK: gl.constexpr = x.numel // 64
    position = gl.load(Positions + row).to(gl.int32)
    if CHANNELS > 1:
        partner = _exchange_registers(x, 0)
    else:
        partner = _exchange_lanes(x, 0, PACK)
    half: gl.constexpr = ROPE // 2
    cosine = gl.load(CosSin + position * SC + d // 2, d < ROPE, 1).to(gl.float32)
    sine = gl.load(CosSin + position * SC + half + d // 2, d < ROPE, 0).to(gl.float32)
    if GROUPED:
        d, cosine, sine = (d[None, :], cosine[None, :], sine[None, :])
    x = (
        gl.where(
            d < ROPE,
            x * cosine + gl.where(d % 2 == 0, -partner, partner) * sine,
            x,
        )
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    return x


@gluon.jit
def _rotate_neox(
    x,
    partner,
    d,
    CosSin,
    Positions,
    row,
    SC: gl.constexpr,
    ROPE: gl.constexpr,
):
    GROUPED: gl.constexpr = len(x.shape) == 2
    half: gl.constexpr = ROPE // 2
    position = gl.load(Positions + row).to(gl.int32)
    cosine = gl.load(CosSin + position * SC + d % half, d < ROPE, 1).to(gl.float32)
    sine = gl.load(CosSin + position * SC + half + d % half, d < ROPE, 0).to(gl.float32)
    if GROUPED:
        d, cosine, sine = (d[None, :], cosine[None, :], sine[None, :])
    return (
        gl.where(
            d < half,
            x * cosine - partner * sine,
            gl.where(d < ROPE, x * cosine + partner * sine, x),
        )
        .to(gl.bfloat16)
        .to(gl.float32)
    )


@gluon.jit
def _quantize_fp8(x):
    axis: gl.constexpr = len(x.shape) - 1
    exponent = gl.ceil(gl.log2(gl.maximum(gl.max(gl.abs(x), axis), 0.0001) / 448.0))
    scale = gl.exp2(exponent)
    reciprocal = gl.exp2(-exponent)
    if len(x.shape) == 2:
        reciprocal = reciprocal[:, None]
    return ((x * reciprocal).to(gl.float8e4nv), scale)


@gluon.jit
def _key_stats(x, eps):
    mean = gl.sum(gl.sum(x.reshape((2, 64)), 1), 0) / 128
    delta = x - mean
    squared = (delta * delta).reshape((2, 64))
    variance = gl.sum(gl.sum(squared, 1), 0) / 128
    return (mean, gl.rsqrt(variance + eps))


@gluon.jit
def _normalize_key(x, Gamma, Beta, d, mean, inv_std):
    return (
        (
            (x - mean) * inv_std * gl.load(Gamma + d).to(gl.float32)
            + gl.load(Beta + d).to(gl.float32)
        )
        .to(gl.bfloat16)
        .to(gl.float32)
    )


@gluon.jit
def _finish_key(
    Parts,
    Gamma,
    Beta,
    CosSin,
    Positions,
    Slots,
    Cache,
    row,
    SC: gl.constexpr,
    PAGE: gl.constexpr,
    eps,
    SPLITS: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    d = gl.arange(0, 128, gl.BlockedLayout([1], [64], [1], [0]))
    slot_wide = gl.load(Slots + row)
    raw = _merge_key(Parts, row, d, SPLITS)
    mean, inv_std = _key_stats(raw, eps)
    x = _normalize_key(raw, Gamma, Beta, d, mean, inv_std)
    if NEOX:
        half: gl.constexpr = ROPE // 2
        partner_d = gl.where(d < half, d + half, gl.where(d < ROPE, d - half, d))
        partner_raw = _merge_key(Parts, row, partner_d, SPLITS)
        partner = _normalize_key(partner_raw, Gamma, Beta, partner_d, mean, inv_std)
        x = _rotate_neox(x, partner, d, CosSin, Positions, row, SC, ROPE)
    else:
        x = _rotate(x, d, CosSin, Positions, row, SC, ROPE)
    d = gl.arange(0, 128, x.type.layout)
    quantized, scale = _quantize_fp8(x)
    active = slot_wide >= 0
    slot = slot_wide.to(gl.int32)
    page = slot // PAGE
    offset = slot % PAGE
    destination = (
        page * PAGE * 132
        + offset // 16 * 2048
        + d // 16 * 256
        + offset % 16 * 16
        + d % 16
    )
    gl.store(Cache + destination, quantized.to(gl.uint8, bitcast=True), active)
    scale_ptr = (
        gl.cast(Cache, gl.pointer_type(gl.float32))
        + page * PAGE * 33
        + PAGE * 32
        + offset
    )
    gl.store(scale_ptr, scale, active)


@gluon.jit
def _finish_query(
    Query,
    GateParts,
    CosSin,
    Positions,
    Out,
    Weights,
    row,
    head_group,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    SC: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    channels: gl.constexpr = 8 if M == 128 else 2
    layout: gl.constexpr = gl.BlockedLayout(
        [1, channels], [GROUP, 64 // GROUP], [1, 1], [1, 0]
    )
    head = head_group * GROUP + gl.arange(0, GROUP, gl.SliceLayout(1, layout))
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    index = (row * HEADS + head[:, None]) * 128 + d[None, :]
    x = gl.load(Query + index).to(gl.float32)
    gate_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [GROUP, 64 // GROUP], [1, 1], [1, 0]
    )
    gate_head = head_group * GROUP + gl.arange(0, GROUP, gl.SliceLayout(1, gate_layout))
    split = gl.arange(0, SPLITS, gl.SliceLayout(0, gate_layout))
    gate_offsets = split[None, :] * M * HEADS + row * HEADS + gate_head[:, None]
    gate = gl.sum(gl.load(GateParts + gate_offsets), 1).to(gl.bfloat16).to(gl.float32)
    gate = (gate * HEADS ** (-0.5)).to(gl.bfloat16).to(gl.float32)
    if NEOX:
        half: gl.constexpr = ROPE // 2
        partner_d = gl.where(d < half, d + half, gl.where(d < ROPE, d - half, d))
        partner_index = (row * HEADS + head[:, None]) * 128 + partner_d[None, :]
        partner = gl.load(Query + partner_index).to(gl.float32)
        x = _rotate_neox(x, partner, d, CosSin, Positions, row, SC, ROPE)
    else:
        x = _rotate(x, d, CosSin, Positions, row, SC, ROPE, channels)
    quantized, scale = _quantize_fp8(x)
    gl.store(Out + index, quantized)
    gate = gl.convert_layout(gate, gl.SliceLayout(1, layout), assert_trivial=True)
    gl.store(Weights + row * HEADS + head, gate * scale * 0.08838834764831845)


@gluon.jit
def _finish(
    KeyParts,
    Query,
    GateParts,
    Gamma,
    Beta,
    CosSin,
    Positions,
    Slots,
    Cache,
    Out,
    Weights,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    SC: gl.constexpr,
    PAGE: gl.constexpr,
    eps,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    ROPE: gl.constexpr,
    NEOX: gl.constexpr,
):
    program = gl.program_id(0)
    row = program % M
    role = program // M
    if role < HEADS // GROUP:
        _finish_query(
            Query,
            GateParts,
            CosSin,
            Positions,
            Out,
            Weights,
            row,
            role,
            M,
            HEADS,
            SC,
            SPLITS,
            GROUP,
            ROPE,
            NEOX,
        )
    else:
        _finish_key(
            KeyParts,
            Gamma,
            Beta,
            CosSin,
            Positions,
            Slots,
            Cache,
            row,
            SC,
            PAGE,
            eps,
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
    m, width = x.shape
    heads = wgate.shape[0]
    splits = 8
    group = 8 if m == 128 else 4
    assert cos_sin.shape[0] * cos_sin.stride(0) < 2**31 and cache.numel() < 2**31
    key_parts = torch.empty((m, splits, 128), device=x.device, dtype=torch.float32)
    query = torch.empty((m, heads * 128), device=x.device, dtype=torch.bfloat16)
    gate_parts = torch.empty((splits, m, heads), device=x.device, dtype=torch.float32)
    out = torch.empty((m, heads, 128), device=x.device, dtype=torch.float8_e4m3fn)
    weights = torch.empty((m, heads), device=x.device, dtype=torch.float32)
    grid = m // 32 * 96
    _projections_async[grid,](
        x,
        q_latent,
        wk,
        wq,
        wgate,
        key_parts,
        query,
        gate_parts,
        m,
        width,
        q_latent.shape[1],
        x.stride(0),
        q_latent.stride(0),
        wk.stride(0),
        wq.stride(0),
        wgate.stride(0),
        num_warps=4,
    )
    _finish[m * (heads // group + 1),](
        key_parts,
        query,
        gate_parts,
        k_gamma,
        k_beta,
        cos_sin,
        positions,
        slots,
        cache,
        out,
        weights,
        m,
        heads,
        cos_sin.stride(0),
        cache.shape[1],
        eps,
        splits,
        group,
        rope_dim,
        is_neox_style,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return (out, weights)
