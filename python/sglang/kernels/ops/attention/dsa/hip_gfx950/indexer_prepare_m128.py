import torch
import triton
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
def _accumulate_projection(a, b, acc):
    matrix: gl.constexpr = acc.type.layout
    a = gl.convert_layout(a, gl.DotOperandLayout(0, matrix, 8))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, matrix, 8))
    return gl.amd.cdna4.mfma(a, b, acc)


@gluon.jit
def _project_irregular_tile(
    X,
    W,
    Y,
    row_tile,
    column_tile,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    BM: gl.constexpr,
):
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    rows = gl.arange(0, BM, gl.SliceLayout(1, load_a))
    cols = gl.arange(0, 16, gl.SliceLayout(0, load_b))
    ka = gl.arange(0, 128, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, 128, gl.SliceLayout(1, load_b))
    a_base = X + row_tile * BM * SX
    b_base = W + column_tile * 16 * SW
    a_offsets = rows[:, None] * SX + ka[None, :]
    b_offsets = cols[None, :] * SW + kb[:, None]
    mask = row_tile * BM + rows[:, None] < M
    acc = gl.zeros((BM, 16), gl.float32, matrix)
    a = gl.load(a_base + a_offsets, mask, 0)
    b = gl.load(b_base + b_offsets)
    for block in gl.static_range(1, K // 128):
        next_a = gl.load(a_base + block * 128 + a_offsets, mask, 0)
        next_b = gl.load(b_base + block * 128 + b_offsets)
        acc = _accumulate_projection(a, b, acc)
        a, b = (next_a, next_b)
    acc = _accumulate_projection(a, b, acc)
    rows_out = row_tile * BM + gl.arange(0, BM, gl.SliceLayout(1, matrix))
    cols_out = column_tile * 16 + gl.arange(0, 16, gl.SliceLayout(0, matrix))
    gl.store(Y + rows_out[:, None] * N + cols_out[None, :], acc, rows_out[:, None] < M)


@gluon.jit
def _project_irregular_key_gate(
    X,
    WK,
    WG,
    Key,
    Gate,
    row_tile,
    col_tile,
    M: gl.constexpr,
    H: gl.constexpr,
    HEADS: gl.constexpr,
    SX: gl.constexpr,
    SWK: gl.constexpr,
    SWG: gl.constexpr,
):
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    rows = gl.arange(0, 16, gl.SliceLayout(1, load_a))
    cols = gl.arange(0, 16, gl.SliceLayout(0, load_b))
    ka = gl.arange(0, 128, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, 128, gl.SliceLayout(1, load_b))
    a_base = X + row_tile * 16 * SX
    k_base = WK + col_tile * 16 * SWK
    g_base = WG + col_tile * 16 * SWG
    a_offsets = rows[:, None] * SX + ka[None, :]
    k_offsets = cols[None, :] * SWK + kb[:, None]
    g_offsets = cols[None, :] * SWG + kb[:, None]
    mask = row_tile * 16 + rows[:, None] < M
    key = gl.zeros((16, 16), gl.float32, matrix)
    gate = gl.zeros((16, 16), gl.float32, matrix)
    for block in range(H // 128):
        a = gl.load(a_base + block * 128 + a_offsets, mask, 0)
        b = gl.load(k_base + block * 128 + k_offsets)
        g = gl.load(g_base + block * 128 + g_offsets)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, matrix, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, matrix, 8))
        g = gl.convert_layout(g, gl.DotOperandLayout(1, matrix, 8))
        key = gl.amd.cdna4.mfma(a, b, key)
        gate = gl.amd.cdna4.mfma(a, g, gate)
    rows_out = row_tile * 16 + gl.arange(0, 16, gl.SliceLayout(1, matrix))
    cols_out = col_tile * 16 + gl.arange(0, 16, gl.SliceLayout(0, matrix))
    gl.store(
        Key + rows_out[:, None] * 128 + cols_out[None, :], key, rows_out[:, None] < M
    )
    gl.store(
        Gate + rows_out[:, None] * HEADS + cols_out[None, :],
        gate,
        rows_out[:, None] < M,
    )


@gluon.jit
def _projections_irregular(
    X,
    QLatent,
    WK,
    WQ,
    WGate,
    Key,
    Query,
    Gate,
    M: gl.constexpr,
    H: gl.constexpr,
    QK: gl.constexpr,
    HEADS: gl.constexpr,
    SX: gl.constexpr,
    SQL: gl.constexpr,
    SWK: gl.constexpr,
    SWQ: gl.constexpr,
    SWG: gl.constexpr,
):
    program = gl.program_id(0)
    row_tiles: gl.constexpr = gl.cdiv(M, 16)
    query_rows: gl.constexpr = gl.cdiv(M, 32)
    query_columns: gl.constexpr = HEADS * 8
    query_programs: gl.constexpr = query_rows * query_columns
    if program < query_programs:
        group_width: gl.constexpr = min(128, query_columns)
        col = (
            program // (query_rows * group_width) * group_width + program % group_width
        )
        row = program // group_width % query_rows
        _project_irregular_tile(
            QLatent, WQ, Query, row, col, M, HEADS * 128, QK, SQL, SWQ, 32
        )
    else:
        program -= query_programs
        col = program // row_tiles
        row = program % row_tiles
        if col < HEADS // 16:
            _project_irregular_key_gate(
                X, WK, WGate, Key, Gate, row, col, M, H, HEADS, SX, SWK, SWG
            )
        else:
            _project_irregular_tile(X, WK, Key, row, col, M, 128, H, SX, SWK, 16)


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
    BM: gl.constexpr = 64 if M == 256 else 32
    query_stages: gl.constexpr = 2 if M == 256 else 3
    key_stages: gl.constexpr = 3 if M == 256 else 2
    key_phases: gl.constexpr = 8 if M == 256 else 16
    stripe: gl.constexpr = 8 if M == 256 else 16
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
def _project_key_async(
    X,
    W,
    Parts,
    M: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SPLITS: gl.constexpr,
):
    program = gl.program_id(0).to(gl.uint32)
    split = program % SPLITS
    row = program // SPLITS % (M // 32)
    column = program // (SPLITS * (M // 32))
    _project_async_tile(
        X,
        W,
        None,
        Parts,
        None,
        row,
        column,
        split,
        M,
        128,
        K,
        SX,
        SW,
        0,
        SPLITS,
        32,
        3,
        False,
        False,
        32,
        True,
    )


@gluon.jit
def _merge_key(Parts, row, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    split = gl.arange(0, SPLITS, gl.SliceLayout(1, layout))
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
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
    CHANNELS: gl.constexpr = 1,
):
    GROUPED: gl.constexpr = len(x.shape) == 2
    PACK: gl.constexpr = x.numel // 64
    position = gl.load(Positions + row).to(gl.int32)
    if CHANNELS > 1:
        partner = _exchange_registers(x, 0)
    else:
        partner = _exchange_lanes(x, 0, PACK)
    cosine = gl.load(CosSin + position * SC + d // 2, d < 64, 1).to(gl.float32)
    sine = gl.load(CosSin + position * SC + 32 + d // 2, d < 64, 0).to(gl.float32)
    if GROUPED:
        d, cosine, sine = (d[None, :], cosine[None, :], sine[None, :])
    x = (
        gl.where(d < 64, x * cosine + gl.where(d % 2 == 0, -partner, partner) * sine, x)
        .to(gl.bfloat16)
        .to(gl.float32)
    )
    return x


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
def _normalize_key(x, Gamma, Beta, d, eps, ORDERED_HALVES: gl.constexpr):
    if ORDERED_HALVES:
        mean = gl.sum(gl.sum(x.reshape((2, 64)), 1), 0) / 128
    else:
        mean = gl.sum(x, 0) / 128
    delta = x - mean
    if ORDERED_HALVES:
        squared = (delta * delta).reshape((2, 64))
        variance = gl.sum(gl.sum(squared, 1), 0) / 128
    else:
        variance = gl.sum(delta * delta, 0) / 128
    inv_std = gl.rsqrt(variance + eps)
    return (
        (
            delta * inv_std * gl.load(Gamma + d).to(gl.float32)
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
    ORDERED_HALVES: gl.constexpr,
    BUFFER_STORE: gl.constexpr,
):
    d = gl.arange(0, 128, gl.BlockedLayout([1], [64], [1], [0]))
    slot_wide = gl.load(Slots + row)
    x = _merge_key(Parts, row, SPLITS)
    x = _normalize_key(x, Gamma, Beta, d, eps, ORDERED_HALVES)
    x = _rotate(x, d, CosSin, Positions, row, SC)
    d = gl.arange(0, 128, x.type.layout)
    quantized, scale = _quantize_fp8(x)
    active = slot_wide >= 0
    slot = slot_wide.to(gl.uint32) if BUFFER_STORE else slot_wide.to(gl.int32)
    page = slot // PAGE
    offset = slot % PAGE
    destination = (
        page * PAGE * 132
        + offset // 16 * 2048
        + d // 16 * 256
        + offset % 16 * 16
        + d % 16
    )
    if BUFFER_STORE:
        gl.amd.cdna4.buffer_store(
            quantized.to(gl.uint8, bitcast=True), Cache, destination, active
        )
        scale_base = gl.cast(Cache, gl.pointer_type(gl.float32))
        scale_offset = (
            page * PAGE * 33
            + PAGE * 32
            + offset
            + gl.arange(0, 1, gl.BlockedLayout([1], [64], [1], [0]))
        )
        gl.amd.cdna4.buffer_store(scale, scale_base, scale_offset, active)
    else:
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
):
    channels: gl.constexpr = 8 if M == 128 else 4 if M == 256 else 2
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
    x = _rotate(x, d, CosSin, Positions, row, SC, channels)
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
    KEY_ONLY: gl.constexpr,
    GROUP: gl.constexpr,
):
    program = gl.program_id(0)
    if KEY_ONLY:
        _finish_key(
            KeyParts,
            Gamma,
            Beta,
            CosSin,
            Positions,
            Slots,
            Cache,
            program,
            SC,
            PAGE,
            eps,
            SPLITS,
            False,
            True,
        )
    else:
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
                True,
                False,
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
    key_only: bool = False,
    eps: float = 1e-06,
):
    m, width = x.shape
    heads = wgate.shape[0]
    splits = 8 if m == 256 or not key_only else 16
    if m % 16:
        splits = 1
    group = 8 if m in (128, 256) else 4
    assert cos_sin.shape[0] * cos_sin.stride(0) < 2**31 and cache.numel() < 2**31
    key_parts = torch.empty((m, splits, 128), device=x.device, dtype=torch.float32)
    query = gate_parts = out = weights = None
    if key_only:
        _project_key_async[m // 32 * 4 * splits,](
            x, wk, key_parts, m, width, x.stride(0), wk.stride(0), splits, num_warps=4
        )
    else:
        query = torch.empty((m, heads * 128), device=x.device, dtype=torch.bfloat16)
        gate_parts = torch.empty(
            (splits, m, heads), device=x.device, dtype=torch.float32
        )
        out = torch.empty((m, heads, 128), device=x.device, dtype=torch.float8_e4m3fn)
        weights = torch.empty((m, heads), device=x.device, dtype=torch.float32)
        if m % 16:
            grid = triton.cdiv(m, 16) * 8 + triton.cdiv(m, 32) * heads * 8
            _projections_irregular[grid,](
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
                heads,
                x.stride(0),
                q_latent.stride(0),
                wk.stride(0),
                wq.stride(0),
                wgate.stride(0),
                num_warps=1,
            )
        else:
            grid = m // (64 if m == 256 else 32) * 64 + m // 32 * 32
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
    _finish[m if key_only else m * (heads // group + 1),](
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
        key_only,
        group,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return (out, weights)
