"""Generated Kimi-K3 attention residual norm kernels for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

"""Selected Kimi attention residual norm kernels.

Layouts, reduction order, synchronization and launch options retain the tuned
candidates; named entrypoints correspond to manifest contracts.
"""


from triton.experimental import gluon


from triton.experimental.gluon import language as gl


from triton.experimental.gluon.language.amd.cdna3 import buffer_load


import torch


import triton


@gluon.jit
def _bank_row(values, ROW: gl.constexpr):
    width: gl.constexpr = values.shape[1]
    rows: gl.constexpr = values.shape[0]
    selected = values.permute((1, 0))
    for stage in gl.static_range(3):
        even, odd = gl.split(selected.reshape((width, rows >> stage + 1, 2)))
        if ROW >> stage & 1:
            selected = odd
        else:
            selected = even
    return gl.convert_layout(selected.reshape((width,)), gl.SliceLayout(0, values.type.layout), assert_trivial=True)


@gluon.constexpr_function
def _butterfly_layout(stage, warps, rows):
    registers = [[0, 0, 0, 1], [0, 0, 32 >> stage, 0]]
    registers += [[1 << bit, 0, 0, 0] for bit in range(stage + 1, rows.bit_length() - 1)]
    lanes = [[0, 0, 1 << bit, 0] for bit in range(5 - stage)]
    lanes += [[1 << bit, 0, 0, 0] for bit in range(stage, -1, -1)]
    waves = [[0, 1 << bit, 0, 0] for bit in range(warps.bit_length() - 1)]
    return gl.DistributedLinearLayout(registers, lanes, waves, [], [rows, warps, 64 >> stage, 2])


@gluon.constexpr_function
def _current_group_layout(warps):
    registers = [[0, 32, 0]]
    lanes = [[0, 1 << bit, 0] for bit in range(5)] + [[0, 0, 1]]
    waves = [[1 << bit, 0, 0] for bit in range(warps.bit_length() - 1)]
    return gl.DistributedLinearLayout(registers, lanes, waves, [], [warps, 64, 2])


@gluon.jit
def _current_stat_partials(current, cw, BH: gl.constexpr, NW: gl.constexpr):
    shape: gl.constexpr = (BH // (NW * 512), NW, 64, 8)
    dot = gl.sum(gl.sum((current * cw).reshape(shape), 0), 2)
    square = gl.sum(gl.sum((current * current).reshape(shape), 0), 2)
    joint = gl.convert_layout(gl.join(dot, square), _current_group_layout(NW))
    joint = gl.sum(joint.reshape((NW, 2, 32, 2)), 1)
    return gl.sum(joint, 1)


@gluon.jit
def _eight_subgroup_sums(values, ROWS: gl.constexpr, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr):
    groups: gl.constexpr = WAVES * 8
    wave_width: gl.constexpr = 8 * VEC
    pieces = gl.reshape(values, (ROWS, BLOCK // (groups * wave_width), groups, wave_width))
    pieces = gl.permute(pieces, (0, 2, 1, 3))
    return gl.sum(gl.reshape(pieces, (ROWS, groups, BLOCK // groups)), 2)


@gluon.jit
def _exact_output_partials(mixed, SIZE: gl.constexpr, VECTOR: gl.constexpr, NW: gl.constexpr):
    shape: gl.constexpr = (SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    return gl.sum(gl.sum((mixed * mixed).reshape(shape), 0), 2)


@gluon.constexpr_function
def _exchange_layout(rows, warps):
    row_bits = rows.bit_length() - 1
    wave_bits = warps.bit_length() - 1
    lanes = [[1 << bit, 0, 0] for bit in range(row_bits)]
    lanes += [[0, 1 << bit, 0] for bit in range(wave_bits)]
    lanes += [[0, 0, 0]] * (6 - row_bits - wave_bits)
    return gl.DistributedLinearLayout([[0, 0, 1]], lanes, [[0, 0, 0]] * wave_bits, [], [rows, warps, 2])


@gluon.jit
def _fast_mix(C, Bank, S, OW, Out, H: gl.constexpr, SC: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, NORM: gl.constexpr, eps, BLOCK: gl.constexpr, ROWS: gl.constexpr, NW: gl.constexpr, VEC: gl.constexpr):
    m = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [NW], [0])
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    h = gl.arange(0, BLOCK, layout=layout)
    rows = gl.arange(0, ROWS, layout=sl)
    ow = gl.load(OW + h, h < H, 0).to(gl.float32)
    scores = gl.load(S + m * (NV + 1) + rows, rows <= NV, -float('inf'))
    e = gl.exp(scores - gl.max(scores, 0))
    p = e * (1.0 / gl.sum(e, 0))
    acc = gl.full((BLOCK,), 0, gl.float32, layout)
    for j in gl.static_range(NV):
        w = gl.sum(gl.gather(p, gl.full((1,), j, gl.int32, sl), 0), 0)
        v = gl.load(Bank + m * SB0 + j * SB1 + h, h < H, 0).to(gl.float32)
        acc += w * v
    w = gl.sum(gl.gather(p, gl.full((1,), NV, gl.int32, sl), 0), 0)
    v = gl.load(C + m * SC + h, h < H, 0).to(gl.float32)
    acc += w * v
    inv = gl.rsqrt(gl.sum(acc * acc, 0) * gl.div_rn(1.0, H * 1.0) + eps)
    acc = acc * inv * ow
    gl.store(Out + m * H + h, acc, h < H)


@gluon.jit
def _sum_pair(a, b, c, d):
    return (a + c, b + d)


@gluon.jit
def _fast_scores(P, A, C, Bank, CW, S, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, eps, BLOCK: gl.constexpr, NW: gl.constexpr, VEC: gl.constexpr):
    m, j = (gl.program_id(0), gl.program_id(1))
    h = gl.arange(0, BLOCK, layout=gl.BlockedLayout([VEC], [64], [NW], [0]))
    if j < NV:
        v = gl.load(Bank + m * SB0 + j * SB1 + h, h < H, 0).to(gl.float32)
    else:
        v = gl.load(P + m * SP + h, h < H, 0).to(gl.float32)
        if ADD:
            a = gl.load(A + m * SA + h, h < H, 0).to(gl.float32)
            v = (v + a).to(C.dtype.element_ty).to(gl.float32)
            gl.store(C + m * H + h, v, h < H)
        if WRITE:
            gl.store(Bank + m * SB0 + NV * SB1 + h, v, h < H)
    cw = gl.load(CW + h, h < H, 0)
    dot, sq = gl.reduce((v * cw, v * v), 0, _sum_pair)
    s = dot * gl.rsqrt(sq * gl.div_rn(1.0, H * 1.0) + eps)
    gl.store(S + m * (NV + 1) + j, s)


@gluon.jit
def _row_subgroup_sums(values, ROWS: gl.constexpr, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr):
    groups: gl.constexpr = WAVES * 4
    wave_width: gl.constexpr = 16 * VEC
    pieces = gl.reshape(values, (ROWS, BLOCK // (groups * wave_width), groups, wave_width))
    pieces = gl.permute(pieces, (0, 2, 1, 3))
    return gl.sum(gl.reshape(pieces, (ROWS, groups, BLOCK // groups)), 2)


@gluon.jit
def _fused_four(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr):
    groups: gl.constexpr = WAVES * 4
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, 4, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    if ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if gl.program_id(0) == 0:
            gl.store(Current + token * H + h, current, h < H)
    if not ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current_dot = _row_subgroup_sums((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _row_subgroup_sums((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    if ADD:
        values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    else:
        b0 = buffer_load(Bank + token * SB0, h, h < H, 0)
        b1 = buffer_load(Bank + token * SB0 + SB1, h, h < H, 0)
        b2 = buffer_load(Bank + token * SB0 + 2 * SB1, h, h < H, 0)
        b3 = buffer_load(Bank + token * SB0 + 3 * SB1, h, h < H, 0)
        values = gl.permute(gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), (BLOCK, 4)), (1, 0))
        values = gl.convert_layout(values, layout, assert_trivial=True).to(gl.float32)
    bank_dot = _row_subgroup_sums(values * score_weight[None, :], 4, BLOCK, WAVES, VEC)
    bank_square = _row_subgroup_sums(values * values, 4, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((4, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((4, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (8, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (8, groups))
    row_lanes: gl.constexpr = 16 if ADD else 8
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [row_lanes, 32 // row_lanes, 2], [WAVES, 1, 1], [0, 2, 1])
    merged = gl.convert_layout(gl.join(dot_parts, square_parts), merge_layout)
    totals = gl.sum(merged, 1)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    rows = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1])))
    scores = gl.where(rows <= 4, scores, -float('inf'))
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, row_layout)
    bank_probabilities = gl.gather(probabilities, bank_rows, 0)
    current_probability = gl.sum(gl.gather(probabilities, gl.full((1,), 4, gl.int32, row_layout), 0), 0)
    paired = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2))
    even, odd = gl.split(paired)
    v0, v2 = gl.split(even)
    v1, v3 = gl.split(odd)
    v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
    v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
    v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
    v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
    w0 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 0, gl.int32, row_layout), 0), 0)
    w1 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 1, gl.int32, row_layout), 0), 0)
    w2 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 2, gl.int32, row_layout), 0), 0)
    w3 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 3, gl.int32, row_layout), 0), 0)
    mixed = v0 * w0
    mixed = gl.fma(v1, w1, mixed)
    mixed = gl.fma(v2, w2, mixed)
    mixed = gl.fma(v3, w3, mixed)
    mixed = gl.fma(current, current_probability, mixed)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
        mixed = mixed * scale * output_weight.to(gl.float32)
    if WRITE:
        if gl.program_id(0) == 0:
            gl.store(Bank + token * SB0 + 4 * SB1 + h, current, h < H)
    gl.store(Out + token * H + h, mixed, owned)


@gluon.jit
def _pair_subgroup_sums(values, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr):
    groups: gl.constexpr = WAVES * 16
    width: gl.constexpr = 4 * VEC
    pieces = gl.reshape(values, (BLOCK // (groups * width), groups, width, 2))
    pieces = gl.permute(pieces, (1, 0, 2, 3))
    return gl.sum(gl.reshape(pieces, (groups, BLOCK // groups, 2)), 1)


@gluon.jit
def _fused_pair(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, ORDER: gl.constexpr):
    token = gl.program_id(ORDER)
    shard = gl.program_id(1 - ORDER)
    pair_layout: gl.constexpr = gl.BlockedLayout([VEC, 1], [64, 1], [WAVES, 1], [0, 1])
    feature_layout: gl.constexpr = gl.SliceLayout(1, pair_layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == shard)
    bank_values = gl.load(Bank + token * SB0 + h, h < H, 0).to(gl.float32)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if shard == 0:
            gl.store(Current + token * H + h, current, h < H)
    if WRITE:
        if shard == 0:
            gl.store(Bank + token * SB0 + SB1 + h, current, h < H)
    values = gl.join(bank_values, current)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    dot_parts = _pair_subgroup_sums(values * score_weight[:, None], BLOCK, WAVES, VEC)
    square_parts = _pair_subgroup_sums(values * values, BLOCK, WAVES, VEC)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [1, WAVES, 1], [1, 2, 0])
    partials = gl.convert_layout(gl.join(dot_parts, square_parts), stats_layout)
    totals = gl.sum(partials, 0)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, gl.SliceLayout(0, pair_layout))
    mixed = gl.sum(values * probabilities[None, :], 1)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
        mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


@gluon.jit
def _fused_pair_m16(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, ORDER: gl.constexpr):
    token = gl.program_id(ORDER)
    shard = gl.program_id(1 - ORDER)
    pair_layout: gl.constexpr = gl.BlockedLayout([VEC, 1], [64, 1], [WAVES, 1], [0, 1])
    feature_layout: gl.constexpr = gl.SliceLayout(1, pair_layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == shard)
    bank_values = gl.load(Bank + token * SB0 + h, h < H, 0).to(gl.float32)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if shard == 0:
            gl.store(Current + token * H + h, current, h < H)
    if WRITE:
        if shard == 0:
            gl.store(Bank + token * SB0 + SB1 + h, current, h < H)
    values = gl.join(bank_values, current)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    dot_parts = _pair_subgroup_sums(values * score_weight[:, None], BLOCK, WAVES, VEC)
    square_parts = _pair_subgroup_sums(values * values, BLOCK, WAVES, VEC)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [1, WAVES, 1], [1, 2, 0])
    partials = gl.convert_layout(gl.join(dot_parts, square_parts), stats_layout)
    totals = gl.sum(partials, 0)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, gl.SliceLayout(0, pair_layout))
    mixed = gl.sum(values * probabilities[None, :], 1)
    if NORM:
        scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
        mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


@gluon.constexpr_function
def _one_group_layout(warps):
    return gl.DistributedLinearLayout([[0, 0, 16, 0]], [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 0, 1], [1, 0, 0, 0]], [[0, 1 << bit, 0, 0] for bit in range(warps.bit_length() - 1)], [], [2, warps, 32, 2])


@gluon.jit
def _padded_softmax(scores):
    exponentials = gl.exp(scores - gl.max(scores, 0))
    return exponentials * gl.div_rn(1.0, gl.sum(exponentials, 0))


@gluon.jit
def _probability(probabilities, row: gl.constexpr):
    index = gl.full((1,), row, gl.int32, probabilities.type.layout)
    return gl.sum(gl.gather(probabilities, index, 0), 0)


@gluon.jit
def _rms_score(values, score_weight, inv_width, eps, AXIS: gl.constexpr):
    return gl.sum(values * score_weight, AXIS) * gl.rsqrt(gl.sum(values * values, AXIS) * inv_width + eps)


@gluon.constexpr_function
def _score_shared_layout(rows, warps):
    bases = [[0, 0, 1], [rows // 2, 0, 0]]
    bases += [[1 << bit, 0, 0] for bit in range(rows.bit_length() - 2)]
    bases += [[0, 1 << bit, 0] for bit in range(warps.bit_length() - 1)]
    return gl.SharedLinearLayout(bases, alignment=16)


@gluon.jit
def _softmax(scores):
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    return exp_scores * (1.0 / gl.sum(exp_scores, 0))


@gluon.jit
def _weight_at(weights, ROW: gl.constexpr):
    index = gl.full((1,), ROW, gl.int32, weights.type.layout)
    return gl.sum(gl.gather(weights, index, 0), 0)


def attention_residual_norm_m16_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
    order = int(m >= 4)
    grid = (triton.cdiv(h, 2048), m) if order else (m, triton.cdiv(h, 2048))
    _fused_pair_m16[grid](*common, order, num_warps=4, waves_per_eu=2)
    return (out, current, bank)


@gluon.jit
def _row_subgroup_sums_m1_2_8_16(values, ROWS: gl.constexpr, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, LANES: gl.constexpr):
    groups: gl.constexpr = WAVES * (64 // LANES)
    wave_width: gl.constexpr = LANES * VEC
    pieces = gl.reshape(values, (ROWS, BLOCK // (groups * wave_width), groups, wave_width))
    pieces = gl.permute(pieces, (0, 2, 1, 3))
    return gl.sum(gl.reshape(pieces, (ROWS, groups, BLOCK // groups)), 2)


@gluon.jit
def _ordered_bank_mix(values, bank_probabilities, current, current_probability, NV: gl.constexpr, BLOCK: gl.constexpr, feature_layout: gl.constexpr):
    if NV == 4:
        paired = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2))
        even, odd = gl.split(paired)
        v0, v2 = gl.split(even)
        v1, v3 = gl.split(odd)
        v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
        v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
        v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
        v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
        w0 = _probability(bank_probabilities, 0)
        w1 = _probability(bank_probabilities, 1)
        w2 = _probability(bank_probabilities, 2)
        w3 = _probability(bank_probabilities, 3)
    else:
        cubes = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2, 2))
        even, odd = gl.split(cubes)
        e0, e1 = gl.split(even)
        o0, o1 = gl.split(odd)
        v0, v4 = gl.split(e0)
        v2, v6 = gl.split(e1)
        v1, v5 = gl.split(o0)
        v3, v7 = gl.split(o1)
        v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
        w0 = _probability(bank_probabilities, 0)
        v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
        w1 = _probability(bank_probabilities, 1)
        v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
        w2 = _probability(bank_probabilities, 2)
        v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
        w3 = _probability(bank_probabilities, 3)
        v4 = gl.convert_layout(v4, feature_layout, assert_trivial=True)
        w4 = _probability(bank_probabilities, 4)
        v5 = gl.convert_layout(v5, feature_layout, assert_trivial=True)
        w5 = _probability(bank_probabilities, 5)
        v6 = gl.convert_layout(v6, feature_layout, assert_trivial=True)
        w6 = _probability(bank_probabilities, 6)
        v7 = gl.convert_layout(v7, feature_layout, assert_trivial=True)
        w7 = _probability(bank_probabilities, 7)
    mixed = v0 * w0
    mixed = gl.fma(v1, w1, mixed)
    mixed = gl.fma(v2, w2, mixed)
    mixed = gl.fma(v3, w3, mixed)
    if NV == 8:
        mixed = gl.fma(v4, w4, mixed)
        mixed = gl.fma(v5, w5, mixed)
        mixed = gl.fma(v6, w6, mixed)
        mixed = gl.fma(v7, w7, mixed)
    return gl.fma(current, current_probability, mixed)


@gluon.jit
def _fused_bank(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, NV: gl.constexpr):
    gl.static_assert(NV == 4 or NV == 8)
    gl.static_assert(NV != 8 or not WRITE, 'Eight-row publication uses the tiled producer')
    lanes: gl.constexpr = 16 if NV == 4 else 8
    groups: gl.constexpr = WAVES * (64 // lanes)
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, NV, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    if ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if gl.program_id(0) == 0:
            gl.store(Current + token * H + h, current, h < H)
    if not ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current_dot = _row_subgroup_sums_m1_2_8_16((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC, lanes)
    current_square = _row_subgroup_sums_m1_2_8_16((current * current)[None, :], 1, BLOCK, WAVES, VEC, lanes)
    values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    bank_dot = _row_subgroup_sums_m1_2_8_16(values * score_weight[None, :], NV, BLOCK, WAVES, VEC, lanes)
    bank_square = _row_subgroup_sums_m1_2_8_16(values * values, NV, BLOCK, WAVES, VEC, lanes)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((NV, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((NV, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (2 * NV, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (2 * NV, groups))
    row_lanes: gl.constexpr = (16 if ADD else 8) if NV == 4 else 4
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [row_lanes, 32 // row_lanes, 2], [WAVES, 1, 1], [0, 2, 1])
    merged = gl.convert_layout(gl.join(dot_parts, square_parts), merge_layout)
    totals = gl.sum(merged, 1)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    rows = gl.arange(0, 2 * NV, layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1])))
    scores = gl.where(rows <= NV, scores, -float('inf'))
    probabilities = _softmax(scores)
    probabilities = gl.convert_layout(probabilities, row_layout)
    bank_probabilities = gl.gather(probabilities, bank_rows, 0)
    current_probability = _probability(probabilities, NV)
    mixed = _ordered_bank_mix(values, bank_probabilities, current, current_probability, NV, BLOCK, feature_layout)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
        mixed = mixed * scale * output_weight.to(gl.float32)
    if WRITE:
        if gl.program_id(0) == 0:
            gl.store(Bank + token * SB0 + NV * SB1 + h, current, h < H)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m1_2_8_16_banks8_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    fuse_small = valid_rows in (1, 4) and block >= 2048
    fuse_eight = valid_rows == 8 and block >= 4096 and apply_output_norm and (not write_bank)
    waves = 8
    store_width = 1024 if m == 1 else 2048 if m <= 4 else 4096
    owners = triton.cdiv(h, store_width)
    common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, waves, 8, store_width)
    _fused_bank[owners, m](*common, valid_rows, num_warps=waves)
    return (out, current, bank)


@gluon.jit
def _fused_four_m1_2(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr):
    groups: gl.constexpr = WAVES * 4
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, 4, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
    current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
    if gl.program_id(0) == 0:
        gl.store(Current + token * H + h, current, h < H)
    current_dot = _row_subgroup_sums((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _row_subgroup_sums((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    bank_dot = _row_subgroup_sums(values * score_weight[None, :], 4, BLOCK, WAVES, VEC)
    bank_square = _row_subgroup_sums(values * values, 4, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((4, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((4, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (8, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (8, groups))
    row_lanes: gl.constexpr = 16 if ADD else 8
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [row_lanes, 32 // row_lanes, 2], [WAVES, 1, 1], [0, 2, 1])
    merged = gl.convert_layout(gl.join(dot_parts, square_parts), merge_layout)
    totals = gl.sum(merged, 1)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    rows = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1])))
    scores = gl.where(rows <= 4, scores, -float('inf'))
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores / gl.sum(exp_scores, 0)
    score_layout: gl.constexpr = probabilities.type.layout
    current_probability = gl.sum(gl.gather(probabilities, gl.full((1,), 4, gl.int32, score_layout), 0), 0)
    paired = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2))
    even, odd = gl.split(paired)
    v0, v2 = gl.split(even)
    v1, v3 = gl.split(odd)
    v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
    v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
    v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
    v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
    w0 = gl.sum(gl.gather(probabilities, gl.full((1,), 0, gl.int32, score_layout), 0), 0)
    w1 = gl.sum(gl.gather(probabilities, gl.full((1,), 1, gl.int32, score_layout), 0), 0)
    w2 = gl.sum(gl.gather(probabilities, gl.full((1,), 2, gl.int32, score_layout), 0), 0)
    w3 = gl.sum(gl.gather(probabilities, gl.full((1,), 3, gl.int32, score_layout), 0), 0)
    mixed = v0 * w0
    mixed = mixed + gl.fma(v1, w1, -0.0)
    mixed = mixed + gl.fma(v2, w2, -0.0)
    mixed = mixed + gl.fma(v3, w3, -0.0)
    mixed = gl.fma(current, current_probability, mixed)
    output_weight = gl.load(OutputWeight + h, h < H, 0)
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
    mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m1_2_banks4_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    _fused_four_m1_2[triton.cdiv(h, 2048), m](prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, score_eps, output_eps, block, 4, 8, 2048, num_warps=4)
    return (out, current, bank)


@gluon.jit
def _score_probabilities(dot, square, inverse_width, eps, VALID_ROWS: gl.constexpr):
    scores = dot * gl.rsqrt(square * inverse_width + eps)
    if VALID_ROWS + 1 < scores.shape[0]:
        rows = gl.arange(0, scores.shape[0], layout=scores.type.layout)
        scores = gl.where(rows <= VALID_ROWS, scores, -float('inf'))
    return _softmax(scores)


@gluon.jit
def _output_norm(values, weight, inverse_width, eps):
    scale = gl.rsqrt(gl.sum(values * values, 0) * inverse_width + eps)
    return values * scale * weight.to(gl.float32)


@gluon.jit
def _exchange_statistics(dot_parts, square_parts, EXCHANGE: gl.constexpr, AXIS: gl.constexpr, WAVES: gl.constexpr):
    partials = gl.convert_layout(gl.join(dot_parts, square_parts), EXCHANGE)
    totals = gl.sum(partials, AXIS)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    return gl.split(totals)


@gluon.jit
def _fast_mix_m1(C, Bank, S, OW, Out, H: gl.constexpr, SC: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, NORM: gl.constexpr, eps, BLOCK: gl.constexpr, ROWS: gl.constexpr, NW: gl.constexpr, VEC: gl.constexpr):
    m = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [NW], [0])
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    h = gl.arange(0, BLOCK, layout=layout)
    rows = gl.arange(0, ROWS, layout=sl)
    if NORM:
        ow = gl.load(OW + h, h < H, 0).to(gl.float32)
    scores = gl.load(S + m * (NV + 1) + rows, rows <= NV, -float('inf'))
    p = _softmax(scores)
    acc = gl.full((BLOCK,), 0, gl.float32, layout)
    for j in gl.static_range(NV):
        w = _probability(p, j)
        v = gl.load(Bank + m * SB0 + j * SB1 + h, h < H, 0).to(gl.float32)
        acc += w * v
    w = _probability(p, NV)
    v = gl.load(C + m * SC + h, h < H, 0).to(gl.float32)
    acc += w * v
    if NORM:
        acc = _output_norm(acc, ow, gl.div_rn(1.0, H * 1.0), eps)
    gl.store(Out + m * H + h, acc, h < H)


@gluon.jit
def _fused_pair_m1(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, ORDER: gl.constexpr):
    token = gl.program_id(ORDER)
    shard = gl.program_id(1 - ORDER)
    pair_layout: gl.constexpr = gl.BlockedLayout([VEC, 1], [64, 1], [WAVES, 1], [0, 1])
    feature_layout: gl.constexpr = gl.SliceLayout(1, pair_layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == shard)
    bank_values = gl.load(Bank + token * SB0 + h, h < H, 0).to(gl.float32)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if shard == 0:
            gl.store(Current + token * H + h, current, h < H)
    if WRITE:
        if shard == 0:
            gl.store(Bank + token * SB0 + SB1 + h, current, h < H)
    values = gl.join(bank_values, current)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    dot_parts = _pair_subgroup_sums(values * score_weight[:, None], BLOCK, WAVES, VEC)
    square_parts = _pair_subgroup_sums(values * values, BLOCK, WAVES, VEC)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [1, WAVES, 1], [1, 2, 0])
    dot, square = _exchange_statistics(dot_parts, square_parts, stats_layout, 0, WAVES)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    probabilities = _score_probabilities(dot, square, inverse_width, score_eps, 1)
    probabilities = gl.convert_layout(probabilities, gl.SliceLayout(0, pair_layout))
    mixed = gl.sum(values * probabilities[None, :], 1)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        mixed = _output_norm(mixed, output_weight, inverse_width, output_eps)
    gl.store(Out + token * H + h, mixed, owned)


@gluon.jit
def _mix_four_rows(values, bank_probabilities, current, current_probability, BLOCK: gl.constexpr, FEATURE_LAYOUT: gl.constexpr):
    paired = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2))
    even, odd = gl.split(paired)
    v0, v2 = gl.split(even)
    v1, v3 = gl.split(odd)
    v0 = gl.convert_layout(v0, FEATURE_LAYOUT, assert_trivial=True)
    v1 = gl.convert_layout(v1, FEATURE_LAYOUT, assert_trivial=True)
    v2 = gl.convert_layout(v2, FEATURE_LAYOUT, assert_trivial=True)
    v3 = gl.convert_layout(v3, FEATURE_LAYOUT, assert_trivial=True)
    w0 = _probability(bank_probabilities, 0)
    w1 = _probability(bank_probabilities, 1)
    w2 = _probability(bank_probabilities, 2)
    w3 = _probability(bank_probabilities, 3)
    mixed = v0 * w0
    mixed = gl.fma(v1, w1, mixed)
    mixed = gl.fma(v2, w2, mixed)
    mixed = gl.fma(v3, w3, mixed)
    mixed = gl.fma(current, current_probability, mixed)
    return mixed


@gluon.jit
def _fused_four_m1(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr):
    groups: gl.constexpr = WAVES * 4
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, 4, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    if ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if gl.program_id(0) == 0:
            gl.store(Current + token * H + h, current, h < H)
    if not ADD:
        score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current_dot = _row_subgroup_sums((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _row_subgroup_sums((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    if ADD:
        values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    else:
        v0 = buffer_load(Bank + token * SB0, h, h < H, 0).to(gl.float32)
        v1 = buffer_load(Bank + token * SB0 + SB1, h, h < H, 0).to(gl.float32)
        v2 = buffer_load(Bank + token * SB0 + 2 * SB1, h, h < H, 0).to(gl.float32)
        v3 = buffer_load(Bank + token * SB0 + 3 * SB1, h, h < H, 0).to(gl.float32)
        packed = gl.join(gl.join(v0, v1), gl.join(v2, v3))
        values = gl.reshape(gl.permute(packed, (2, 1, 0)), (4, BLOCK))
        values = gl.convert_layout(values, layout, assert_trivial=True)
    bank_dot = _row_subgroup_sums(values * score_weight[None, :], 4, BLOCK, WAVES, VEC)
    bank_square = _row_subgroup_sums(values * values, 4, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((4, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((4, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (8, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (8, groups))
    row_lanes: gl.constexpr = 16 if ADD else 8
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [row_lanes, 32 // row_lanes, 2], [WAVES, 1, 1], [0, 2, 1])
    dot, square = _exchange_statistics(dot_parts, square_parts, merge_layout, 1, WAVES)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    probabilities = _score_probabilities(dot, square, inverse_width, score_eps, 4)
    probabilities = gl.convert_layout(probabilities, row_layout)
    bank_probabilities = gl.gather(probabilities, bank_rows, 0)
    current_probability = _probability(probabilities, 4)
    mixed = _mix_four_rows(values, bank_probabilities, current, current_probability, BLOCK, feature_layout)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        mixed = _output_norm(mixed, output_weight, inverse_width, output_eps)
    if WRITE:
        if gl.program_id(0) == 0:
            gl.store(Bank + token * SB0 + 4 * SB1 + h, current, h < H)
    gl.store(Out + token * H + h, mixed, owned)


@gluon.jit
def _eight_subgroup_sums_m1(values, ROWS: gl.constexpr, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr):
    groups: gl.constexpr = WAVES * 8
    width: gl.constexpr = 8 * VEC
    pieces = gl.reshape(values, (ROWS, BLOCK // (groups * width), groups, width))
    pieces = gl.permute(pieces, (0, 2, 1, 3))
    return gl.sum(gl.reshape(pieces, (ROWS, groups, BLOCK // groups)), 2)


@gluon.jit
def _fused_eight(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, ORDER: gl.constexpr):
    groups: gl.constexpr = WAVES * 8
    token = gl.program_id(ORDER)
    shard = gl.program_id(1 - ORDER)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == shard)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if shard == 0:
            gl.store(Current + token * H + h, current, h < H)
    current_dot = _eight_subgroup_sums_m1((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _eight_subgroup_sums_m1((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    bank_rows = gl.arange(0, 8, layout=row_layout)
    values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    bank_dot = _eight_subgroup_sums_m1(values * score_weight[None, :], 8, BLOCK, WAVES, VEC)
    bank_square = _eight_subgroup_sums_m1(values * values, 8, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot += gl.full((8, groups), 0, gl.float32, bank_dot.type.layout)
    current_square += gl.full((8, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (16, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (16, groups))
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [WAVES, 1, 1], [0, 2, 1])
    dot, square = _exchange_statistics(dot_parts, square_parts, merge_layout, 1, WAVES)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    probabilities = _score_probabilities(dot, square, inverse_width, score_eps, 8)
    probabilities = gl.convert_layout(probabilities, row_layout)
    packed = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2, 2))
    even, odd = gl.split(packed)
    e0, e2 = gl.split(even)
    o1, o3 = gl.split(odd)
    v0, v4 = gl.split(e0)
    v2, v6 = gl.split(e2)
    v1, v5 = gl.split(o1)
    v3, v7 = gl.split(o3)
    v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
    v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
    v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
    v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
    v4 = gl.convert_layout(v4, feature_layout, assert_trivial=True)
    v5 = gl.convert_layout(v5, feature_layout, assert_trivial=True)
    v6 = gl.convert_layout(v6, feature_layout, assert_trivial=True)
    v7 = gl.convert_layout(v7, feature_layout, assert_trivial=True)
    mixed = v0 * _probability(probabilities, 0)
    mixed = gl.fma(v1, _probability(probabilities, 1), mixed)
    mixed = gl.fma(v2, _probability(probabilities, 2), mixed)
    mixed = gl.fma(v3, _probability(probabilities, 3), mixed)
    mixed = gl.fma(v4, _probability(probabilities, 4), mixed)
    mixed = gl.fma(v5, _probability(probabilities, 5), mixed)
    mixed = gl.fma(v6, _probability(probabilities, 6), mixed)
    mixed = gl.fma(v7, _probability(probabilities, 7), mixed)
    mixed = gl.fma(current, _probability(probabilities, 8), mixed)
    output_weight = gl.load(OutputWeight + h, h < H, 0)
    mixed = _output_norm(mixed, output_weight, inverse_width, output_eps)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m1_banks1_4_8_modes4_6(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device) if has_addend else prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    source_strides = (h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1))
    mix_strides = (h, current.stride(0), bank.stride(0), bank.stride(1))
    if valid_rows in (1, 4) and block >= 2048:
        common = (prefix, addend, current, bank, score_weight, output_weight, out, *source_strides, has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
        if valid_rows == 1:
            order = int(m >= 4)
            grid = (triton.cdiv(h, 2048), m) if order else (m, triton.cdiv(h, 2048))
            _fused_pair_m1[grid](*common, order, num_warps=4, waves_per_eu=2)
        else:
            _fused_four_m1[triton.cdiv(h, 2048), m](*common, num_warps=4)
    else:
        if valid_rows == 8 and apply_output_norm and (not write_bank) and (block == 8192) and (m <= 16):
            mix_width = 1024
            _fused_eight[triton.cdiv(h, mix_width), m](prefix, addend, current, bank, score_weight, output_weight, out, *source_strides, has_addend, score_eps, output_eps, block, 8, 8, mix_width, 1, num_warps=8, waves_per_eu=1)
        else:
            scores = torch.empty((m, valid_rows + 1), dtype=torch.float32, device=prefix.device)
            _fast_scores[m, valid_rows + 1](prefix, addend, current, bank, score_weight, scores, *source_strides, valid_rows, has_addend, write_bank, score_eps, block, 4, 8, num_warps=4)
            _fast_mix_m1[m,](current, bank, scores, output_weight, out, *mix_strides, valid_rows, apply_output_norm, output_eps, block, triton.next_power_of_2(valid_rows + 1), 4, 8, num_warps=4)
    return (out, current, bank)


def attention_residual_norm_m1_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
    order = int(m >= 4)
    grid = (triton.cdiv(h, 2048), m) if order else (m, triton.cdiv(h, 2048))
    _fused_pair_m16[grid](*common, order, num_warps=4, waves_per_eu=2)
    return (out, current, bank)


@gluon.jit
def _fused_eight_normalized(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr):
    groups: gl.constexpr = WAVES * 8
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, 8, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current_dot = _eight_subgroup_sums((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _eight_subgroup_sums((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    bank_dot = _eight_subgroup_sums(values * score_weight[None, :], 8, BLOCK, WAVES, VEC)
    bank_square = _eight_subgroup_sums(values * values, 8, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((8, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((8, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (16, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (16, groups))
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [WAVES, 1, 1], [0, 2, 1])
    merged = gl.convert_layout(gl.join(dot_parts, square_parts), merge_layout)
    totals = gl.sum(merged, 1)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1])))
    scores = gl.where(rows <= 8, scores, -float('inf'))
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, row_layout)
    bank_probabilities = gl.gather(probabilities, bank_rows, 0)
    current_probability = gl.sum(gl.gather(probabilities, gl.full((1,), 8, gl.int32, row_layout), 0), 0)
    packed = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2, 2))
    even, odd = gl.split(packed)
    ev0, ev2 = gl.split(even)
    od1, od3 = gl.split(odd)
    v0, v4 = gl.split(ev0)
    v2, v6 = gl.split(ev2)
    v1, v5 = gl.split(od1)
    v3, v7 = gl.split(od3)
    v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
    v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
    v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
    v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
    v4 = gl.convert_layout(v4, feature_layout, assert_trivial=True)
    v5 = gl.convert_layout(v5, feature_layout, assert_trivial=True)
    v6 = gl.convert_layout(v6, feature_layout, assert_trivial=True)
    v7 = gl.convert_layout(v7, feature_layout, assert_trivial=True)
    w0 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 0, gl.int32, row_layout), 0), 0)
    w1 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 1, gl.int32, row_layout), 0), 0)
    w2 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 2, gl.int32, row_layout), 0), 0)
    w3 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 3, gl.int32, row_layout), 0), 0)
    w4 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 4, gl.int32, row_layout), 0), 0)
    w5 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 5, gl.int32, row_layout), 0), 0)
    w6 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 6, gl.int32, row_layout), 0), 0)
    w7 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 7, gl.int32, row_layout), 0), 0)
    mixed = v0 * w0
    mixed = gl.fma(v1, w1, mixed)
    mixed = gl.fma(v2, w2, mixed)
    mixed = gl.fma(v3, w3, mixed)
    mixed = gl.fma(v4, w4, mixed)
    mixed = gl.fma(v5, w5, mixed)
    mixed = gl.fma(v6, w6, mixed)
    mixed = gl.fma(v7, w7, mixed)
    mixed = gl.fma(current, current_probability, mixed)
    output_weight = gl.load(OutputWeight + h, h < H, 0)
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
    mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m2_8_banks1_4_8_modes4_6(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device) if has_addend else prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    if valid_rows in (1, 4) and block >= 2048:
        common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
        if valid_rows == 1:
            order = int(m >= 4)
            grid = (triton.cdiv(h, 2048), m) if order else (m, triton.cdiv(h, 2048))
            _fused_pair[grid](*common, order, num_warps=4, waves_per_eu=2)
        else:
            _fused_four[triton.cdiv(h, 2048), m](*common, num_warps=4)
    else:
        if valid_rows == 8 and apply_output_norm and (not write_bank) and (block == 8192):
            store_width = 2048
            _fused_eight_normalized[triton.cdiv(h, store_width), m](prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, score_eps, output_eps, block, 8, 8, store_width, num_warps=8)
        else:
            scores = torch.empty((m, valid_rows + 1), dtype=torch.float32, device=prefix.device)
            _fast_scores[m, valid_rows + 1](prefix, addend, current, bank, score_weight, scores, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, score_eps, block, 4, 8, num_warps=4)
            _fast_mix[m,](current, bank, scores, output_weight, out, h, current.stride(0), bank.stride(0), bank.stride(1), valid_rows, apply_output_norm, output_eps, block, triton.next_power_of_2(valid_rows + 1), 4, 8, num_warps=4)
    return (out, current, bank)


@gluon.jit
def _normalize_output(mixed, weight, inverse_width, eps):
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + eps)
    return mixed * scale * weight.to(gl.float32)


@gluon.jit
def _subgroup_sums(values, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, LANES: gl.constexpr):

    groups: gl.constexpr = WAVES * (64 // LANES)
    width: gl.constexpr = LANES * VEC
    rows: gl.constexpr = values.type.numel // BLOCK
    pieces = gl.reshape(values, (rows, BLOCK // (groups * width), groups, width))
    pieces = gl.permute(pieces, (0, 2, 1, 3))
    partials = gl.sum(gl.reshape(pieces, (rows, groups, BLOCK // groups)), 2)
    if len(values.shape) == 1:
        return gl.reshape(partials, (groups,))
    else:
        return partials


@gluon.jit
def _reduce_statistics(dot, square, AXIS: gl.constexpr, LAYOUT: gl.constexpr, WAVES: gl.constexpr):
    merged = gl.convert_layout(gl.join(dot, square), LAYOUT)
    totals = gl.sum(merged, AXIS)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    return gl.split(totals)


@gluon.jit
def _fused_pair_m2(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, ORDER: gl.constexpr):
    token = gl.program_id(ORDER)
    shard = gl.program_id(1 - ORDER)
    pair_layout: gl.constexpr = gl.BlockedLayout([VEC, 1], [64, 1], [WAVES, 1], [0, 1])
    feature_layout: gl.constexpr = gl.SliceLayout(1, pair_layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == shard)
    bank_values = gl.load(Bank + token * SB0 + h, h < H, 0).to(gl.float32)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if shard == 0:
            gl.store(Current + token * H + h, current, h < H)
    if WRITE:
        if shard == 0:
            gl.store(Bank + token * SB0 + SB1 + h, current, h < H)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    bank_dot = _subgroup_sums(bank_values * score_weight, BLOCK, WAVES, VEC, 4)
    bank_square = _subgroup_sums(bank_values * bank_values, BLOCK, WAVES, VEC, 4)
    current_dot = _subgroup_sums(current * score_weight, BLOCK, WAVES, VEC, 4)
    current_square = _subgroup_sums(current * current, BLOCK, WAVES, VEC, 4)
    dot_parts = gl.join(bank_dot, current_dot)
    square_parts = gl.join(bank_square, current_square)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [4, 8, 2], [1, WAVES, 1], [1, 2, 0])
    dot, square = _reduce_statistics(dot_parts, square_parts, 0, stats_layout, WAVES)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    probabilities = _softmax(scores)
    probabilities = gl.convert_layout(probabilities, gl.SliceLayout(0, pair_layout))
    values = gl.join(bank_values, current)
    mixed = gl.sum(values * probabilities[None, :], 1)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
        mixed = _normalize_output(mixed, output_weight, inverse_width, output_eps)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m2_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):

    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    input_strides = (prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1))
    mix_strides = (current.stride(0), bank.stride(0), bank.stride(1))
    small_shard, mix_shard = (2048, 4096)
    shards = triton.cdiv(h, small_shard)
    common = (prefix, addend, current, bank, score_weight, output_weight, out, h, *input_strides, has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, small_shard)
    order = int(m >= 4)
    grid = (shards, m) if order else (m, shards)
    _fused_pair_m2[grid](*common, order, num_warps=4, waves_per_eu=2)
    return (out, current, bank)


@gluon.constexpr_function
def _exchange_layout_m32_64_128_256(rows, warps, row_lanes):
    row_bits = row_lanes.bit_length() - 1
    wave_bits = warps.bit_length() - 1
    registers = [[0, 0, 1]] + [[1 << bit, 0, 0] for bit in range(row_bits, rows.bit_length() - 1)]
    lanes = [[1 << bit, 0, 0] for bit in range(row_bits)]
    lanes += [[0, 1 << bit, 0] for bit in range(wave_bits)]
    lanes += [[0, 0, 0]] * (6 - row_bits - wave_bits)
    return gl.DistributedLinearLayout(registers, lanes, [[0, 0, 0]] * wave_bits, [], [rows, warps, 2])


@gluon.constexpr_function
def _wave_first_exchange_layout(rows, warps):
    wave_bits = warps.bit_length() - 1
    row_bits = rows.bit_length() - 1
    lanes = [[0, 1 << bit, 0] for bit in range(wave_bits)]
    lanes += [[1 << bit, 0, 0] for bit in range(row_bits)]
    lanes += [[0, 0, 0]] * (6 - wave_bits - row_bits)
    return gl.DistributedLinearLayout([[0, 0, 1]], lanes, [[0, 0, 0]] * wave_bits, [], [rows, warps, 2])


@gluon.jit
def _native_square_partials(values, BH: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr):
    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [NW, 1, 1], [0, 1, 2])
    local = values.to(gl.bfloat16).reshape((BV, BH // (NW * 512), NW, 64, 8))
    local = local.permute((0, 2, 3, 1, 4)).reshape((BV * NW * 64, 1, BH // (NW * 64)))
    lhs = gl.convert_layout(local, gl.DotOperandLayout(0, dot_layout, 0))
    rhs = gl.convert_layout(local.permute((0, 2, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    acc = gl.full((BV * NW * 64, 1, 1), 0, gl.float32, dot_layout)
    return gl.dot_fma(lhs, rhs, acc).reshape((BV, NW, 64))


@gluon.jit
def _staged_bank_partials(values, cw, BH: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr):
    shape: gl.constexpr = (BV, BH // (NW * 512), NW, 64, 8)
    square = _native_square_partials(values, BH, NW, BV)
    dot = gl.sum(gl.sum((values * cw[None, :]).reshape(shape), 1), 3)
    square = gl.convert_layout(square, dot.type.layout, assert_trivial=True)
    joint = gl.join(dot, square)
    for stage in gl.static_range(3):
        joint = gl.convert_layout(joint, _butterfly_layout(stage, NW, BV))
        joint = gl.sum(joint.reshape((BV, NW, 2, 32 >> stage, 2)), 2)
    return gl.sum(joint, 2)


@gluon.jit
def _segmented_scores(values, current, cw, inv_width, eps, BH: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr):
    joint = _staged_bank_partials(values, cw, BH, NW, BV)
    current_joint = _current_stat_partials(current, cw, BH, NW)
    bank_storage = gl.allocate_shared_memory(gl.float32, (BV, NW, 2), _score_shared_layout(BV, NW), joint)
    current_storage = gl.allocate_shared_memory(gl.float32, (NW, 2), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), current_joint)
    exchange_layout: gl.constexpr = _wave_first_exchange_layout(BV, NW)
    bank_partials = bank_storage.load(exchange_layout)
    current_partials = current_storage.load(gl.SliceLayout(0, exchange_layout))
    bank_dot, bank_square = gl.split(gl.sum(bank_partials, 1))
    current_dot, current_square = gl.split(gl.sum(current_partials, 0))
    bank_score = bank_dot * gl.rsqrt(bank_square * inv_width + eps)
    current_score = current_dot * gl.rsqrt(current_square * inv_width + eps)
    bank_score = gl.convert_layout(bank_score, gl.BlockedLayout([1], [64], [NW], [0]))
    padded_current = gl.full((BV,), current_score, gl.float32, bank_score.type.layout)
    scores = gl.join(bank_score, padded_current).permute((1, 0)).reshape((2 * BV,))
    return (scores, (bank_storage, current_storage))


@gluon.jit
def _attention_residual_kernel(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BH: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr, COMPACT_BANK: gl.constexpr, REPLICAS: gl.constexpr=1):
    token = gl.program_id(0) // REPLICAS
    shard = gl.program_id(0) % REPLICAS
    layout: gl.constexpr = gl.BlockedLayout([2 if NV == 8 and REPLICAS > 1 else 1, 8], [1, 64], [1, NW], [1, 0])
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    channels = gl.arange(0, BH, layout=gl.SliceLayout(0, layout))
    valid_channel = channels < H
    store_channel = valid_channel
    if REPLICAS > 1:
        store_channel = valid_channel & (channels >= shard * (BH // REPLICAS)) & (channels < (shard + 1) * (BH // REPLICAS))
    current = gl.load(Prefix + token * SP + channels, valid_channel, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + channels, valid_channel, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        gl.store(Current + token * H + channels, current, store_channel, cache_modifier='.cs' if NV == 8 else '')
    if WRITE:
        gl.store(Bank + token * SB0 + NV * SB1 + channels, current, store_channel)
    values = gl.load(Bank + token * SB0 + rows[:, None] * SB1 + channels[None, :], (rows[:, None] < NV) & valid_channel[None, :], 0).to(gl.float32)
    cw = gl.load(ScoreWeight + channels, valid_channel, 0)
    inv_width = gl.div_rn(1.0, 1.0 * H)
    if not COMPACT_BANK:
        values = gl.where(rows[:, None] == NV, current[None, :], values)
    if COMPACT_BANK:
        scores, score_storage = _segmented_scores(values, current, cw, inv_width, score_eps, BH, BV, NW)
        all_rows = gl.arange(0, BV * 2, layout=scores.type.layout)
        scores = gl.where(all_rows <= NV, scores, -float('inf'))
    else:
        scores = _rms_score(values, cw[None, :], inv_width, score_eps, 1)
        scores = gl.where(rows <= NV, scores, -float('inf'))
    weights = _padded_softmax(scores)
    if COMPACT_BANK:
        mixed = gl.full((BH,), 0, gl.float32, gl.SliceLayout(0, layout))
        for row in gl.static_range(BV):
            mixed = mixed + _weight_at(weights, row) * _bank_row(values, row)
        mixed = mixed + _weight_at(weights, NV) * current
    else:
        mixed = gl.sum(weights[:, None] * values, 0)
    if NV == 8 and NORM:
        ow = gl.load(OutputWeight + channels, valid_channel, 0)
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inv_width + output_eps)
    if NV != 8:
        ow = gl.load(OutputWeight + channels, valid_channel, 0).to(gl.float32)
    mixed = mixed * scale * ow.to(gl.float32)
    if COMPACT_BANK:
        score_storage[0]._keep_alive()
        score_storage[1]._keep_alive()
    gl.store(Out + token * H + channels, mixed, store_channel, cache_modifier='.cs' if NV == 8 else '')


@gluon.jit
def _exact_mix(values, current, weights, BV: gl.constexpr):
    row_layout: gl.constexpr = gl.SliceLayout(1, values.type.layout)
    rows = gl.arange(0, BV, layout=row_layout)
    bank_weights = gl.full((BV,), 0, gl.float32, row_layout)
    for row in gl.static_range(BV):
        bank_weights = gl.where(rows == row, _weight_at(weights, row), bank_weights)
    return gl.sum(bank_weights[:, None] * values, 0) + _weight_at(weights, BV) * current


@gluon.jit
def _exact_store(Out, ow, mixed, scale, token, shard, SIZE: gl.constexpr, OFFSET: gl.constexpr, REPLICAS: gl.constexpr, NORM: gl.constexpr):
    channels = OFFSET + gl.arange(0, SIZE, layout=mixed.type.layout)
    mixed = mixed * scale * ow.to(gl.float32)
    mask = (channels >= shard * (7168 // REPLICAS)) & (channels < (shard + 1) * (7168 // REPLICAS))
    gl.store(Out + token * 7168 + channels, mixed, mask)


@gluon.jit
def _load_output_weight(Weight, channels, USE_BUFFER: gl.constexpr):
    return gl.load(Weight + channels)


@gluon.jit
def _snapshot_chunk(Prefix, Bank, CW, token, shard, SP: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, REPLICAS: gl.constexpr, SIZE: gl.constexpr, OFFSET: gl.constexpr, VECTOR: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [1, 64], [1, NW], [1, 0])
    channels = OFFSET + gl.arange(0, SIZE, layout=gl.SliceLayout(0, layout))
    rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    current = gl.load(Prefix + token * SP + channels).to(gl.float32)
    cw = gl.load(CW + channels)
    cur_shape: gl.constexpr = (SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    native_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [NW, 1, 1], [0, 1, 2])
    cur_dot = gl.sum(gl.sum((current * cw).reshape(cur_shape), 0), 2)
    cur_local = current.to(gl.bfloat16).reshape(cur_shape).permute((1, 2, 0, 3))
    cur_local = cur_local.reshape((NW * 64, 1, SIZE // (NW * 64)))
    cur_lhs = gl.convert_layout(cur_local, gl.DotOperandLayout(0, native_layout, 0))
    cur_rhs = gl.convert_layout(cur_local.permute((0, 2, 1)), gl.DotOperandLayout(1, native_layout, 0))
    cur_acc = gl.full((NW * 64, 1, 1), 0, gl.float32, native_layout)
    cur_square = gl.dot_fma(cur_lhs, cur_rhs, cur_acc).reshape((NW, 64))
    cur_square = gl.convert_layout(cur_square, cur_dot.type.layout, assert_trivial=True)
    current_stats = gl.join(cur_dot, cur_square)
    values = buffer_load(Bank + token * SB0, rows[:, None] * SB1 + channels[None, :]).to(gl.float32)
    bank_shape: gl.constexpr = (4, SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    dot = gl.sum(gl.sum((values * cw[None, :]).reshape(bank_shape), 1), 3)
    local = values.to(gl.bfloat16).reshape(bank_shape).permute((0, 2, 3, 1, 4))
    local = local.reshape((4 * NW * 64, 1, SIZE // (NW * 64)))
    lhs = gl.convert_layout(local, gl.DotOperandLayout(0, native_layout, 0))
    rhs = gl.convert_layout(local.permute((0, 2, 1)), gl.DotOperandLayout(1, native_layout, 0))
    acc = gl.full((4 * NW * 64, 1, 1), 0, gl.float32, native_layout)
    square = gl.dot_fma(lhs, rhs, acc).reshape((4, NW, 64))
    square = gl.convert_layout(square, dot.type.layout, assert_trivial=True)
    bank_stats = gl.join(dot, square)
    store_channel = (channels >= shard * (7168 // REPLICAS)) & (channels < (shard + 1) * (7168 // REPLICAS))
    gl.store(Bank + token * SB0 + 4 * SB1 + channels, current, store_channel)
    return (values, current, bank_stats, current_stats)


@gluon.jit
def _snapshot_kernel(Prefix, Bank, ScoreWeight, OutputWeight, Out, SP: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps):
    NW: gl.constexpr = 4
    BV: gl.constexpr = 4
    token = gl.program_id(0)
    shard = 0
    v0, c0, b0, s0 = _snapshot_chunk(Prefix, Bank, ScoreWeight, token, shard, SP, SB0, SB1, 1, 2048, 0, 8, 4)
    v1, c1, b1, s1 = _snapshot_chunk(Prefix, Bank, ScoreWeight, token, shard, SP, SB0, SB1, 1, 2048, 2048, 8, 4)
    v2, c2, b2, s2 = _snapshot_chunk(Prefix, Bank, ScoreWeight, token, shard, SP, SB0, SB1, 1, 2048, 4096, 8, 4)
    v3, c3, b3, s3 = _snapshot_chunk(Prefix, Bank, ScoreWeight, token, shard, SP, SB0, SB1, 1, 1024, 6144, 4, 4)
    joint = b0
    current_joint = s0
    joint = joint + gl.convert_layout(b1, joint.type.layout, assert_trivial=True)
    current_joint = current_joint + gl.convert_layout(s1, current_joint.type.layout, assert_trivial=True)
    joint = joint + gl.convert_layout(b2, joint.type.layout, assert_trivial=True)
    current_joint = current_joint + gl.convert_layout(s2, current_joint.type.layout, assert_trivial=True)
    joint = joint + gl.convert_layout(b3, joint.type.layout, assert_trivial=True)
    current_joint = current_joint + gl.convert_layout(s3, current_joint.type.layout, assert_trivial=True)
    inv_width = gl.div_rn(1.0, 7168.0)
    for stage in gl.static_range(2):
        joint = gl.convert_layout(joint, _butterfly_layout(stage, NW, BV))
        joint = gl.sum(joint.reshape((BV, NW, 2, 32 >> stage, 2)), 2)
    joint = gl.sum(joint, 2)
    current_joint = gl.convert_layout(current_joint, _current_group_layout(NW))
    current_joint = gl.sum(current_joint.reshape((NW, 2, 32, 2)), 1)
    current_joint = gl.sum(current_joint, 1)
    bank_storage = gl.allocate_shared_memory(gl.float32, (BV, NW, 2), _score_shared_layout(BV, NW), joint)
    current_storage = gl.allocate_shared_memory(gl.float32, (NW, 2), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), current_joint)
    exchange_layout: gl.constexpr = _exchange_layout_m32_64_128_256(BV, NW, BV)
    bank_partials = bank_storage.load(exchange_layout)
    current_partials = current_storage.load(gl.SliceLayout(0, exchange_layout))
    bank_dp, bank_sp = gl.split(bank_partials)
    current_dp, current_sp = gl.split(current_partials)
    bank_score = gl.sum(bank_dp, 1) * gl.rsqrt(gl.sum(bank_sp, 1) * inv_width + score_eps)
    current_score = gl.sum(current_dp, 0) * gl.rsqrt(gl.sum(current_sp, 0) * inv_width + score_eps)
    padded_current = gl.full((BV,), current_score, gl.float32, bank_score.type.layout)
    scores = gl.join(bank_score, padded_current).permute((1, 0)).reshape((2 * BV,))
    rows = gl.arange(0, 2 * BV, layout=scores.type.layout)
    weights = _padded_softmax(gl.where(rows <= BV, scores, -float('inf')))
    ow0 = 0.0
    ow1 = 0.0
    ow2 = 0.0
    ow3 = 0.0
    ow0 = _load_output_weight(OutputWeight, 0 + gl.arange(0, 2048, layout=c0.type.layout), False)
    mix0 = _exact_mix(v0, c0, weights, 4)
    ow1 = _load_output_weight(OutputWeight, 2048 + gl.arange(0, 2048, layout=c1.type.layout), False)
    mix1 = _exact_mix(v1, c1, weights, 4)
    ow2 = _load_output_weight(OutputWeight, 4096 + gl.arange(0, 2048, layout=c2.type.layout), False)
    mix2 = _exact_mix(v2, c2, weights, 4)
    ow3 = _load_output_weight(OutputWeight, 6144 + gl.arange(0, 1024, layout=c3.type.layout), False)
    mix3 = _exact_mix(v3, c3, weights, 4)
    scale = 1.0
    square0 = _exact_output_partials(mix0, 2048, 8, 4)
    square1 = _exact_output_partials(mix1, 2048, 8, 4)
    square2 = _exact_output_partials(mix2, 2048, 8, 4)
    square3 = _exact_output_partials(mix3, 1024, 4, 4)
    partials = square0
    partials = partials + gl.convert_layout(square1, partials.type.layout, assert_trivial=True)
    partials = partials + gl.convert_layout(square2, partials.type.layout, assert_trivial=True)
    partials = partials + gl.convert_layout(square3, partials.type.layout, assert_trivial=True)
    scale = gl.rsqrt(gl.sum(gl.sum(partials, 1), 0) * inv_width + output_eps)
    bank_storage._keep_alive()
    current_storage._keep_alive()
    _exact_store(Out, ow0, mix0, scale, token, 0, 2048, 0, 1, NORM)
    _exact_store(Out, ow1, mix1, scale, token, 0, 2048, 2048, 1, NORM)
    _exact_store(Out, ow2, mix2, scale, token, 0, 2048, 4096, 1, NORM)
    _exact_store(Out, ow3, mix3, scale, token, 0, 1024, 6144, 1, NORM)


def attention_residual_norm_m32_64_128_256_banks1_8_modes4_6(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device) if has_addend else prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    if h == 7168 and valid_rows == 4 and (not has_addend) and write_bank:
        _snapshot_kernel[m,](prefix, bank, score_weight, output_weight, out, prefix.stride(0), bank.stride(0), bank.stride(1), apply_output_norm, score_eps, output_eps, num_warps=4, waves_per_eu=2)
        return (out, current, bank)
    use_exact_width = h == 7168 and m in (32, 64) and (not write_bank) and (valid_rows == 1 and (not has_addend) or (valid_rows == 4 and has_addend))
    num_warps = 8 if valid_rows > 4 else 4
    compact_bank = valid_rows == 8
    bank_tile = triton.next_power_of_2(valid_rows if compact_bank else valid_rows + 1)
    replicas = 2 if m == 32 and valid_rows == 8 else 1
    grid = (m * replicas,)
    waves_per_eu = (1 if m == 32 else 2) if valid_rows == 8 else 0
    _attention_residual_kernel[grid](prefix, addend, bank, score_weight, output_weight, current, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, apply_output_norm, score_eps, output_eps, triton.next_power_of_2(h), bank_tile, num_warps, compact_bank, REPLICAS=replicas, num_warps=num_warps, waves_per_eu=waves_per_eu)
    return (out, current, bank)


@gluon.jit
def _exact_tile_load(Prefix, Addend, Bank, ScoreWeight, token, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, OFFSET: gl.constexpr, WIDTH: gl.constexpr, PACK: gl.constexpr):
    row_registers: gl.constexpr = 4 if WRITE and (not ADD) else 2
    layout: gl.constexpr = gl.BlockedLayout([row_registers, PACK], [1, 64], [1, 4], [1, 0])
    channels = OFFSET + gl.arange(0, WIDTH, layout=gl.SliceLayout(0, layout))
    rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    current = gl.load(Prefix + token * SP + channels).to(gl.float32)
    addend = gl.load(Addend + token * SA + channels).to(gl.float32)
    current = (current + addend).to(gl.bfloat16).to(gl.float32)
    cw = gl.load(ScoreWeight + channels)
    values = gl.load(Bank + token * SB0 + rows[:, None] * SB1 + channels[None, :]).to(gl.float32)
    return (values, current, cw, channels)


@gluon.jit
def _exact_square(values, accum, WIDTH: gl.constexpr, PACK: gl.constexpr):
    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [4, 1, 1], [0, 1, 2])
    local = values.to(gl.bfloat16).reshape((4, WIDTH // (256 * PACK), 4, 64, PACK))
    local = local.permute((0, 2, 3, 1, 4)).reshape((1024, WIDTH // 256))
    lhs = gl.convert_layout(local.reshape((1024, 1, WIDTH // 256)), gl.DotOperandLayout(0, dot_layout, 0))
    rhs = gl.convert_layout(local.reshape((1024, WIDTH // 256, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    return gl.dot_fma(lhs, rhs, accum)


@gluon.jit
def _exact_dot(values, cw, WIDTH: gl.constexpr, PACK: gl.constexpr):
    products = (values * cw[None, :]).reshape((4, WIDTH // (256 * PACK), 4, 64, PACK))
    return gl.sum(gl.sum(products, 1), 3)


@gluon.jit
def _exact_current(current, cw, WIDTH: gl.constexpr, PACK: gl.constexpr):
    shape: gl.constexpr = (WIDTH // (256 * PACK), 4, 64, PACK)
    dot = gl.sum(gl.sum((current * cw).reshape(shape), 0), 2)
    square = gl.sum(gl.sum((current * current).reshape(shape), 0), 2)
    return gl.join(dot, square)


@gluon.jit
def _exact_mix_m32_64(values, current, weights):
    layout: gl.constexpr = values.type.layout
    rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    bank_weights = gl.full((4,), 0, gl.float32, gl.SliceLayout(1, layout))
    for row in gl.static_range(4):
        bank_weights = gl.where(rows == row, _weight_at(weights, row), bank_weights)
    return gl.sum(bank_weights[:, None] * values, 0) + _weight_at(weights, 4) * current


@gluon.jit
def _exact_output_square(mixed, WIDTH: gl.constexpr, PACK: gl.constexpr):
    values = (mixed * mixed).reshape((WIDTH // (256 * PACK), 4, 64, PACK))
    return gl.sum(gl.sum(values, 0), 2)


@gluon.jit
def _exact_four_kernel(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps):
    token = gl.program_id(0)
    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [4, 1, 1], [0, 1, 2])
    square = gl.full((1024, 1, 1), 0, gl.float32, dot_layout)
    native_current: gl.constexpr = WRITE and (not ADD)
    v0, c0, w0, h0 = _exact_tile_load(Prefix, Addend, Bank, ScoreWeight, token, SP, SA, SB0, SB1, ADD, WRITE, 0, 4096, 8)
    cj0 = _exact_current(c0, w0, 4096, 8)
    dot0 = _exact_dot(v0, w0, 4096, 8)
    square = _exact_square(v0, square, 4096, 8)
    v1, c1, w1, h1 = _exact_tile_load(Prefix, Addend, Bank, ScoreWeight, token, SP, SA, SB0, SB1, ADD, WRITE, 4096, 2048, 8)
    cj1 = _exact_current(c1, w1, 2048, 8)
    cj = cj0 + gl.convert_layout(cj1, cj0.type.layout, assert_trivial=True)
    dot1 = _exact_dot(v1, w1, 2048, 8)
    dot = dot0 + gl.convert_layout(dot1, dot0.type.layout, assert_trivial=True)
    square = _exact_square(v1, square, 2048, 8)
    v2, c2, w2, h2 = _exact_tile_load(Prefix, Addend, Bank, ScoreWeight, token, SP, SA, SB0, SB1, ADD, WRITE, 6144, 1024, 4)
    cj2 = _exact_current(c2, w2, 1024, 4)
    cj = cj + gl.convert_layout(cj2, cj.type.layout, assert_trivial=True)
    dot2 = _exact_dot(v2, w2, 1024, 4)
    dot = dot + gl.convert_layout(dot2, dot.type.layout, assert_trivial=True)
    square = _exact_square(v2, square, 1024, 4)
    gl.store(Current + token * 7168 + h0, c0)
    gl.store(Current + token * 7168 + h1, c1)
    gl.store(Current + token * 7168 + h2, c2)
    ow0 = gl.load(OutputWeight + h0)
    ow1 = gl.load(OutputWeight + h1)
    ow2 = gl.load(OutputWeight + h2)
    square = gl.convert_layout(square.reshape((4, 4, 64)), dot.type.layout, assert_trivial=True)
    joint = gl.join(dot, square)
    for stage in gl.static_range(2):
        joint = gl.convert_layout(joint, _butterfly_layout(stage, 4, 4))
        joint = gl.sum(joint.reshape((4, 4, 2, 32 >> stage, 2)), 2)
    joint = gl.sum(joint, 2)
    cj = gl.convert_layout(cj, _current_group_layout(4))
    cj = gl.sum(cj.reshape((4, 2, 32, 2)), 1)
    cj = gl.sum(cj, 1)
    bank_storage = gl.allocate_shared_memory(gl.float32, (4, 4, 2), _score_shared_layout(4, 4), joint)
    current_storage = gl.allocate_shared_memory(gl.float32, (4, 2), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), cj)
    exchange: gl.constexpr = _exchange_layout(4, 4)
    bank_partials = bank_storage.load(exchange)
    current_partials = current_storage.load(gl.SliceLayout(0, exchange))
    bank_dot, bank_square = gl.split(gl.sum(bank_partials, 1))
    current_dot, current_square = gl.split(gl.sum(current_partials, 0))
    inv_width = gl.div_rn(1.0, 7168.0)
    bank_score = bank_dot * gl.rsqrt(bank_square * inv_width + score_eps)
    current_score = current_dot * gl.rsqrt(current_square * inv_width + score_eps)
    current_scores = gl.full((4,), current_score, gl.float32, bank_score.type.layout)
    scores = gl.join(bank_score, current_scores).permute((1, 0)).reshape((8,))
    rows = gl.arange(0, 8, layout=scores.type.layout)
    weights = _padded_softmax(gl.where(rows <= 4, scores, -float('inf')))
    m0 = _exact_mix_m32_64(v0, c0, weights)
    m1 = _exact_mix_m32_64(v1, c1, weights)
    m2 = _exact_mix_m32_64(v2, c2, weights)
    ss0 = _exact_output_square(m0, 4096, 8)
    ss1 = _exact_output_square(m1, 2048, 8)
    ss2 = _exact_output_square(m2, 1024, 4)
    ss = ss0 + gl.convert_layout(ss1, ss0.type.layout, assert_trivial=True)
    ss = ss + gl.convert_layout(ss2, ss0.type.layout, assert_trivial=True)
    scale = gl.rsqrt(gl.sum(gl.sum(ss, 1), 0) * inv_width + output_eps)
    m0 = m0 * scale * ow0.to(gl.float32)
    m1 = m1 * scale * ow1.to(gl.float32)
    m2 = m2 * scale * ow2.to(gl.float32)
    bank_storage._keep_alive()
    current_storage._keep_alive()
    gl.store(Out + token * 7168 + h0, m0)
    gl.store(Out + token * 7168 + h1, m1)
    gl.store(Out + token * 7168 + h2, m2)


def attention_residual_norm_m32_64_banks4_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    _exact_four_kernel[m,](prefix, addend, bank, score_weight, output_weight, current, out, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, num_warps=4, waves_per_eu=2)
    return (out, current, bank)


@gluon.constexpr_function
def _exchange_layout_m32(rows, warps, row_lanes, wave_first):
    row_bits = row_lanes.bit_length() - 1
    wave_bits = warps.bit_length() - 1
    registers = [[0, 0, 1]] + [[1 << bit, 0, 0] for bit in range(row_bits, rows.bit_length() - 1)]
    row_axes = [[1 << bit, 0, 0] for bit in range(row_bits)]
    wave_axes = [[0, 1 << bit, 0] for bit in range(wave_bits)]
    lanes = wave_axes + row_axes if wave_first else row_axes + wave_axes
    lanes += [[0, 0, 0]] * (6 - row_bits - wave_bits)
    return gl.DistributedLinearLayout(registers, lanes, [[0, 0, 0]] * wave_bits, [], [rows, warps, 2])


@gluon.jit
def _native_bank_square(values, BH: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr):
    local_width: gl.constexpr = BH // (NW * 64)
    batch_count: gl.constexpr = BV * NW * 64
    acc_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [NW, 1, 1], [0, 1, 2])
    local = values.to(gl.bfloat16).reshape((BV, BH // (NW * 512), NW, 64, 8))
    local = local.permute((0, 2, 3, 1, 4)).reshape((batch_count, 1, local_width))
    lhs = gl.convert_layout(local, gl.DotOperandLayout(0, acc_layout, 0))
    rhs = gl.convert_layout(local.permute((0, 2, 1)), gl.DotOperandLayout(1, acc_layout, 0))
    acc = gl.full((batch_count, 1, 1), 0, gl.float32, acc_layout)
    return gl.dot_fma(lhs, rhs, acc).reshape((BV, NW, 64))


@gluon.jit
def _exchange_score_partials(joint, inv_width, eps, ROWS: gl.constexpr, NW: gl.constexpr, ROW_LANES: gl.constexpr, WAVE_FIRST: gl.constexpr=False, SPLIT_SUMS: gl.constexpr=False):
    storage = gl.allocate_shared_memory(gl.float32, (ROWS, NW, 2), _score_shared_layout(ROWS, NW), joint)
    joint = storage.load(_exchange_layout_m32(ROWS, NW, ROW_LANES, WAVE_FIRST))
    dot, square = gl.split(gl.sum(joint, 1))
    scores = dot * gl.rsqrt(square * inv_width + eps)
    return (scores, storage)


@gluon.jit
def _compact_one_row_scores(values, current, cw, inv_width, eps, BH: gl.constexpr, NW: gl.constexpr, WAVE_FIRST: gl.constexpr, NATIVE_BANK: gl.constexpr=False):
    shape: gl.constexpr = (BH // (NW * 512), NW, 64, 8)
    bank = gl.sum(values, 0)
    bank_dot = gl.sum(gl.sum((bank * cw).reshape(shape), 0), 2)
    bank_square = gl.sum(_native_bank_square(values, BH, NW, 1), 0)
    bank_square = gl.convert_layout(bank_square, bank_dot.type.layout, assert_trivial=True)
    current_dot = gl.sum(gl.sum((current * cw).reshape(shape), 0), 2)
    current_square = gl.sum(gl.sum((current * current).reshape(shape), 0), 2)
    dot = gl.join(bank_dot, current_dot).permute((2, 0, 1))
    square = gl.join(bank_square, current_square).permute((2, 0, 1))
    joint = gl.join(dot, square)
    joint = gl.convert_layout(joint, _butterfly_layout(0, NW, 2))
    joint = gl.sum(joint.reshape((2, NW, 2, 32, 2)), 2)
    grouped: gl.constexpr = _one_group_layout(NW)
    joint = gl.convert_layout(joint, grouped)
    joint = gl.sum(joint.reshape((2, NW, 2, 16, 2)), 2)
    joint = gl.sum(joint, 2)
    return _exchange_score_partials(joint, inv_width, eps, 2, NW, 2, WAVE_FIRST)


@gluon.jit
def _attention_residual_kernel_m32(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BH: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr, COMPACT_BANK: gl.constexpr, REPLICAS: gl.constexpr=1):
    token = gl.program_id(1)
    shard = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([2 if NV == 8 and REPLICAS > 1 else 1, 8], [1, 64], [1, NW], [1, 0])
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    channels = gl.arange(0, BH, layout=gl.SliceLayout(0, layout))
    valid_channel = channels < H
    store_channel = valid_channel
    SHARD_WIDTH: gl.constexpr = (H + REPLICAS - 1) // REPLICAS if NV == 4 and (not ADD) and WRITE else BH // REPLICAS
    store_channel = valid_channel & (channels >= shard * SHARD_WIDTH) & (channels < (shard + 1) * SHARD_WIDTH)
    current = gl.load(Prefix + token * SP + channels, valid_channel, 0).to(gl.float32)
    state_value = current
    LATE_BANK_STORE: gl.constexpr = NV == 4 and REPLICAS > 1
    values = gl.load(Bank + token * SB0 + rows[:, None] * SB1 + channels[None, :], (rows[:, None] < NV) & valid_channel[None, :], 0).to(gl.float32)
    redirected_channels = gl.where(valid_channel, channels, channels - H)
    cw = gl.load(ScoreWeight + redirected_channels)
    cw = gl.where(valid_channel, cw, 0.0)
    inv_width = gl.div_rn(1.0, 1.0 * H)
    scores, score_storage = _compact_one_row_scores(values, current, cw, inv_width, score_eps, BH, NW, True, NATIVE_BANK=REPLICAS > 1 and H == 7168)
    all_rows = gl.arange(0, BV * 2, layout=scores.type.layout)
    scores = gl.where(all_rows <= NV, scores, -float('inf'))
    ow = gl.load(OutputWeight + channels, valid_channel, 0)
    weights = _padded_softmax(scores)
    mixed = _weight_at(weights, 0) * gl.sum(values, 0) + _weight_at(weights, 1) * current
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inv_width + output_eps)
    mixed = mixed * scale * ow.to(gl.float32)
    score_storage._keep_alive()
    gl.store(Out + token * H + channels, mixed, store_channel)


def attention_residual_norm_m32_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    num_warps = 4
    compact_bank = valid_rows in (1, 4, 8)
    bank_tile = triton.next_power_of_2(valid_rows)
    replicas = 1
    replicas = 4
    grid = (replicas, m)
    _attention_residual_kernel_m32[grid](prefix, addend, bank, score_weight, output_weight, current, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, apply_output_norm, score_eps, output_eps, triton.next_power_of_2(h), bank_tile, num_warps, compact_bank, REPLICAS=replicas, num_warps=num_warps, waves_per_eu=1)
    return (out, current, bank)


@gluon.constexpr_function
def _stable_stage_layout(rows, warps, channels, row_lanes, order):

    order = [int(bit) for bit in order]
    stage_count = row_lanes.bit_length()
    remaining = [i for i in range(6) if i not in order[:stage_count]]
    lane_bases = []
    for i in range(6):
        if i in remaining:
            lane_bases.append([0, 0, 0, 1 << remaining.index(i)])
        else:
            lane_bases.append([rows >> 1 + order.index(i), 0, 0, 0])
    register_rows = rows // (2 * row_lanes)
    return gl.DistributedLinearLayout(reg_bases=[[0, 0, 1, 0]] + [[1 << i, 0, 0, 0] for i in range(register_rows.bit_length() - 1)], lane_bases=lane_bases, warp_bases=[[0, 1 << i, 0, 0] for i in range(warps.bit_length() - 1)], block_bases=[], shape=[rows, warps, 2, channels // 2])


@gluon.constexpr_function
def _relative_channel_bit(order, row_lanes):
    order = [int(bit) for bit in order]
    stage = row_lanes.bit_length() - 1
    physical_bit = order[stage]
    return physical_bit - sum((bit < physical_bit for bit in order[:stage]))


@gluon.jit
def _contract_channels(values, ROWS: gl.constexpr, WARPS: gl.constexpr, CHANNELS: gl.constexpr, ROW_LANES: gl.constexpr, STAGES: gl.constexpr, STABLE: gl.constexpr=False, ORDER: gl.constexpr=(5, 4, 3, 2, 1, 0)):

    if STAGES == 0:
        return gl.sum(values, 2)
    else:
        stage_layout: gl.constexpr = _stable_stage_layout(ROWS, WARPS, CHANNELS, ROW_LANES, ORDER)
        channel_bit: gl.constexpr = _relative_channel_bit(ORDER, ROW_LANES)
        low: gl.constexpr = 1 << channel_bit
        high: gl.constexpr = CHANNELS // (2 * low)
        pairs = gl.reshape(values, (ROWS, WARPS, high, 2, low))
        pairs = gl.reshape(gl.permute(pairs, (0, 1, 3, 2, 4)), (ROWS, WARPS, 2, CHANNELS // 2))
        pairs = gl.convert_layout(pairs, stage_layout)
        values = gl.sum(pairs, 2)
        return _contract_channels(values, ROWS, WARPS, CHANNELS // 2, ROW_LANES * 2, STAGES - 1, STABLE, ORDER)


@gluon.jit
def _bf16_lane_square(value, ROWS: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, SPT: gl.constexpr):

    REPEATS: gl.constexpr = BLOCK // (WARPS * 64 * SPT)
    BATCH: gl.constexpr = ROWS * WARPS * 64
    packed = gl.reshape(value.to(gl.bfloat16), (ROWS, REPEATS, WARPS * 64, SPT))
    packed = gl.reshape(gl.permute(packed, (0, 2, 1, 3)), (BATCH, 1, REPEATS * SPT))
    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [WARPS, 1, 1], [0, 1, 2])
    a = gl.convert_layout(packed, gl.DotOperandLayout(0, dot_layout, 0))
    b = gl.convert_layout(gl.permute(packed, (0, 2, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    acc = gl.full((BATCH, 1, 1), 0, gl.float32, dot_layout)
    square = gl.dot_fma(a, b, acc)
    return gl.reshape(square, (ROWS, WARPS, 64))


@gluon.jit
def _wave_statistics(value, weight, ROWS: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, SPT: gl.constexpr, STAGES: gl.constexpr=0, STABLE: gl.constexpr=False, ORDER: gl.constexpr=(5, 4, 3, 2, 1, 0), DOT_SQUARE: gl.constexpr=False):

    WAVE: gl.constexpr = 64 * SPT
    REPEATS: gl.constexpr = BLOCK // (WARPS * WAVE)
    dot = gl.sum(gl.reshape(value * weight, (ROWS, REPEATS, WARPS * WAVE)), 1)
    square = _bf16_lane_square(value, ROWS, BLOCK, WARPS, SPT)
    dot = gl.sum(gl.reshape(dot, (ROWS, WARPS, WAVE)), 2)
    square = gl.sum(square, 2)
    partial_layout: gl.constexpr = gl.SliceLayout(2, gl.BlockedLayout([1, 1, SPT], [1, 1, 64], [1, WARPS, 1], [2, 1, 0]))
    return (gl.convert_layout(dot, partial_layout, assert_trivial=True), gl.convert_layout(square, partial_layout, assert_trivial=True))


@gluon.jit
def _register_row(values, INDEX: gl.constexpr, ROWS: gl.constexpr, BLOCK: gl.constexpr):

    if ROWS == 1:
        return gl.sum(values, 0)
    else:
        halves = gl.reshape(values, (2, ROWS // 2, BLOCK))
        low, high = gl.split(gl.permute(halves, (1, 2, 0)))
        if INDEX < ROWS // 2:
            return _register_row(low, INDEX, ROWS // 2, BLOCK)
        else:
            return _register_row(high, INDEX - ROWS // 2, ROWS // 2, BLOCK)


@gluon.jit
def _packed_bank_statistics(bank, weight, ROWS: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, SPT: gl.constexpr, STAGES: gl.constexpr, ORDER: gl.constexpr=(5, 4, 3, 2, 1, 0)):

    WAVE: gl.constexpr = 64 * SPT
    REPEATS: gl.constexpr = BLOCK // (WARPS * WAVE)
    dot = gl.sum(gl.reshape(bank * weight[None, :], (ROWS, REPEATS, WARPS * WAVE)), 1)
    square = _bf16_lane_square(bank, ROWS, BLOCK, WARPS, SPT)
    dot = gl.sum(gl.reshape(dot, (ROWS, WARPS, 64, SPT)), 3)
    square = gl.convert_layout(square, dot.type.layout, assert_trivial=True)
    stats = gl.reshape(gl.permute(gl.join(dot, square), (0, 3, 1, 2)), (2 * ROWS, WARPS, 64))
    stats = _contract_channels(stats, 2 * ROWS, WARPS, 64, 1, STAGES, True, ORDER)
    return gl.permute(gl.reshape(stats, (ROWS, 2, WARPS)), (0, 2, 1))


@gluon.jit
def _row_probabilities(bank, current, weight, score_eps, H: gl.constexpr, VALID: gl.constexpr, BLOCK: gl.constexpr, ROWS: gl.constexpr, WARPS: gl.constexpr, SPT: gl.constexpr, MERGE_ROWS: gl.constexpr, STAGES: gl.constexpr, STABLE: gl.constexpr=False, JOINT: gl.constexpr=False, PAD_JOIN: gl.constexpr=False, PACK_STATS: gl.constexpr=False, ORDER: gl.constexpr=(5, 4, 3, 2, 1, 0), SOFTMAX: gl.constexpr=0):

    SCORE_ROWS: gl.constexpr = ROWS * 2 if VALID == ROWS else ROWS
    gl.static_assert(VALID == ROWS)
    bank_stats = _packed_bank_statistics(bank, weight, ROWS, BLOCK, WARPS, SPT, STAGES, ORDER)
    current_dot, current_square = _wave_statistics(current[None, :], weight[None, :], 1, BLOCK, WARPS, SPT, DOT_SQUARE=VALID == 8)
    current_stats = gl.join(gl.sum(current_dot, 0), gl.sum(current_square, 0))
    packed = gl.reshape(gl.permute(gl.join(bank_stats, bank_stats), (3, 0, 1, 2)), (SCORE_ROWS, WARPS, 2))
    current_stats = gl.convert_layout(current_stats, gl.SliceLayout(0, packed.type.layout))
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, packed.type.layout))
    partial_row = gl.arange(0, SCORE_ROWS, layout=row_layout)
    packed = gl.where(partial_row[:, None, None] < VALID, packed, current_stats[None, :, :])
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [MERGE_ROWS, 64 // MERGE_ROWS, 1], [1, 1, WARPS], [0, 1, 2])
    packed = gl.convert_layout(packed, merge_layout)
    dot, square = gl.split(gl.sum(packed, 1))
    score_row = gl.arange(0, SCORE_ROWS, layout=dot.type.layout)
    all_scores = dot * gl.rsqrt(square * gl.div_rn(1.0, H * 1.0) + score_eps)
    all_scores = gl.where(score_row <= VALID, all_scores, -float('inf'))
    centered = all_scores - gl.max(all_scores, 0)
    if SOFTMAX:
        exps = gl.exp2(centered * 1.4426950408889634)
    else:
        exps = gl.exp(centered)
    return gl.div_rn(exps, gl.sum(exps, 0))


@gluon.jit
def _mix_rows(bank, current, probabilities, VALID: gl.constexpr, ROWS: gl.constexpr, BLOCK: gl.constexpr):

    mixed = gl.full((BLOCK,), 0, gl.float32, current.type.layout)
    for row in gl.static_range(VALID + 1):
        index = gl.full((1,), row, gl.int32, probabilities.type.layout)
        probability = gl.sum(gl.gather(probabilities, index, 0), 0)
        if row < VALID:
            value = _register_row(bank, row, ROWS, BLOCK)
            value = gl.convert_layout(value, current.type.layout, assert_trivial=True)
        else:
            value = current
        mixed += value * probability
    return mixed


@gluon.jit
def _compact_rows(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, H: gl.constexpr, PREFIX_STRIDE: gl.constexpr, ADDEND_STRIDE: gl.constexpr, BANK_TOKEN_STRIDE: gl.constexpr, BANK_ROW_STRIDE: gl.constexpr, VALID: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, ROWS: gl.constexpr, WARPS: gl.constexpr, SPT: gl.constexpr, REPLICAS: gl.constexpr, MERGE_ROWS: gl.constexpr, STAGES: gl.constexpr, PRIVATE_NORM: gl.constexpr=False, PREFETCH: gl.constexpr=False, STABLE: gl.constexpr=False, JOINT: gl.constexpr=False, PAD_JOIN: gl.constexpr=False, PACK_STATS: gl.constexpr=False, ADJACENT: gl.constexpr=False, EARLY_WEIGHT: gl.constexpr=False, DELAY_CURRENT: gl.constexpr=False, CONTRACTION_ORDER: gl.constexpr=(5, 4, 3, 2, 1, 0), RMS_GROUPS: gl.constexpr=1, RMS_READ: gl.constexpr=1, WEIGHT_IN_EXCHANGE: gl.constexpr=False, SOFTMAX: gl.constexpr=0):
    norm_scratch = gl.allocate_shared_memory(gl.float32, (WARPS * RMS_GROUPS,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
    token = gl.program_id(0)
    replica = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, SPT], [1, 64], [1, WARPS], [1, 0])
    channel = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
    owns_output = True
    owns_current = True
    owns_output = channel // (BLOCK // REPLICAS) == replica
    owns_current = replica == 0
    row = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    rounded_current = gl.load(Prefix + token * PREFIX_STRIDE + channel, channel < H, 0)
    current = rounded_current.to(gl.float32)
    addend = gl.load(Addend + token * ADDEND_STRIDE + channel, channel < H, 0).to(gl.float32)
    rounded_current = (current + addend).to(Current.dtype.element_ty)
    if owns_current and (not DELAY_CURRENT):
        gl.store(Current + token * H + channel, rounded_current, channel < H)
    current = rounded_current.to(gl.float32)
    bank = gl.load(Bank + token * BANK_TOKEN_STRIDE + row[:, None] * BANK_ROW_STRIDE + channel[None, :], (row[:, None] < VALID) & (channel[None, :] < H), 0).to(gl.float32)
    weight = gl.load(ScoreWeight + channel, channel < H, 0)
    probabilities = _row_probabilities(bank, current, weight, score_eps, H, VALID, BLOCK, ROWS, WARPS, SPT, MERGE_ROWS, STAGES, STABLE, JOINT, PAD_JOIN, PACK_STATS, CONTRACTION_ORDER, SOFTMAX)
    mixed = _mix_rows(bank, current, probabilities, VALID, ROWS, BLOCK)
    if owns_current:
        gl.store(Current + token * H + channel, rounded_current, channel < H)
    WAVE: gl.constexpr = 64 * SPT
    values = gl.reshape(mixed * mixed, (BLOCK // (WARPS * WAVE), WARPS * RMS_GROUPS, WAVE // RMS_GROUPS))
    partial = gl.sum(gl.sum(values, 0), 1)
    norm_scratch.store(partial)
    output_weight_raw = gl.load(OutputWeight + channel, channel < H, 0)
    totals = norm_scratch.load(gl.BlockedLayout([RMS_READ], [64], [WARPS], [0]))
    square_sum = gl.sum(totals, 0)
    output_weight = output_weight_raw.to(gl.float32)
    scale = gl.rsqrt(square_sum * gl.div_rn(1.0, H * 1.0) + output_eps)
    mixed = mixed * scale * output_weight
    gl.store(Out + token * H + channel, mixed, (channel < H) & owns_output)


def _launch_config(m, h, valid_rows, has_addend, write_bank):
    """Tuned layout for the selected (32, 7168, 8, mode=5) contract."""
    return {
        'BLOCK': 8192,
        'ROWS': 8,
        'WARPS': 8,
        'SPT': 8,
        'SOFTMAX': 2,
        'CONTRACTION_ORDER': (5, 4, 3, 2, 1, 0),
        'RMS_GROUPS': 4,
        'RMS_READ': 2,
        'WEIGHT_IN_EXCHANGE': True,
        'REPLICAS': 2,
        'MERGE_ROWS': 64,
        'STAGES': 3,
        'PRIVATE_NORM': True,
        'PREFETCH': False,
        'STABLE': True,
        'JOINT': False,
        'PAD_JOIN': True,
        'ADJACENT': False,
        'EARLY_WEIGHT': False,
        'waves_per_eu': 0,
        'DELAY_CURRENT': True,
        'PACK_STATS': True,
    }


def attention_residual_norm_m32_banks8_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):

    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    config = _launch_config(m, h, valid_rows, has_addend, write_bank)
    replicas = config['REPLICAS']
    grid = (m * replicas,) if config['ADJACENT'] else (m, replicas)
    _compact_rows[grid](prefix, addend, bank, score_weight, output_weight, current, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, apply_output_norm, score_eps, output_eps, **config, num_warps=config['WARPS'])
    return (out, current, bank)


@gluon.jit
def _tiled_scores(Prefix, Addend, Current, Bank, ScoreWeight, Stats, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, TILE: gl.constexpr, TILES: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, ORDER: gl.constexpr, M: gl.constexpr):
    if ORDER == 0:
        token, row, tile = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    else:
        token = gl.program_id(0) % M
        tile = gl.program_id(0) // M
        row = gl.program_id(1)
    h = tile * TILE + gl.arange(0, TILE, layout=gl.BlockedLayout([VEC], [64], [WAVES], [0]))
    pair_layout: gl.constexpr = gl.BlockedLayout([2], [64], [WAVES], [0])
    pair = gl.arange(0, 2, layout=pair_layout)
    if row < NV:
        bank_value = gl.load(Bank + token * SB0 + row * SB1 + h, h < H, 0).to(gl.float32)
        bank_weight = buffer_load(ScoreWeight, h, h < H, 0)
        bank_stat = gl.sum(gl.join(bank_value * bank_weight, bank_value * bank_value), 0)
        bank_stat = gl.convert_layout(bank_stat, pair_layout)
        gl.store(Stats + ((token * TILES + tile) * 16 + row) * 2 + pair, bank_stat)
    else:
        current_value = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
        current_weight = buffer_load(ScoreWeight, h, h < H, 0)
        current_stat = gl.sum(gl.join(current_value * current_weight, current_value * current_value), 0)
        current_stat = gl.convert_layout(current_stat, pair_layout)
        gl.store(Stats + ((token * TILES + tile) * 16 + NV) * 2 + pair, current_stat)
        pad_layout: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [1, 0])
        pad_row = gl.arange(0, 16, layout=gl.SliceLayout(1, pad_layout))
        pad_stat = gl.arange(0, 2, layout=gl.SliceLayout(0, pad_layout))
        gl.store(Stats + ((token * TILES + tile) * 16 + pad_row[:, None]) * 2 + pad_stat[None, :], 0.0, pad_row[:, None] > NV)


@gluon.jit
def _tiled_mix(Current, Bank, Stats, OutputWeight, Out, H: gl.constexpr, SC: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, TILES: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, PREFETCH_THIRD: gl.constexpr):
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [WAVES], [0])
    h = gl.arange(0, BLOCK, layout=layout)
    output_weight = buffer_load(OutputWeight, h, h < H, 0)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [16, 4, 1], [WAVES, 1, 1], [2, 0, 1])
    row_tile_layout: gl.constexpr = gl.SliceLayout(2, stats_layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, row_tile_layout)
    tile_layout: gl.constexpr = gl.SliceLayout(0, row_tile_layout)
    rows = gl.arange(0, 16, layout=row_layout)
    tiles = gl.arange(0, TILES, layout=tile_layout)
    stat_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, stats_layout))
    stat = gl.arange(0, 2, layout=stat_layout)
    pairs = gl.load(Stats + ((token * TILES + tiles[None, :, None]) * 16 + rows[:, None, None]) * 2 + stat[None, None, :])
    dot, square = gl.split(gl.sum(pairs, 1))
    score_layout: gl.constexpr = gl.BlockedLayout([1], [64], [WAVES], [0])
    dot = gl.convert_layout(dot, score_layout)
    square = gl.convert_layout(square, score_layout)
    rows = gl.arange(0, 16, layout=score_layout)
    first_row = gl.load(Bank + token * SB0 + h, h < H, 0)
    second_row = gl.load(Bank + token * SB0 + SB1 + h, h < H, 0)
    if PREFETCH_THIRD:
        third_row = gl.load(Bank + token * SB0 + 2 * SB1 + h, h < H, 0)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    scores = gl.where(rows <= NV, scores, -float('inf'))
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    mixed = gl.full((BLOCK,), 0, gl.float32, layout)
    for row in gl.static_range(NV):
        probability = gl.sum(gl.gather(probabilities, gl.full((1,), row, gl.int32, score_layout), 0), 0)
        if row == 0:
            values = first_row.to(gl.float32)
        elif row == 1:
            values = second_row.to(gl.float32)
        elif row == 2 and PREFETCH_THIRD:
            values = third_row.to(gl.float32)
        else:
            values = gl.load(Bank + token * SB0 + row * SB1 + h, h < H, 0).to(gl.float32)
        mixed += probability * values
    probability = gl.sum(gl.gather(probabilities, gl.full((1,), NV, gl.int32, score_layout), 0), 0)
    current = gl.load(Current + token * SC + h, h < H, 0).to(gl.float32)
    mixed += probability * current
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
    mixed = mixed * scale * output_weight.to(gl.float32)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(1))
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m4_16_banks1_4_8_modes4_6(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device) if has_addend else prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    if valid_rows in (1, 4) and block >= 2048:
        common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
        if valid_rows == 1:
            order = int(m >= 4)
            grid = (triton.cdiv(h, 2048), m) if order else (m, triton.cdiv(h, 2048))
            _fused_pair[grid](*common, order, num_warps=4, waves_per_eu=2)
        else:
            _fused_four[triton.cdiv(h, 2048), m](*common, num_warps=4)
    else:
        if valid_rows == 8:
            tile = 2048
            tiles = triton.next_power_of_2(triton.cdiv(h, tile))
            stats = torch.empty((m, tiles, 16, 2), device=prefix.device, dtype=torch.float32)
            order = 0 if m >= 8 else 1
            grid = (m, valid_rows + 1, tiles) if order == 0 else (m * tiles, valid_rows + 1)
            _tiled_scores[grid](prefix, addend, current, bank, score_weight, stats, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, tile, tiles, 4, 8, order, m, num_warps=4)
            mix_width = 8192 if m in (2, 4) else 4096
            mix_block, mix_waves, mix_vec = (block, 8, 8)
            _tiled_mix[m, triton.cdiv(h, mix_width)](current, bank, stats, output_weight, out, h, current.stride(0), bank.stride(0), bank.stride(1), valid_rows, apply_output_norm, score_eps, output_eps, mix_block, tiles, mix_waves, mix_vec, mix_width, m <= 4, num_warps=mix_waves, waves_per_eu=1)
        else:
            scores = torch.empty((m, valid_rows + 1), dtype=torch.float32, device=prefix.device)
            _fast_scores[m, valid_rows + 1](prefix, addend, current, bank, score_weight, scores, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, score_eps, block, 4, 8, num_warps=4)
            _fast_mix[m,](current, bank, scores, output_weight, out, h, current.stride(0), bank.stride(0), bank.stride(1), valid_rows, apply_output_norm, output_eps, block, triton.next_power_of_2(valid_rows + 1), 4, 8, num_warps=4)
    return (out, current, bank)


@gluon.jit
def _current_row(Prefix, Addend, Bank, Current, m, h, mask, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, PUBLISH: gl.constexpr=True, KEEP_BF16: gl.constexpr=False):
    value = gl.load(Prefix + m * SP + h, mask, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + m * SA + h, mask, 0).to(gl.float32)
        value = (value + addend).to(Current.dtype.element_ty)
        if PUBLISH:
            gl.store(Current + m * H + h, value, mask)
        value = value.to(gl.float32)
    if WRITE:
        if PUBLISH:
            gl.store(Bank + m * SB0 + NV * SB1 + h, value, mask)
    if KEEP_BF16:
        return value.to(Current.dtype.element_ty)
    else:
        return value


@gluon.jit
def _softmax_m4_8_16(scores):
    numerators = gl.exp2((scores - gl.max(scores, 0)) * 1.4426950408889634)
    denominator = gl.sum(numerators, 0)
    return numerators * gl.div_rn(1.0, denominator)


@gluon.jit
def _store_output(mixed, OutputWeight, Out, m, h, eps, H: gl.constexpr, NORM: gl.constexpr, owner=True):
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * gl.div_rn(1.0, H) + eps)
    weight = gl.load(OutputWeight + h, h < H, 0).to(gl.float32)
    mixed = mixed * scale * weight
    gl.store(Out + m * H + h, mixed, (h < H) & owner)


@gluon.jit
def _store_output_quarter(mixed, OutputWeight, Out, m, eps, H: gl.constexpr, NORM: gl.constexpr, BLOCK: gl.constexpr, VEC: gl.constexpr, BUFFER_WEIGHT: gl.constexpr=False):
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * gl.div_rn(1.0, H) + eps)
    quarters = gl.permute(gl.reshape(mixed, (4, BLOCK // 4)), (1, 0))
    even, odd = gl.split(gl.reshape(quarters, (BLOCK // 4, 2, 2)))
    first, third = gl.split(even)
    second, fourth = gl.split(odd)
    shard = gl.program_id(1)
    selected = gl.where(shard < 2, gl.where(shard == 0, first, second), gl.where(shard == 2, third, fourth))
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    selected = gl.convert_layout(selected, layout, assert_trivial=True)
    h = shard * (BLOCK // 4) + gl.arange(0, BLOCK // 4, layout=layout)
    if BUFFER_WEIGHT:
        weight = buffer_load(OutputWeight, h, h < H, 0).to(gl.float32)
    else:
        weight = gl.load(OutputWeight + h, h < H, 0).to(gl.float32)
    selected = selected * scale * weight
    gl.store(Out + m * H + h, selected, h < H)


@gluon.jit
def _local_feature_sum(value, BLOCK: gl.constexpr, VEC: gl.constexpr):
    shaped = gl.reshape(value, (BLOCK // (VEC * 64 * gl.num_warps()), 64 * gl.num_warps(), VEC))
    return gl.sum(gl.sum(shaped, 0), 1)


@gluon.jit
def _local_bank_sum(value, BLOCK: gl.constexpr, VEC: gl.constexpr):
    shaped = gl.reshape(value, (4, BLOCK // (VEC * 64 * gl.num_warps()), 64 * gl.num_warps(), VEC))
    return gl.sum(gl.sum(shaped, 1), 2)


@gluon.jit
def _native_feature_square(value, BLOCK: gl.constexpr, VEC: gl.constexpr, CHAIN: gl.constexpr):
    THREADS: gl.constexpr = 64 * gl.num_warps()
    LOCAL: gl.constexpr = BLOCK // THREADS
    GROUPS: gl.constexpr = LOCAL // CHAIN
    local = gl.permute(gl.reshape(value, (LOCAL // VEC, THREADS, VEC)), (1, 0, 2))
    dot_layout: gl.constexpr = gl.BlockedLayout([GROUPS, 1, 1], [64, 1, 1], [gl.num_warps(), 1, 1], [0, 1, 2])
    left = gl.convert_layout(gl.reshape(local, (THREADS * GROUPS, 1, CHAIN)), gl.DotOperandLayout(0, dot_layout, 0))
    right = gl.convert_layout(gl.reshape(local, (THREADS * GROUPS, CHAIN, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    square = gl.dot_fma(left, right, gl.full((THREADS * GROUPS, 1, 1), 0, gl.float32, dot_layout))
    return gl.sum(gl.reshape(square, (THREADS, GROUPS)), 1)


@gluon.jit
def _native_bank_square_m4_8_16(value, BLOCK: gl.constexpr, VEC: gl.constexpr, CHAIN: gl.constexpr):
    THREADS: gl.constexpr = 64 * gl.num_warps()
    LOCAL: gl.constexpr = BLOCK // THREADS
    GROUPS: gl.constexpr = LOCAL // CHAIN
    local = gl.permute(gl.reshape(value, (4, LOCAL // VEC, THREADS, VEC)), (0, 2, 1, 3))
    dot_layout: gl.constexpr = gl.BlockedLayout([GROUPS, 1, 1], [64, 1, 1], [gl.num_warps(), 1, 1], [0, 1, 2])
    left = gl.convert_layout(gl.reshape(local, (4 * THREADS * GROUPS, 1, CHAIN)), gl.DotOperandLayout(0, dot_layout, 0))
    right = gl.convert_layout(gl.reshape(local, (4 * THREADS * GROUPS, CHAIN, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    square = gl.dot_fma(left, right, gl.full((4 * THREADS * GROUPS, 1, 1), 0, gl.float32, dot_layout))
    return gl.sum(gl.reshape(square, (4, THREADS, GROUPS)), 2)


@gluon.jit
def _pair_attention(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, score_eps, output_eps, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, ROWS: gl.constexpr, BLOCK: gl.constexpr, VEC: gl.constexpr, SHARDS: gl.constexpr, CHAIN: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    m = gl.program_id(0)
    h = gl.arange(0, BLOCK, layout=layout)
    owner = h // (BLOCK // SHARDS) == gl.program_id(1)
    current_bf16 = gl.load(Prefix + m * SP + h, h < H, 0)
    current = current_bf16.to(gl.float32)
    bank_bf16 = gl.load(Bank + m * SB0 + h, h < H, 0)
    bank_value = bank_bf16.to(gl.float32)
    weight = buffer_load(ScoreWeight, h, h < H, 0)
    bank_square = _native_feature_square(bank_bf16, BLOCK, VEC, CHAIN)
    current_square = _native_feature_square(current_bf16, BLOCK, VEC, CHAIN)
    bank_dot = _local_feature_sum(bank_value * weight, BLOCK, VEC)
    current_dot = _local_feature_sum(current * weight, BLOCK, VEC)
    bank_square = gl.convert_layout(bank_square, bank_dot.type.layout, assert_trivial=True)
    current_square = gl.convert_layout(current_square, current_dot.type.layout, assert_trivial=True)
    partial = gl.join(gl.join(bank_dot, current_dot), gl.join(bank_square, current_square))
    partial = gl.reshape(gl.permute(gl.reshape(partial, (gl.num_warps(), 2, 32, 2, 2)), (0, 2, 1, 3, 4)), (64 * gl.num_warps(), 2, 2))
    partial = gl.convert_layout(partial, gl.BlockedLayout([2, 1, 2], [32, 2, 1], [gl.num_warps(), 1, 1], [0, 1, 2]))
    reduced = gl.sum(partial, 0)
    dot, square = gl.split(reduced)
    scores = dot * gl.rsqrt(square * gl.div_rn(1.0, H) + score_eps)
    score_layout: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
    scores = gl.convert_layout(scores, score_layout)
    probabilities = _softmax_m4_8_16(scores)
    rows = gl.arange(0, 2, layout=score_layout)
    bank_probability = gl.sum(gl.where(rows == 0, probabilities, 0), 0)
    current_probability = gl.sum(gl.where(rows == 1, probabilities, 0), 0)
    mixed = bank_value * bank_probability + current * current_probability
    if BLOCK >= 4 * VEC * 64 * gl.num_warps():
        _store_output_quarter(mixed, OutputWeight, Out, m, output_eps, H, NORM, BLOCK, VEC, BUFFER_WEIGHT=True)
    else:
        _store_output(mixed, OutputWeight, Out, m, h, output_eps, H, NORM, owner)


@gluon.jit
def _compact_attention_residual(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, score_eps, output_eps, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, ROWS: gl.constexpr, BLOCK: gl.constexpr, VEC: gl.constexpr, SHARDS: gl.constexpr, CHAIN: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, gl.num_warps()], [1, 0])
    m = gl.program_id(0)
    h = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
    if not ADD:
        current = _current_row(Prefix, Addend, Bank, Current, m, h, h < H, H, SP, SA, SB0, SB1, NV, ADD, WRITE, PUBLISH=False, KEEP_BF16=not ADD)
    if (ADD or WRITE) and gl.program_id(1) == SHARDS:
        if ADD:
            _current_row(Prefix, Addend, Bank, Current, m, h, h < H, H, SP, SA, SB0, SB1, NV, ADD, WRITE)
        elif WRITE:
            gl.store(Bank + m * SB0 + NV * SB1 + h, current, h < H)
    else:
        bank_rows = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        owner = h // (BLOCK // SHARDS) == gl.program_id(1)
        if ADD:
            current = _current_row(Prefix, Addend, Bank, Current, m, h, h < H, H, SP, SA, SB0, SB1, NV, ADD, WRITE, PUBLISH=False, KEEP_BF16=not ADD)
        values = gl.load(Bank + m * SB0 + bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0)
        weight = buffer_load(ScoreWeight, h, h < H, 0)
        bank_square = _native_bank_square_m4_8_16(values, BLOCK, VEC, CHAIN)
        bank_dot = _local_bank_sum(values.to(gl.float32) * weight[None, :], BLOCK, VEC)
        bank_square = gl.convert_layout(bank_square, bank_dot.type.layout, assert_trivial=True)
        bank_statistics = gl.join(bank_dot, bank_square)
        bank_statistics = gl.reshape(gl.permute(gl.reshape(bank_statistics, (4, gl.num_warps(), 2, 32, 2)), (0, 1, 3, 2, 4)), (4, 64 * gl.num_warps(), 2))
        bank_statistics = gl.convert_layout(bank_statistics, gl.BlockedLayout([1, 2, 2], [2, 32, 1], [1, gl.num_warps(), 1], [1, 0, 2]))
        bank_statistics = gl.sum(bank_statistics, 1)
        bank_dot, bank_square = gl.split(bank_statistics)
        bank_scores = bank_dot * gl.rsqrt(bank_square * gl.div_rn(1.0, H) + score_eps)
        bank_scores = gl.convert_layout(bank_scores, gl.SliceLayout(1, layout))
        current_square = _native_feature_square(current.to(gl.bfloat16), BLOCK, VEC, CHAIN)
        current_dot = _local_feature_sum(current.to(gl.float32) * weight, BLOCK, VEC)
        current_square = gl.convert_layout(current_square, current_dot.type.layout, assert_trivial=True)
        current_statistics = gl.sum(gl.join(current_dot, current_square), 0)
        current_dot, current_square = gl.split(current_statistics)
        current_score = current_dot * gl.rsqrt(current_square * gl.div_rn(1.0, H) + score_eps)
        rows = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
        scores = gl.gather(bank_scores, rows % 4, 0)
        scores = gl.where(rows < 4, scores, gl.where(rows == 4, current_score, -float('inf')))
        probabilities = _softmax_m4_8_16(scores)
        bank_probabilities = gl.gather(probabilities, bank_rows, 0)
        current_probability = gl.sum(gl.where(rows == 4, probabilities, 0), 0)
        if ADD:
            mixed = gl.sum(values.to(gl.float32) * bank_probabilities[:, None], 0) + current * current_probability
        else:
            even, odd = gl.split(gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2)))
            first, third = gl.split(even)
            second, fourth = gl.split(odd)
            group_values = (first, second, third, fourth)
            mixed = gl.full((BLOCK,), 0, gl.float32, gl.SliceLayout(0, layout))
            for row in gl.static_range(4):
                value = group_values[row].to(gl.float32)
                value = gl.convert_layout(value, gl.SliceLayout(0, layout), assert_trivial=True)
                probability = gl.sum(gl.where(bank_rows == row, bank_probabilities, 0), 0)
                mixed = mixed + probability * value
            mixed = mixed + current.to(gl.float32) * current_probability
        if ADD and SHARDS == 4 and (BLOCK >= 4 * VEC * 64 * gl.num_warps()):
            _store_output_quarter(mixed, OutputWeight, Out, m, output_eps, H, NORM, BLOCK, VEC)
        else:
            _store_output(mixed, OutputWeight, Out, m, h, output_eps, H, NORM, owner)


def attention_residual_norm_m4_8_16_banks1_4_modes4_5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device) if has_addend else prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    kernel_meta = dict(H=h, SB0=bank.stride(0), SB1=bank.stride(1), NV=valid_rows, ROWS=triton.next_power_of_2(valid_rows + 1), BLOCK=triton.next_power_of_2(h), VEC=8, num_warps=4)
    update_meta = dict(SP=prefix.stride(0), SA=addend.stride(0), ADD=has_addend, WRITE=write_bank)
    if valid_rows == 1 and (not has_addend) and (not write_bank) and (h >= 2048):
        _pair_attention[m, 4](prefix, addend, bank, score_weight, output_weight, current, out, score_eps, output_eps, NORM=apply_output_norm, SHARDS=4, CHAIN=min(32, kernel_meta['BLOCK'] // 256), **update_meta, **kernel_meta)
    else:
        _compact_attention_residual[m, 4 + int(has_addend or write_bank)](prefix, addend, bank, score_weight, output_weight, current, out, score_eps, output_eps, NORM=apply_output_norm, SHARDS=4, CHAIN=min(32 if has_addend else 16, kernel_meta['BLOCK'] // 256), **update_meta, **kernel_meta, waves_per_eu=1)
    return (out, current, bank)


@gluon.jit
def _fused_eight_m4(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr):
    groups: gl.constexpr = WAVES * 8
    token = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, WAVES], [1, 0])
    feature_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    bank_rows = gl.arange(0, 8, layout=row_layout)
    owned = (h < H) & (h // STORE_WIDTH == gl.program_id(0))
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
    current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
    if gl.program_id(0) == 0:
        gl.store(Current + token * H + h, current, h < H)
    current_dot = _eight_subgroup_sums((current * score_weight)[None, :], 1, BLOCK, WAVES, VEC)
    current_square = _eight_subgroup_sums((current * current)[None, :], 1, BLOCK, WAVES, VEC)
    values = buffer_load(Bank + token * SB0, bank_rows[:, None] * SB1 + h[None, :], h[None, :] < H, 0).to(gl.float32)
    bank_dot = _eight_subgroup_sums(values * score_weight[None, :], 8, BLOCK, WAVES, VEC)
    bank_square = _eight_subgroup_sums(values * values, 8, BLOCK, WAVES, VEC)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout)
    current_square = gl.convert_layout(current_square, bank_square.type.layout)
    current_dot = current_dot + gl.full((8, groups), 0, gl.float32, bank_dot.type.layout)
    current_square = current_square + gl.full((8, groups), 0, gl.float32, bank_square.type.layout)
    dot_parts = gl.reshape(gl.permute(gl.join(bank_dot, current_dot), (2, 0, 1)), (16, groups))
    square_parts = gl.reshape(gl.permute(gl.join(bank_square, current_square), (2, 0, 1)), (16, groups))
    row_lanes: gl.constexpr = 4
    merge_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [row_lanes, 32 // row_lanes, 2], [WAVES, 1, 1], [0, 2, 1])
    merged = gl.convert_layout(gl.join(dot_parts, square_parts), merge_layout)
    totals = gl.sum(merged, 1)
    totals = gl.convert_layout(totals, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1]))
    dot, square = gl.split(totals)
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    rows = gl.arange(0, 16, layout=gl.SliceLayout(1, gl.BlockedLayout([1, 2], [64, 1], [WAVES, 1], [0, 1])))
    scores = gl.where(rows <= 8, scores, -float('inf'))
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, row_layout)
    bank_probabilities = gl.gather(probabilities, bank_rows, 0)
    current_probability = gl.sum(gl.gather(probabilities, gl.full((1,), 8, gl.int32, row_layout), 0), 0)
    cubes = gl.reshape(gl.permute(values, (1, 0)), (BLOCK, 2, 2, 2))
    even, odd = gl.split(cubes)
    e0, e1 = gl.split(even)
    o0, o1 = gl.split(odd)
    v0, v4 = gl.split(e0)
    v2, v6 = gl.split(e1)
    v1, v5 = gl.split(o0)
    v3, v7 = gl.split(o1)
    v0 = gl.convert_layout(v0, feature_layout, assert_trivial=True)
    w0 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 0, gl.int32, row_layout), 0), 0)
    v1 = gl.convert_layout(v1, feature_layout, assert_trivial=True)
    w1 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 1, gl.int32, row_layout), 0), 0)
    v2 = gl.convert_layout(v2, feature_layout, assert_trivial=True)
    w2 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 2, gl.int32, row_layout), 0), 0)
    v3 = gl.convert_layout(v3, feature_layout, assert_trivial=True)
    w3 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 3, gl.int32, row_layout), 0), 0)
    v4 = gl.convert_layout(v4, feature_layout, assert_trivial=True)
    w4 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 4, gl.int32, row_layout), 0), 0)
    v5 = gl.convert_layout(v5, feature_layout, assert_trivial=True)
    w5 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 5, gl.int32, row_layout), 0), 0)
    v6 = gl.convert_layout(v6, feature_layout, assert_trivial=True)
    w6 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 6, gl.int32, row_layout), 0), 0)
    v7 = gl.convert_layout(v7, feature_layout, assert_trivial=True)
    w7 = gl.sum(gl.gather(bank_probabilities, gl.full((1,), 7, gl.int32, row_layout), 0), 0)
    mixed = v0 * w0
    mixed = gl.fma(v1, w1, mixed)
    mixed = gl.fma(v2, w2, mixed)
    mixed = gl.fma(v3, w3, mixed)
    mixed = gl.fma(v4, w4, mixed)
    mixed = gl.fma(v5, w5, mixed)
    mixed = gl.fma(v6, w6, mixed)
    mixed = gl.fma(v7, w7, mixed)
    mixed = gl.fma(current, current_probability, mixed)
    output_weight = gl.load(OutputWeight + h, h < H, 0)
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
    mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m4_banks8_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    store_width = 2048
    _fused_eight_m4[triton.cdiv(h, store_width), m](prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, apply_output_norm, score_eps, output_eps, block, 8, 8, store_width, num_warps=8)
    return (out, current, bank)


@gluon.constexpr_function
def _exchange_layout_m64(rows, warps, wave_first=False):

    row_bits = rows.bit_length() - 1
    wave_bits = warps.bit_length() - 1
    row_lanes = [[1 << bit, 0, 0] for bit in range(row_bits)]
    wave_lanes = [[0, 1 << bit, 0] for bit in range(wave_bits)]
    lanes = wave_lanes + row_lanes if wave_first else row_lanes + wave_lanes
    lanes += [[0, 0, 0]] * (6 - row_bits - wave_bits)
    return gl.DistributedLinearLayout([[0, 0, 1]], lanes, [[0, 0, 0]] * wave_bits, [], [rows, warps, 2])


@gluon.jit
def _bf16_square_dot(local, NW: gl.constexpr):

    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [NW, 1, 1], [0, 1, 2])
    lhs = gl.convert_layout(local, gl.DotOperandLayout(0, dot_layout, 0))
    rhs = gl.convert_layout(local.permute((0, 2, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    acc = gl.full((local.shape[0], 1, 1), 0, gl.float32, dot_layout)
    return gl.dot_fma(lhs, rhs, acc)


@gluon.jit
def _native_square_partials_m64(values, SIZE: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr, VECTOR: gl.constexpr=8):
    shape: gl.constexpr = (BV, SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    local = values.to(gl.bfloat16).reshape(shape).permute((0, 2, 3, 1, 4))
    local = local.reshape((BV * NW * 64, 1, SIZE // (NW * 64)))
    return _bf16_square_dot(local, NW).reshape((BV, NW, 64))


@gluon.jit
def _finish_one_row_scores(bank_dot, bank_square, current_dot, current_square, inv_width, eps, NW: gl.constexpr):

    dot = gl.join(bank_dot, current_dot).permute((2, 0, 1))
    square = gl.join(bank_square, current_square).permute((2, 0, 1))
    joint = gl.join(dot, square)
    joint = gl.convert_layout(joint, _butterfly_layout(0, NW, 2))
    joint = gl.sum(joint.reshape((2, NW, 2, 32, 2)), 2)
    joint = gl.convert_layout(joint, _one_group_layout(NW))
    joint = gl.sum(joint.reshape((2, NW, 2, 16, 2)), 2)
    joint = gl.sum(joint, 2)
    storage = gl.allocate_shared_memory(gl.float32, (2, NW, 2), _score_shared_layout(2, NW), joint)
    joint = storage.load(_exchange_layout_m64(2, NW))
    dot, square = gl.split(gl.sum(joint, 1))
    return (dot * gl.rsqrt(square * inv_width + eps), storage)


@gluon.jit
def _load_exact_chunk(Prefix, Addend, Bank, CW, Current, token, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, SIZE: gl.constexpr, OFFSET: gl.constexpr, VECTOR: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [1, 64], [1, NW], [1, 0])
    channels = OFFSET + gl.arange(0, SIZE, layout=gl.SliceLayout(0, layout))
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    current = gl.load(Prefix + token * SP + channels).to(gl.float32)
    values = gl.load(Bank + token * SB0 + rows[:, None] * SB1 + channels[None, :]).to(gl.float32)
    store_channel = (channels >= 0) & (channels < 7168)
    cw = _load_output_weight(CW, channels, BV == 4 and ADD)
    return (values, current, cw)


@gluon.jit
def _exact_statistics(values, current, cw, SIZE: gl.constexpr, VECTOR: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr, NATIVE_CURRENT: gl.constexpr):
    bank_shape: gl.constexpr = (BV, SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    cur_shape: gl.constexpr = (SIZE // (NW * 64 * VECTOR), NW, 64, VECTOR)
    dot = gl.sum(gl.sum((values * cw[None, :]).reshape(bank_shape), 1), 3)
    square = _native_square_partials_m64(values, SIZE, NW, BV, VECTOR)
    square = gl.convert_layout(square, dot.type.layout, assert_trivial=True)
    cur_dot = gl.sum(gl.sum((current * cw).reshape(cur_shape), 0), 2)
    cur_local = current.to(gl.bfloat16).reshape(cur_shape).permute((1, 2, 0, 3))
    cur_local = cur_local.reshape((NW * 64, 1, SIZE // (NW * 64)))
    cur_square = _bf16_square_dot(cur_local, NW).reshape((NW, 64))
    cur_square = gl.convert_layout(cur_square, cur_dot.type.layout, assert_trivial=True)
    return (gl.join(dot, square), gl.join(cur_dot, cur_square))


@gluon.jit
def _exact_store_m64(Out, ow, mixed, scale, token, SIZE: gl.constexpr, OFFSET: gl.constexpr, NORM: gl.constexpr):
    channels = OFFSET + gl.arange(0, SIZE, layout=mixed.type.layout)
    mixed = mixed * scale * ow.to(gl.float32)
    gl.store(Out + token * 7168 + channels, mixed, (channels >= 0) & (channels < 7168))


@gluon.jit
def _add_exact_partials(first, middle, tail):

    return first + gl.convert_layout(middle, first.type.layout, assert_trivial=True) + gl.convert_layout(tail, first.type.layout, assert_trivial=True)


@gluon.jit
def _exact_width_kernel(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BV: gl.constexpr, NW: gl.constexpr):
    token = gl.program_id(0)
    token = token.to(gl.uint32)
    tiles: gl.constexpr = ((4096, 0, 8), (2048, 4096, 8), (1024, 6144, 4))
    chunks = ()
    for i in gl.static_range(3):
        chunks += (_load_exact_chunk(Prefix, Addend, Bank, ScoreWeight, Current, token, SP, SA, SB0, SB1, ADD, WRITE, tiles[i][0], tiles[i][1], tiles[i][2], BV, NW),)
    statistics = ()
    for i in gl.static_range(3):
        values, current, cw = chunks[i]
        statistics += (_exact_statistics(values, current, cw, tiles[i][0], tiles[i][2], BV, NW, not ADD),)
    joint = _add_exact_partials(statistics[0][0], statistics[1][0], statistics[2][0])
    current_joint = _add_exact_partials(statistics[0][1], statistics[1][1], statistics[2][1])
    inv_width = gl.div_rn(1.0, 7168.0)
    bank_dot, bank_square = gl.split(joint)
    current_dot, current_square = gl.split(current_joint)
    bank_dot = gl.sum(bank_dot, 0)
    bank_square = gl.sum(bank_square, 0)
    current_dot = gl.convert_layout(current_dot, bank_dot.type.layout, assert_trivial=True)
    current_square = gl.convert_layout(current_square, bank_square.type.layout, assert_trivial=True)
    scores, bank_storage = _finish_one_row_scores(bank_dot, bank_square, current_dot, current_square, inv_width, score_eps, NW)
    current_storage = bank_storage
    weights = _padded_softmax(scores)
    output_weights = ()
    for i in gl.static_range(3):
        channels = tiles[i][1] + gl.arange(0, tiles[i][0], layout=chunks[i][1].type.layout)
        output_weights += (_load_output_weight(OutputWeight, channels, BV == 4 and ADD),)
    mixtures = ()
    for i in gl.static_range(3):
        mixtures += (_exact_mix(chunks[i][0], chunks[i][1], weights, BV),)
    scale = 1.0
    squares = ()
    for i in gl.static_range(3):
        squares += (_exact_output_partials(mixtures[i], tiles[i][0], tiles[i][2], NW),)
    partials = _add_exact_partials(squares[0], squares[1], squares[2])
    scale = gl.rsqrt(gl.sum(gl.sum(partials, 1), 0) * inv_width + output_eps)
    bank_storage._keep_alive()
    current_storage._keep_alive()
    for i in gl.static_range(3):
        _exact_store_m64(Out, output_weights[i], mixtures[i], scale, token, tiles[i][0], tiles[i][1], NORM)


def attention_residual_norm_m64_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    use_exact_width = h == 7168 and m in (32, 64) and (valid_rows == 1 or (valid_rows == 4 and (m == 64 or not has_addend)))
    _exact_width_kernel[m,](prefix, addend, bank, score_weight, output_weight, current, out, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, valid_rows, 4, num_warps=4, waves_per_eu=2)
    return (out, current, bank)


@gluon.jit
def _packed_square_partials(values, BH: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr):
    batches: gl.constexpr = BV * NW * 64
    local_width: gl.constexpr = BH // (NW * 64)
    dot_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [NW, 1, 1], [0, 1, 2])
    local = values.to(gl.bfloat16).reshape((BV, BH // (NW * 512), NW, 64, 8))
    local = local.permute((0, 2, 3, 1, 4)).reshape((batches, local_width))
    lhs = gl.convert_layout(local.reshape((batches, 1, local_width)), gl.DotOperandLayout(0, dot_layout, 0))
    rhs = gl.convert_layout(local.reshape((batches, local_width, 1)), gl.DotOperandLayout(1, dot_layout, 0))
    sums = gl.dot_fma(lhs, rhs, gl.full((batches, 1, 1), 0, gl.float32, dot_layout))
    return sums.reshape((BV, NW, 64))


@gluon.jit
def _staged_bank_partials_m64(values, cw, BH: gl.constexpr, NW: gl.constexpr, BV: gl.constexpr):
    shape: gl.constexpr = (BV, BH // (NW * 512), NW, 64, 8)
    square = _packed_square_partials(values, BH, NW, BV)
    dot = gl.sum(gl.sum((values * cw[None, :]).reshape(shape), 1), 3)
    square = gl.convert_layout(square, dot.type.layout)
    joint = gl.join(dot, square)
    for stage in gl.static_range(2 if BV == 4 else 3):
        joint = gl.convert_layout(joint, _butterfly_layout(stage, NW, BV))
        joint = gl.sum(joint.reshape((BV, NW, 2, 32 >> stage, 2)), 2)
    return gl.sum(joint, 2)


@gluon.jit
def _combined_score_partials(joint, current_joint, inv_width, eps, BV: gl.constexpr, NW: gl.constexpr):
    storage = gl.allocate_shared_memory(gl.float32, (BV * 2, NW, 2), gl.SwizzledSharedLayout(1, 1, 1, [2, 0, 1]))
    storage.slice(0, BV).store(joint)
    current_local = gl.convert_layout(current_joint, gl.SliceLayout(0, joint.type.layout))
    padded_current, _ = gl.broadcast(current_local[None, :, :], joint)
    storage.slice(BV, BV).store(padded_current)
    exchange: gl.constexpr = gl.DistributedLinearLayout([[0, 0, 1], [0, 4, 0]], [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [], [16, 8, 2])
    partials = storage.load(exchange)
    dot, square = gl.split(gl.sum(partials, 1))
    scores = dot * gl.rsqrt(square * inv_width + eps)
    scores = gl.convert_layout(scores, gl.BlockedLayout([1], [64], [NW], [0]))
    return (scores, storage)


@gluon.jit
def _segmented_scores_m64(values, current, cw, inv_width, eps, BH: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr):
    joint = _staged_bank_partials_m64(values, cw, BH, NW, BV)
    current_joint = _current_stat_partials(current, cw, BH, NW)
    scores, storage = _combined_score_partials(joint, current_joint, inv_width, eps, BV, NW)
    return (scores, (storage, storage))


@gluon.jit
def _attention_residual_kernel_m64(Prefix, Addend, Bank, ScoreWeight, OutputWeight, Current, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, NV: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BH: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr, COMPACT_BANK: gl.constexpr, REPLICAS: gl.constexpr=1):
    replica_axis: gl.constexpr = REPLICAS > 1 and (NV == 1 or (NV == 4 and WRITE))
    token = gl.program_id(0) // REPLICAS
    shard = gl.program_id(0) % REPLICAS
    row_registers: gl.constexpr = 2 if NV == 8 and REPLICAS > 1 else 1
    layout: gl.constexpr = gl.BlockedLayout([row_registers, 8], [1, 64], [1, NW], [1, 0])
    rows = gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    channels = gl.arange(0, BH, layout=gl.SliceLayout(0, layout))
    valid_channel = channels < H
    store_channel = valid_channel
    current = gl.load(Prefix + token * SP + channels, valid_channel, 0).to(gl.float32)
    store_policy: gl.constexpr = '.cs' if NV == 8 and REPLICAS == 1 else ''
    addend = gl.load(Addend + token * SA + channels, valid_channel, 0).to(gl.float32)
    current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
    values = gl.load(Bank + token * SB0 + rows[:, None] * SB1 + channels[None, :], (rows[:, None] < NV) & valid_channel[None, :], 0).to(gl.float32)
    cw = gl.load(ScoreWeight + channels, valid_channel, 0)
    inv_width = gl.div_rn(1.0, 1.0 * H)
    scores, score_storage = _segmented_scores_m64(values, current, cw, inv_width, score_eps, BH, BV, NW)
    all_rows = gl.arange(0, BV * 2, layout=scores.type.layout)
    scores = gl.where(all_rows <= NV, scores, -float('inf'))
    gl.store(Current + token * H + channels, current, store_channel, cache_modifier=store_policy)
    weights = _padded_softmax(scores)
    mixed = gl.full((BH,), 0, gl.float32, gl.SliceLayout(0, layout))
    for row in gl.static_range(BV):
        mixed = mixed + _weight_at(weights, row) * _bank_row(values, row)
    mixed = mixed + _weight_at(weights, NV) * current
    ow = gl.load(OutputWeight + channels, valid_channel, 0)
    scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inv_width + output_eps)
    mixed = mixed * scale * ow.to(gl.float32)
    score_storage[0]._keep_alive()
    score_storage[1]._keep_alive()
    gl.store(Out + token * H + channels, mixed, store_channel, cache_modifier=store_policy)


def attention_residual_norm_m64_banks8_modes5(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    num_warps = 8
    compact_bank = valid_rows in (1, 4, 8)
    bank_tile = triton.next_power_of_2(valid_rows)
    replicas = 1
    waves_per_eu = 2
    replica_axis = replicas > 1 and (valid_rows == 1 or (valid_rows == 4 and write_bank))
    grid = (m * replicas,)
    _attention_residual_kernel_m64[grid](prefix, addend, bank, score_weight, output_weight, current, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), valid_rows, has_addend, write_bank, apply_output_norm, score_eps, output_eps, triton.next_power_of_2(h), bank_tile, num_warps, compact_bank, REPLICAS=replicas, num_warps=num_warps, waves_per_eu=waves_per_eu)
    return (out, current, bank)


@gluon.jit
def _pair_subgroup_sums_m8(values, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr):
    groups: gl.constexpr = 4 * WAVES
    group_width: gl.constexpr = 16 * VEC
    pieces = gl.reshape(values, (BLOCK // (groups * group_width), groups, group_width, 2))
    pieces = gl.permute(pieces, (1, 0, 2, 3))
    return gl.sum(gl.reshape(pieces, (groups, BLOCK // groups, 2)), 1)


@gluon.jit
def _fused_pair_m8(Prefix, Addend, Current, Bank, ScoreWeight, OutputWeight, Out, H: gl.constexpr, SP: gl.constexpr, SA: gl.constexpr, SB0: gl.constexpr, SB1: gl.constexpr, ADD: gl.constexpr, WRITE: gl.constexpr, NORM: gl.constexpr, score_eps, output_eps, BLOCK: gl.constexpr, WAVES: gl.constexpr, VEC: gl.constexpr, STORE_WIDTH: gl.constexpr, OWNER_FIRST: gl.constexpr):
    token = gl.program_id(1) if OWNER_FIRST else gl.program_id(0)
    owner = gl.program_id(0) if OWNER_FIRST else gl.program_id(1)
    pair_layout: gl.constexpr = gl.BlockedLayout([VEC, 1], [64, 1], [WAVES, 1], [0, 1])
    feature_layout: gl.constexpr = gl.SliceLayout(1, pair_layout)
    h = gl.arange(0, BLOCK, layout=feature_layout)
    owned = (h < H) & (h // STORE_WIDTH == owner)
    bank_values = gl.load(Bank + token * SB0 + h, h < H, 0).to(gl.float32)
    current = gl.load(Prefix + token * SP + h, h < H, 0).to(gl.float32)
    if ADD:
        addend = gl.load(Addend + token * SA + h, h < H, 0).to(gl.float32)
        current = (current + addend).to(Current.dtype.element_ty).to(gl.float32)
        if owner == 0:
            gl.store(Current + token * H + h, current, h < H)
    if WRITE:
        if owner == 0:
            gl.store(Bank + token * SB0 + SB1 + h, current, h < H)
    values = gl.join(bank_values, current)
    score_weight = gl.load(ScoreWeight + h, h < H, 0)
    dot_parts = _pair_subgroup_sums_m8(values * score_weight[:, None], BLOCK, WAVES, VEC)
    square_parts = _pair_subgroup_sums_m8(values * values, BLOCK, WAVES, VEC)
    if NORM:
        output_weight = gl.load(OutputWeight + h, h < H, 0)
    records = gl.convert_layout(gl.join(dot_parts, square_parts), gl.BlockedLayout([1, 1, 2], [16, 4, 1], [WAVES, 1, 1], [0, 1, 2]))
    dot, square = gl.split(gl.sum(records, 0))
    inverse_width = gl.div_rn(1.0, H * 1.0)
    scores = dot * gl.rsqrt(square * inverse_width + score_eps)
    exp_scores = gl.exp(scores - gl.max(scores, 0))
    probabilities = exp_scores * (1.0 / gl.sum(exp_scores, 0))
    probabilities = gl.convert_layout(probabilities, gl.SliceLayout(0, pair_layout))
    mixed = gl.sum(values * probabilities[None, :], 1)
    if NORM:
        scale = gl.rsqrt(gl.sum(mixed * mixed, 0) * inverse_width + output_eps)
        mixed = mixed * scale * output_weight.to(gl.float32)
    gl.store(Out + token * H + h, mixed, owned)


def attention_residual_norm_m8_banks1_modes4(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, score_weight: torch.Tensor, output_weight: torch.Tensor, *, valid_rows: int, has_addend: bool=True, write_bank: bool=False, apply_output_norm: bool=True, score_eps: float=1e-05, output_eps: float=1e-05):
    m, h = prefix.shape
    current = prefix
    out = torch.empty((m, h), dtype=prefix.dtype, device=prefix.device)
    block = triton.next_power_of_2(h)
    common = (prefix, addend, current, bank, score_weight, output_weight, out, h, prefix.stride(0), addend.stride(0), bank.stride(0), bank.stride(1), has_addend, write_bank, apply_output_norm, score_eps, output_eps, block, 4, 8, 2048)
    owner_first = m >= 4
    grid = (triton.cdiv(h, 2048), m) if owner_first else (m, triton.cdiv(h, 2048))
    _fused_pair_m8[grid](*common, owner_first, num_warps=4, waves_per_eu=2)
    return (out, current, bank)
