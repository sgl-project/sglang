"""Generated Kimi-K3 MLA key projection and cache kernels for gfx950.

Source: OpenAI-Partners/artemis-kernel-integrations PR 17,
commit 35b249f7a551278946a81b7da1d58c286c41fb8f.
"""

# ruff: noqa
# fmt: off

"""Selected mla kc cache schedules and shared helpers."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy


@gluon.constexpr_function
def _m128_physical_copy_layout(shared, shape):

    bases = shared.offset_bases
    return gl.DistributedLinearLayout(
        bases[:3] + bases[11:], bases[3:9], bases[9:11], [], shape,
    )


@gluon.jit
def _m128_prepare_keys_and_tails(
    Q, Latent, Tail, Locations, Cache, Out, Fresh, group,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LOCATION_BLOCK: gl.constexpr, COPY_BLOCK: gl.constexpr,
    GROUP: gl.constexpr, VEC: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VEC], [1, 64], [GROUP, 4 // GROUP], [1, 0],
    )
    row = group * GROUP + gl.arange(0, GROUP, layout=gl.SliceLayout(1, layout))
    x = gl.arange(0, 512, layout=gl.SliceLayout(0, layout))
    t = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    q = gl.arange(0, COPY_BLOCK, layout=gl.SliceLayout(0, layout))
    slot = gl.load(Locations + row, row < M, other=-1)
    latent = gl.load(
        Latent + row[:, None] * LS0 + x[None, :] * LS1,
        row[:, None] < M, other=0,
    )
    tail = gl.load(
        Tail + row[:, None] * TS0 + t[None, :] * TS1,
        row[:, None] < M, other=0,
    )
    query_tail = gl.load(
        Q + row[:, None] * QS0 + (q[None, :] // 64) * QS1
        + (128 + q[None, :] % 64) * QS2,
        (row[:, None] < M) & (q[None, :] < H * 64), other=0,
    )
    write = slot > 0


    group_ids = group * GROUP + gl.arange(
        0, GROUP, layout=gl.BlockedLayout([1], [64], [4], [0]),
    )
    group_slots = gl.load(Locations + group_ids, group_ids < M, other=-1)
    any_zero = gl.max((group_slots == 0).to(gl.int32), 0)
    if any_zero:


        scan_layout: gl.constexpr = gl.SliceLayout(
            0, gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0]),
        )
        ids = gl.arange(0, LOCATION_BLOCK, layout=scan_layout)
        slots = gl.load(Locations + ids, ids < M, other=-1)
        last_zero = gl.max(gl.where((ids < M) & (slots == 0), ids, -1), 0)
        write = (slot > 0) | ((slot == 0) & (row == last_zero))

    gl.store(
        Fresh + row[:, None] * 576 + x[None, :], latent,
        row[:, None] < M, cache_modifier=".cs",
    )
    gl.store(
        Fresh + row[:, None] * 576 + 512 + t[None, :], tail,
        row[:, None] < M, cache_modifier=".cs",
    )
    gl.store(
        Cache + slot[:, None] * CS0 + x[None, :] * CS1, latent,
        (row[:, None] < M) & write[:, None], cache_modifier=".cs",
    )
    gl.store(
        Cache + slot[:, None] * CS0 + (512 + t[None, :]) * CS1, tail,
        (row[:, None] < M) & write[:, None], cache_modifier=".cs",
    )
    gl.store(
        Out + row[:, None] * H * 576 + (q[None, :] // 64) * 576
        + 512 + q[None, :] % 64,
        query_tail, (row[:, None] < M) & (q[None, :] < H * 64),
        cache_modifier=".cs",
    )


@gluon.jit
def _m128_project_tile(
    Q, W, Out, head, tile_m, tile_n,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[1, 4],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a, [32, 128], gl.bfloat16,
    )
    shared_b: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_b, [128, 64], gl.bfloat16,
    )
    load_a: gl.constexpr = _m128_physical_copy_layout(shared_a, [32, 128])
    load_b: gl.constexpr = _m128_physical_copy_layout(shared_b, [128, 64])
    rows = gl.arange(0, 32, gl.SliceLayout(1, load_a))
    ka = gl.arange(0, 128, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, 128, gl.SliceLayout(1, load_b))
    cols = gl.arange(0, 64, gl.SliceLayout(0, load_b))
    smem_a = gl.allocate_shared_memory(gl.bfloat16, (32, 128), shared_a)
    smem_b = gl.allocate_shared_memory(gl.bfloat16, (128, 64), shared_b)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_b,
        W + head * WS0 + tile_n * 64 * WS2,
        kb[:, None] * WS1 + cols[None, :] * WS2,
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_a,
        Q + head * QS1,
        (rows[:, None] + tile_m * 32) * QS0 + ka[None, :] * QS2,
        rows[:, None] + tile_m * 32 < M, other=0,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(1)
    b_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_b, dot_b)
    gl.amd.cdna4.async_copy.wait_group(0)
    for piece in gl.static_range(2):
        a_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_a.slice(piece * 16, 16, dim=0), dot_a,
        )
        acc = gl.amd.cdna4.mfma(
            a_dot, b_dot, gl.full((16, 64), 0, gl.float32, mma),
        )
        out_rows = gl.arange(0, 16, gl.SliceLayout(1, mma))
        out_cols = gl.arange(0, 64, gl.SliceLayout(0, mma))
        gl.amd.cdna4.buffer_store(
            acc.to(gl.bfloat16), Out + head * 576,
            (out_rows[:, None] + tile_m * 32 + piece * 16) * H * 576
            + out_cols[None, :] + tile_n * 64,
            out_rows[:, None] + tile_m * 32 + piece * 16 < M,
            cache=".cs",
        )


@gluon.jit
def _m128_project_large_tile(
    Q, W, Out, head, tile_m, tile_n,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=False,
        warps_per_cta=[2, 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_a, [64, 64], gl.bfloat16,
    )
    shared_b: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        dot_b, [64, 64], gl.bfloat16,
    )
    load_a: gl.constexpr = _m128_physical_copy_layout(shared_a, [64, 64])
    load_b: gl.constexpr = _m128_physical_copy_layout(shared_b, [64, 64])
    a0 = gl.allocate_shared_memory(gl.bfloat16, (64, 64), shared_a)
    b0 = gl.allocate_shared_memory(gl.bfloat16, (64, 64), shared_b)
    a1 = gl.allocate_shared_memory(gl.bfloat16, (64, 64), shared_a)
    b1 = gl.allocate_shared_memory(gl.bfloat16, (64, 64), shared_b)
    rows = gl.arange(0, 64, gl.SliceLayout(1, load_a))
    ka = gl.arange(0, 64, gl.SliceLayout(0, load_a))
    kb = gl.arange(0, 64, gl.SliceLayout(1, load_b))
    cols = gl.arange(0, 64, gl.SliceLayout(0, load_b))

    row_offsets = (rows & ~4) * QS0 + (rows & 4) * QS0
    a_offsets = row_offsets[:, None] + ka[None, :] * QS2
    b_offsets = kb[:, None] * WS1 + (cols[None, :] + tile_n * 64) * WS2
    a_base = Q + head * QS1 + tile_m * 64 * QS0
    b_base = W + head * WS0
    valid = rows[:, None] + tile_m * 64 < M
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a0, a_base, a_offsets, valid, other=0,
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b0, b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b1, b_base + 64 * WS1, b_offsets,
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a1, a_base + 64 * QS2, a_offsets, valid, other=0,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(2)
    a_first = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, dot_a)
    gl.amd.cdna4.async_copy.wait_group(1)
    b_first = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, dot_b)


    with gl.amd.warp_pipeline_stage("first_half"):
        acc = gl.amd.cdna4.mfma(
            a_first, b_first, gl.full((64, 64), 0, gl.float32, mma),
        )
    gl.amd.cdna4.async_copy.wait_group(0)
    b_second = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, dot_b)
    a_second = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, dot_a)

    acc = gl.amd.cdna4.mfma(a_second, b_second, acc)
    out_rows = gl.arange(0, 64, gl.SliceLayout(1, mma))
    out_cols = gl.arange(0, 64, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        acc.to(gl.bfloat16),
        Out + tile_m * 64 * H * 576 + head * 576 + tile_n * 64,
        out_rows[:, None] * H * 576 + out_cols[None, :],
        out_rows[:, None] + tile_m * 64 < M,
        cache=".cs",
    )


@gluon.jit
def _m128_project_strided_tile(
    Q, W, Out, head, tile_m, tile_n,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):

    if M > 128:
        BM: gl.constexpr = 64
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 16], transposed=False,
            warps_per_cta=[2, 2],
        )
    else:
        BM: gl.constexpr = 32
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 32], transposed=False,
            warps_per_cta=[1, 4],
        )
    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 4], [0, 1])
    r = gl.arange(0, BM, gl.SliceLayout(1, a_layout)) + tile_m * BM
    ak = gl.arange(0, 64, gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, 64, gl.SliceLayout(1, b_layout))
    c = gl.arange(0, 64, gl.SliceLayout(0, b_layout)) + tile_n * 64
    acc = gl.full((BM, 64), 0, gl.float32, mma)
    for piece in gl.static_range(2):
        a = gl.load(
            Q + r[:, None] * QS0 + head * QS1 + (ak[None, :] + piece * 64) * QS2,
            r[:, None] < M, other=0,
        )
        b = gl.load(
            W + head * WS0 + (bk[:, None] + piece * 64) * WS1 + c[None, :] * WS2,
        )
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)),
            gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8)), acc,
        )
    out_rows = gl.arange(0, BM, gl.SliceLayout(1, mma)) + tile_m * BM
    out_cols = gl.arange(0, 64, gl.SliceLayout(0, mma)) + tile_n * 64
    gl.store(
        Out + out_rows[:, None] * H * 576 + head * 576 + out_cols[None, :],
        acc.to(gl.bfloat16), out_rows[:, None] < M,
    )


@gluon.jit
def _m128_project_and_cache(
    Q, Latent, Tail, W, Locations, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LOCATION_BLOCK: gl.constexpr, COPY_BLOCK: gl.constexpr, PROJECT_MASK: gl.constexpr,
):
    raw = gl.program_id(0).to(gl.uint32)
    if M > 128:
        ROW_TILES: gl.constexpr = gl.cdiv(M, 64)
        GROUP: gl.constexpr = 4
        VECTOR: gl.constexpr = 8
    else:
        ROW_TILES: gl.constexpr = gl.cdiv(M, 32)
        GROUP: gl.constexpr = 2
        VECTOR: gl.constexpr = 4
    PREP_TILES: gl.constexpr = gl.cdiv(M, GROUP)
    PROJECT_TILES: gl.constexpr = H * ROW_TILES * 8
    PREP_BEGIN: gl.constexpr = min(192, PROJECT_TILES)
    prepare = (raw >= PREP_BEGIN) & (raw < PREP_BEGIN + PREP_TILES)
    prep_id = raw - PREP_BEGIN
    pid = gl.where(raw < PREP_BEGIN, raw, raw - PREP_TILES)
    pid = pid & PROJECT_MASK
    if prepare:
        _m128_prepare_keys_and_tails(
            Q, Latent, Tail, Locations, Cache, Out, Fresh, prep_id,
            M, H, QS0, QS1, QS2, LS0, LS1, TS0, TS1, CS0, CS1,
            LOCATION_BLOCK, COPY_BLOCK, GROUP, VECTOR,
        )
    else:
        head = pid // (ROW_TILES * 8)
        tile_n = pid // ROW_TILES % 8
        tile_m = pid % ROW_TILES
        if (QS2 != 1 or QS0 % 8 != 0 or QS1 % 8 != 0
                or WS1 != 1 or WS0 % 8 != 0 or WS2 % 8 != 0):
            _m128_project_strided_tile(
                Q, W, Out, head, tile_m, tile_n, M, H,
                QS0, QS1, QS2, WS0, WS1, WS2,
            )
        elif M > 128:
            _m128_project_large_tile(
                Q, W, Out, head, tile_m, tile_n, M, H,
                QS0, QS1, QS2, WS0, WS1, WS2,
            )
        else:
            _m128_project_tile(
                Q, W, Out, head, tile_m, tile_n, M, H,
                QS0, QS1, QS2, WS0, WS1, WS2,
            )


def mla_kc_cache_m128(
    query: torch.Tensor, latent: torch.Tensor, key_tail: torch.Tensor,
    weight: torch.Tensor, locations: torch.Tensor, cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    block_m, rows_per_prepare = (32, 2)
    grid = (triton.cdiv(m, rows_per_prepare) + h * triton.cdiv(m, block_m) * 8,)
    _m128_project_and_cache[grid](
        query, latent, key_tail, weight, locations, cache, out, fresh, m, h,
        *query.stride(), *latent.stride(), *key_tail.stride(),
        *weight.stride(), *cache.stride(),
        triton.next_power_of_2(m), triton.next_power_of_2(max(576, h * 64)),
        triton.next_power_of_2(h * triton.cdiv(m, block_m) * 8) - 1,
        num_warps=4,
    )
    return out, fresh


@gluon.jit
def _m1_4_8_project_tile(
    Q, W, Out, head, row_start, col_start,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 2], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 2], [0, 1])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 2],
    )
    rows = row_start + gl.arange(0, 16, gl.SliceLayout(1, a_layout))


    if M < 16:
        load_rows = rows % M
        mask = gl.full((16, 1), True, gl.int1, a_layout)
    else:
        load_rows = rows
        mask = rows[:, None] < M
    ak = gl.arange(0, 32, gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, 32, gl.SliceLayout(1, b_layout))
    cols = col_start + gl.arange(0, 32, gl.SliceLayout(0, b_layout))
    zero = gl.full((16, 32), 0, gl.float32, mma_layout)
    acc = zero

    for block in gl.static_range(4):
        a_offsets = load_rows[:, None] * QS0 + (block * 32 + ak[None, :]) * QS2
        b_offsets = (block * 32 + bk[:, None]) * WS1 + cols[None, :] * WS2
        a = gl.load(Q + head * QS1 + a_offsets, mask, other=0)
        b = gl.load(W + head * WS0 + b_offsets)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma_layout, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma_layout, 8))
        acc += gl.amd.cdna4.mfma(a, b, zero)
    out_rows = row_start + gl.arange(0, 16, gl.SliceLayout(1, mma_layout))
    out_cols = col_start + gl.arange(0, 32, gl.SliceLayout(0, mma_layout))
    gl.store(
        Out + out_rows[:, None] * H * 576 + head * 576 + out_cols[None, :],
        acc.to(gl.bfloat16), out_rows[:, None] < M,
    )


@gluon.jit
def _m1_4_8_prepare_tile(
    Q, KV, Tail, Loc, Cache, Out, Fresh, row, chunk,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LM: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
    x = chunk * 256 + gl.arange(0, 256, layout)
    latent = gl.load(KV + row * KS0 + x * KS1, x < 512, other=0)
    tail = gl.load(
        Tail + row * TS0 + (x - 512) * TS1,
        (x >= 512) & (x < 576), other=0,
    )
    key = gl.where(x < 512, latent, tail)
    gl.store(Fresh + row * 576 + x, key, x < 576)

    slot = gl.load(Loc + row).to(gl.int64)
    if M == 1:
        write = slot >= 0
    else:
        ids = gl.arange(0, LM, layout)
        locations = gl.load(Loc + ids, ids < M, other=-1)
        last_zero = gl.max(gl.where((ids < M) & (locations == 0), ids, -1), 0)

        write = (slot > 0) | ((slot == 0) & (row == last_zero))
    gl.store(Cache + slot * CS0 + x * CS1, key, (x < 576) & write)

    query_tail = gl.load(
        Q + row * QS0 + (x // 64) * QS1 + (128 + x % 64) * QS2,
        x < H * 64, other=0,
    )
    gl.store(
        Out + row * H * 576 + (x // 64) * 576 + 512 + x % 64,
        query_tail, x < H * 64,
    )


@gluon.jit
def _m1_4_8_fused_kc_cache(
    Q, KV, Tail, W, Loc, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr, LM: gl.constexpr,
):
    pid = gl.program_id(0)
    row_tiles: gl.constexpr = gl.cdiv(M, 16)
    aux_chunks: gl.constexpr = gl.cdiv(max(576, H * 64), 256)
    project_tiles: gl.constexpr = H * row_tiles * 16

    if pid < project_tiles:
        head = pid // (16 * row_tiles)
        row_start = pid // 16 % row_tiles * 16
        col_start = pid % 16 * 32
        _m1_4_8_project_tile(
            Q, W, Out, head, row_start, col_start,
            M, H, QS0, QS1, QS2, WS0, WS1, WS2,
        )
    else:
        auxiliary_id = pid - project_tiles
        _m1_4_8_prepare_tile(
            Q, KV, Tail, Loc, Cache, Out, Fresh,
            auxiliary_id // aux_chunks, auxiliary_id % aux_chunks,
            M, H, QS0, QS1, QS2, KS0, KS1, TS0, TS1, CS0, CS1,
            LM,
        )


def mla_kc_cache_m1_4_8(
    query: torch.Tensor,
    latent: torch.Tensor,
    key_tail: torch.Tensor,
    weight: torch.Tensor,
    locations: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    grid = h * triton.cdiv(m, 16) * 16 + m * triton.cdiv(max(576, h * 64), 256)
    _m1_4_8_fused_kc_cache[(grid,)](
        query, latent, key_tail, weight, locations, cache, out, fresh, m, h,
        *query.stride(), *latent.stride(), *key_tail.stride(),
        *weight.stride(), *cache.stride(), triton.next_power_of_2(m), num_warps=2,
    )
    return out, fresh


@triton.constexpr_function
def _m256_1024_8192_tile_rows(m):

    return (64, 4) if m > 128 else (32, 2)


@triton.constexpr_function
def _m256_1024_8192_physical_copy_layout(shared):

    bases = shared.offset_bases
    return gl.DistributedLinearLayout(
        reg_bases=bases[:3] + bases[11:],
        lane_bases=bases[3:9], warp_bases=bases[9:11],
        block_bases=[], shape=shared.shape,
    )


@gluon.jit
def _m256_1024_8192_allocate_operand(dot: gl.constexpr, rows: gl.constexpr, cols: gl.constexpr):

    shared: gl.constexpr = cdna4.compute_efficient_padded_shared_layout(
        dot, [rows, cols], gl.bfloat16,
    )
    copy: gl.constexpr = _m256_1024_8192_physical_copy_layout(shared)
    i = gl.arange(0, rows, gl.SliceLayout(1, copy))
    j = gl.arange(0, cols, gl.SliceLayout(0, copy))
    panel = gl.allocate_shared_memory(gl.bfloat16, (rows, cols), shared)
    return panel, i, j


@gluon.jit
def _m256_1024_8192_project_full_k(
    Q, W, Out, head, tile_m, tile_n,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False,
        warps_per_cta=[1, 4],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    smem_a, rows, ka = _m256_1024_8192_allocate_operand(dot_a, 32, 128)
    smem_b, kb, cols = _m256_1024_8192_allocate_operand(dot_b, 128, 64)
    async_copy.buffer_load_to_shared(
        smem_b, W + head * WS0,
        kb[:, None] * WS1 + (cols[None, :] + tile_n * 64) * WS2,
    )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        smem_a, Q + head * QS1 + tile_m * (32 * QS0),
        rows[:, None] * QS0 + ka[None, :] * QS2,
        rows[:, None] + tile_m * 32 < M, other=0,
    )
    async_copy.commit_group()
    async_copy.wait_group(1)
    b_dot = async_copy.load_shared_relaxed(smem_b, dot_b)
    async_copy.wait_group(0)


    for piece in gl.static_range(2):
        a_dot = async_copy.load_shared_relaxed(
            smem_a.slice(piece * 16, 16, dim=0), dot_a,
        )
        acc = cdna4.mfma(
            a_dot, b_dot, gl.full((16, 64), 0, gl.float32, mma),
        )
        out_rows = gl.arange(0, 16, gl.SliceLayout(1, mma))
        out_cols = gl.arange(0, 64, gl.SliceLayout(0, mma))
        cdna4.buffer_store(
            acc.to(gl.bfloat16), Out + head * 576,
            (out_rows[:, None] + tile_m * 32 + piece * 16) * H * 576
            + out_cols[None, :] + tile_n * 64,
            out_rows[:, None] + tile_m * 32 + piece * 16 < M,
            cache=".cs",
        )


@gluon.jit
def _m256_1024_8192_project_split_k(
    Q, W, Out, head, tile_m, tile_n,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=False,
        warps_per_cta=[2, 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    a0_smem, rows, ka = _m256_1024_8192_allocate_operand(dot_a, 64, 64)
    b0_smem, kb, cols = _m256_1024_8192_allocate_operand(dot_b, 64, 64)
    a1_smem = gl.allocate_shared_memory(gl.bfloat16, (64, 64), a0_smem.layout)
    b1_smem = gl.allocate_shared_memory(gl.bfloat16, (64, 64), b0_smem.layout)
    q_base = Q + (head * QS1 + tile_m * (64 * QS0))

    a_offsets = (
        (rows[:, None] & ~4) * QS0
        + (rows[:, None] & 4) * QS0 + ka[None, :] * QS2
    )
    b_offsets = kb[:, None] * WS1 + (cols[None, :] + tile_n * 64) * WS2
    valid_rows = rows[:, None] + tile_m * 64 < M


    async_copy.buffer_load_to_shared(
        a0_smem, q_base, a_offsets, valid_rows, other=0,
    )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        b0_smem, W + head * WS0, b_offsets,
    )
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(
        b1_smem, W + head * WS0, b_offsets + 64 * WS1,
    )
    async_copy.buffer_load_to_shared(
        a1_smem, q_base, a_offsets + 64 * QS2, valid_rows, other=0,
    )
    async_copy.commit_group()
    async_copy.wait_group(2)
    a0 = async_copy.load_shared_relaxed(a0_smem, dot_a)
    async_copy.wait_group(1)
    b0 = async_copy.load_shared_relaxed(b0_smem, dot_b)

    with gl.amd.warp_pipeline_stage("first_half"):
        acc = cdna4.mfma(
            a0, b0, gl.full((64, 64), 0, gl.float32, mma),
        )
    async_copy.wait_group(0)
    b1 = async_copy.load_shared_relaxed(b1_smem, dot_b)
    a1 = async_copy.load_shared_relaxed(a1_smem, dot_a)
    acc = cdna4.mfma(a1, b1, acc)
    out_rows = gl.arange(0, 64, gl.SliceLayout(1, mma))
    out_cols = gl.arange(0, 64, gl.SliceLayout(0, mma))
    cdna4.buffer_store(
        acc.to(gl.bfloat16),
        Out + ((tile_m * (64 * H) + head) * 576 + tile_n * 64),
        out_rows[:, None] * H * 576 + out_cols[None, :],
        out_rows[:, None] + tile_m * 64 < M,
        cache=".cs",
    )


@gluon.jit
def _m256_1024_8192_cache_write_owner(Locations, slot, row, M: gl.constexpr, BLOCK: gl.constexpr):

    write = slot > 0
    if gl.max((slot == 0).to(gl.int32), 0):

        scan: gl.constexpr = gl.BlockedLayout([max(1, BLOCK // 64)], [64], [4], [0])
        ids = gl.arange(0, BLOCK, layout=scan)
        slots = gl.load(Locations + ids, ids < M, other=-1)
        last_zero = gl.max(gl.where((ids < M) & (slots == 0), ids, -1), 0)
        write = (slot > 0) | ((slot == 0) & (row == last_zero))
    return write


@gluon.jit
def _m256_1024_8192_prepare_rows(
    Q, Latent, Tail, Locations, Cache, Out, Fresh, group,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LOCATION_BLOCK: gl.constexpr, COPY_BLOCK: gl.constexpr,
):
    GROUP: gl.constexpr = _m256_1024_8192_tile_rows(M)[1]
    if GROUP == 4:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [2, 2], [1, 0])
    row = group * GROUP + gl.arange(0, GROUP, layout=gl.SliceLayout(1, layout))
    x = gl.arange(0, COPY_BLOCK, layout=gl.SliceLayout(0, layout))
    slot = gl.load(Locations + row, row < M, other=-1)
    latent = gl.load(
        Latent + row[:, None] * LS0 + x[None, :] * LS1,
        (row[:, None] < M) & (x[None, :] < 512), other=0,
    )
    tail = gl.load(
        Tail + row[:, None] * TS0 + (x[None, :] - 512) * TS1,
        (row[:, None] < M) & (x[None, :] >= 512) & (x[None, :] < 576),
        other=0,
    )
    query_tail = gl.load(
        Q + row[:, None] * QS0 + (x[None, :] // 64) * QS1
        + (128 + x[None, :] % 64) * QS2,
        (row[:, None] < M) & (x[None, :] < H * 64), other=0,
    )
    key = gl.where(x[None, :] < 512, latent, tail)
    gl.store(
        Fresh + row[:, None] * 576 + x[None, :], key,
        (row[:, None] < M) & (x[None, :] < 576), cache_modifier=".cs",
    )
    write = _m256_1024_8192_cache_write_owner(Locations, slot, row, M, LOCATION_BLOCK)

    gl.store(
        Cache + slot[:, None] * CS0 + x[None, :] * CS1, key,
        (row[:, None] < M) & (x[None, :] < 576) & write[:, None],
        cache_modifier=".cs",
    )
    gl.store(
        Out + row[:, None] * H * 576 + (x[None, :] // 64) * 576
        + 512 + x[None, :] % 64,
        query_tail, (row[:, None] < M) & (x[None, :] < H * 64),
        cache_modifier=".cs",
    )


@gluon.jit
def _m256_1024_8192_project_and_cache(
    Q, Latent, Tail, W, Locations, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LOCATION_BLOCK: gl.constexpr, COPY_BLOCK: gl.constexpr,
):

    raw = gl.program_id(0).to(gl.uint32)
    ROW_TILES: gl.constexpr = gl.cdiv(M, _m256_1024_8192_tile_rows(M)[0])
    PREP_TILES: gl.constexpr = gl.cdiv(M, _m256_1024_8192_tile_rows(M)[1])
    PROJECT_TILES: gl.constexpr = H * ROW_TILES * 8
    PREP_BEGIN: gl.constexpr = PROJECT_TILES // 2
    prepare = (raw >= PREP_BEGIN) & (raw < PREP_BEGIN + PREP_TILES)
    prep_id = raw - PREP_BEGIN
    pid = gl.where(raw < PREP_BEGIN, raw, raw - PREP_TILES)
    if prepare:
        _m256_1024_8192_prepare_rows(
            Q, Latent, Tail, Locations, Cache, Out, Fresh, prep_id,
            M, H, QS0, QS1, QS2, LS0, LS1, TS0, TS1, CS0, CS1,
            LOCATION_BLOCK, COPY_BLOCK,
        )
    else:

        gl.assume(pid < PROJECT_TILES)

        head = pid // (ROW_TILES * 8)
        tile_n = pid // ROW_TILES % 8
        tile_m = pid % ROW_TILES
        gl.assume(head < H)
        gl.assume(tile_n < 8)
        gl.assume(tile_m < ROW_TILES)
        project: gl.constexpr = _m256_1024_8192_project_split_k if M > 128 else _m256_1024_8192_project_full_k
        project(
            Q, W, Out, head, tile_m, tile_n, M, H,
            QS0, QS1, QS2, WS0, WS1, WS2,
        )


def mla_kc_cache_m256_1024_8192(
    query: torch.Tensor, latent: torch.Tensor, key_tail: torch.Tensor,
    weight: torch.Tensor, locations: torch.Tensor, cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    block_m, rows_per_prepare = _m256_1024_8192_tile_rows(m)
    grid = (triton.cdiv(m, rows_per_prepare) + h * triton.cdiv(m, block_m) * 8,)
    _m256_1024_8192_project_and_cache[grid](
        query, latent, key_tail, weight, locations, cache, out, fresh, m, h,
        *query.stride(), *latent.stride(), *key_tail.stride(),
        *weight.stride(), *cache.stride(),
        triton.next_power_of_2(m), triton.next_power_of_2(max(576, h * 64)),
        num_warps=4,
    )
    return out, fresh


@gluon.jit
def _m2_16_project_tile(
    Q, W, Out, head, row_base, col_base,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [0, 1])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    rows = row_base + gl.arange(0, 16, layout=gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, 32, layout=gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, 32, layout=gl.SliceLayout(1, b_layout))
    cols = col_base + gl.arange(0, 32, layout=gl.SliceLayout(0, b_layout))


    if M < 16:
        read_rows = rows % M
    else:
        read_rows = rows
    zero = gl.full((16, 32), 0, gl.float32, mma_layout)
    acc = zero
    for block in gl.static_range(4):
        a_ptrs = Q + head * QS1 + read_rows[:, None] * QS0 + (block * 32 + ak[None, :]) * QS2
        if M < 16:
            a = gl.load(a_ptrs)
        else:
            a = gl.load(a_ptrs, rows[:, None] < M, other=0)
        b = gl.load(W + head * WS0 + (block * 32 + bk[:, None]) * WS1 + cols[None, :] * WS2)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma_layout, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma_layout, 8))


        acc = acc + gl.amd.cdna4.mfma(a, b, zero)

    out_rows = row_base + gl.arange(0, 16, layout=gl.SliceLayout(1, mma_layout))
    out_cols = col_base + gl.arange(0, 32, layout=gl.SliceLayout(0, mma_layout))
    gl.store(
        Out + out_rows[:, None] * H * 576 + head * 576 + out_cols[None, :],
        acc.to(gl.bfloat16), out_rows[:, None] < M,
    )


@gluon.jit
def _m2_16_prepare_row(
    Q, KV, Tail, Loc, Cache, Out, Fresh, row,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LM: gl.constexpr, B: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    x = gl.arange(0, B, layout=layout)
    latent = gl.load(KV + row * KS0 + x * KS1, x < 512, other=0)
    tail = gl.load(Tail + row * TS0 + (x - 512) * TS1, (x >= 512) & (x < 576), other=0)
    key = gl.where(x < 512, latent, tail)
    qr = gl.load(
        Q + row * QS0 + (x // 64) * QS1 + (128 + x % 64) * QS2,
        x < H * 64, other=0,
    )
    slot = gl.load(Loc + row).to(gl.int64)
    ids = gl.arange(0, LM, layout=layout)
    locations = gl.load(Loc + ids, ids < M, other=-1)

    gl.store(Fresh + row * 576 + x, key, x < 576)


    last_zero = gl.max(gl.where((ids < M) & (locations == 0), ids, -1), 0)
    write = (slot > 0) | ((slot == 0) & (row == last_zero))
    gl.store(Cache + slot * CS0 + x * CS1, key, (x < 576) & write)
    gl.store(Out + row * H * 576 + (x // 64) * 576 + 512 + x % 64, qr, x < H * 64)


@gluon.jit
def _m2_16_kimi_cache_kernel(
    Q, W, KV, Tail, Loc, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LM: gl.constexpr, B: gl.constexpr,
):
    tile = gl.program_id(0)
    head = gl.program_id(1)
    project_tiles: gl.constexpr = gl.cdiv(M, 16) * 16
    if tile < project_tiles:
        if M <= 16:
            row_base = 0
            col_base = tile * 32
        else:
            row_base = (tile // 16) * 16
            col_base = (tile % 16) * 32
        _m2_16_project_tile(
            Q, W, Out, head, row_base, col_base,
            M, H, QS0, QS1, QS2, WS0, WS1, WS2,
        )
    else:
        row = (tile - project_tiles) * H + head
        if row < M:
            _m2_16_prepare_row(
                Q, KV, Tail, Loc, Cache, Out, Fresh, row,
                M, H, QS0, QS1, QS2, KS0, KS1, TS0, TS1, CS0, CS1, LM, B,
            )


def mla_kc_cache_m2_16(
    query: torch.Tensor, latent: torch.Tensor, key_tail: torch.Tensor,
    weight: torch.Tensor, locations: torch.Tensor, cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    grid = (triton.cdiv(m, 16) * 16 + triton.cdiv(m, h), h)
    _m2_16_kimi_cache_kernel[grid](
        query, weight, latent, key_tail, locations, cache, out, fresh, m, h,
        *query.stride(), *weight.stride(), *latent.stride(), *key_tail.stride(), *cache.stride(),
        triton.next_power_of_2(m), triton.next_power_of_2(max(576, h * 64)),
        num_warps=1,
    )
    return out, fresh


@gluon.jit
def _m32_project_tile_small(
    Q, W, Out, head, row_block, col_block,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    BN: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32],
        transposed=False, warps_per_cta=[1, 2],
    )

    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [2, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 2], [0, 1])
    ar = gl.arange(0, 16, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, 32, gl.SliceLayout(0, load_a))
    bk = gl.arange(0, 32, gl.SliceLayout(1, load_b))
    bc = gl.arange(0, BN, gl.SliceLayout(0, load_b))

    qbase = Q + (row_block * 16 * QS0 + head * QS1)
    wbase = W + (head * WS0 + col_block * BN * WS2)
    qoffset = ar[:, None] * QS0 + ak[None, :] * QS2
    woffset = bk[:, None] * WS1 + bc[None, :] * WS2
    acc = gl.zeros((16, BN), gl.float32, mma)
    for block in gl.static_range(4):
        a = gl.load(
            qbase + (qoffset + block * 32 * QS2),
            row_block * 16 + ar[:, None] < M, other=0,
        )
        b = gl.load(wbase + (woffset + block * 32 * WS1))
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))

        acc = gl.amd.cdna4.mfma(a, b, acc)
    om = gl.arange(0, 16, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    obase = Out + (row_block * 16 * H * 576 + head * 576 + col_block * BN)
    gl.store(
        obase + (om[:, None] * H * 576 + on[None, :]),
        acc.to(gl.bfloat16), row_block * 16 + om[:, None] < M,
    )


@gluon.jit
def _m32_project_tile_large(
    Q, W, Out, head, row_block, col_block,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32],
        transposed=False, warps_per_cta=[1, 1],
    )
    load_a: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
    load_b: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [0, 1])
    ar = gl.arange(0, 16, gl.SliceLayout(1, load_a))
    ak = gl.arange(0, 32, gl.SliceLayout(0, load_a))
    bk = gl.arange(0, 32, gl.SliceLayout(1, load_b))
    bc = gl.arange(0, 16, gl.SliceLayout(0, load_b))
    qbase = Q + (row_block * 16 * QS0 + head * QS1)
    wbase = W + (head * WS0 + col_block * 32 * WS2)
    qoffset = ar[:, None] * QS0 + ak[None, :] * QS2
    woffset = bk[:, None] * WS1 + bc[None, :] * WS2


    acc0 = gl.zeros((16, 16), gl.float32, mma)
    acc1 = gl.zeros((16, 16), gl.float32, mma)
    for block in gl.static_range(4):
        a = gl.load(
            qbase + (qoffset + block * 32 * QS2),
            row_block * 16 + ar[:, None] < M, other=0,
        )
        b0 = gl.load(wbase + (woffset + block * 32 * WS1))
        b1 = gl.load(wbase + (woffset + block * 32 * WS1 + 16 * WS2))
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b0 = gl.convert_layout(b0, gl.DotOperandLayout(1, mma, 8))
        b1 = gl.convert_layout(b1, gl.DotOperandLayout(1, mma, 8))
        acc0 = gl.amd.cdna4.mfma(a, b0, acc0)
        acc1 = gl.amd.cdna4.mfma(a, b1, acc1)
    om = gl.arange(0, 16, gl.SliceLayout(1, mma))
    on = gl.arange(0, 16, gl.SliceLayout(0, mma))
    obase = Out + (row_block * 16 * H * 576 + head * 576 + col_block * 32)
    ooffset = om[:, None] * H * 576 + on[None, :]
    gl.store(
        obase + ooffset, acc0.to(gl.bfloat16),
        row_block * 16 + om[:, None] < M,
    )
    gl.store(
        obase + (ooffset + 16), acc1.to(gl.bfloat16),
        row_block * 16 + om[:, None] < M,
    )


@gluon.jit
def _m32_prepare_row(
    Q, KV, Tail, Loc, Cache, Out, Fresh, row,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LM: gl.constexpr, B: gl.constexpr,
):
    waves: gl.constexpr = 2 if M <= 32 else 1
    vector: gl.constexpr = 4 if M <= 32 else 8
    layout: gl.constexpr = gl.BlockedLayout([vector], [64], [waves], [0])
    x = gl.arange(0, B, layout)
    slot = gl.load(Loc + row).to(gl.int64)
    write = slot > 0
    if slot == 0:


        ids = gl.arange(0, LM, gl.BlockedLayout([1], [64], [waves], [0]))
        locations = gl.load(Loc + ids, ids < M, other=-1)
        last_zero = gl.max(gl.where((ids < M) & (locations == 0), ids, -1), 0)
        write = row == last_zero
    qr = gl.load(
        Q + row * QS0 + (x // 64) * QS1 + (128 + x % 64) * QS2,
        x < H * 64, other=0,
    )
    latent = gl.load(KV + row * KS0 + x * KS1, x < 512, other=0)
    tail = gl.load(
        Tail + row * TS0 + (x - 512) * TS1,
        (x >= 512) & (x < 576), other=0,
    )
    key = gl.where(x < 512, latent, tail)
    gl.store(Fresh + row * 576 + x, key, x < 576)
    gl.store(Cache + slot * CS0 + x * CS1, key, (x < 576) & write)
    gl.store(
        Out + row * H * 576 + (x // 64) * 576 + 512 + x % 64,
        qr, x < H * 64,
    )


@gluon.jit
def _m32_fused(
    Q, KV, Tail, W, Loc, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    KS0: gl.constexpr, KS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    LM: gl.constexpr, B: gl.constexpr,
):

    pid = gl.program_id(0)
    if pid < M:
        _m32_prepare_row(
            Q, KV, Tail, Loc, Cache, Out, Fresh, pid,
            M, H, QS0, QS1, QS2, KS0, KS1, TS0, TS1, CS0, CS1, LM, B,
        )
    else:
        tile = pid - M
        columns: gl.constexpr = 8 if M <= 32 else 16
        head = tile // (columns * gl.cdiv(M, 16))
        if M <= 32:

            row_block = tile % gl.cdiv(M, 16)
            col_block = (tile // gl.cdiv(M, 16)) % 8
            _m32_project_tile_small(
                Q, W, Out, head, row_block, col_block,
                M, H, QS0, QS1, QS2, WS0, WS1, WS2, 64,
            )
        else:

            row_block = (tile // 2) % gl.cdiv(M, 16)
            col_block = tile % 2 + ((tile // (2 * gl.cdiv(M, 16))) % 8) * 2
            _m32_project_tile_large(
                Q, W, Out, head, row_block, col_block,
                M, H, QS0, QS1, QS2, WS0, WS1, WS2,
            )


def mla_kc_cache_m32(
    query: torch.Tensor, latent: torch.Tensor, key_tail: torch.Tensor,
    weight: torch.Tensor, locations: torch.Tensor, cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    columns = 8
    _m32_fused[(h * triton.cdiv(m, 16) * columns + m,)](
        query, latent, key_tail, weight, locations, cache, out, fresh, m, h,
        *query.stride(), *latent.stride(), *key_tail.stride(),
        *weight.stride(), *cache.stride(),
        triton.next_power_of_2(m), triton.next_power_of_2(max(576, h * 64)),
        num_warps=2, waves_per_eu=4,
    )
    return out, fresh


@gluon.jit
def _m64_prepare_row(
    Q, Latent, Tail, Locations, Cache, Out, Fresh, row,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    COPY_SIZE: gl.constexpr, LOCATION_SIZE: gl.constexpr,
    CACHE_OFFSETS_32: gl.constexpr,
):
    gl.assume(row >= 0)
    gl.assume(row < M)

    if M == 32:
        slot = gl.load(Locations + row).to(gl.int64)

    COPY_VECTOR: gl.constexpr = 2 if M == 64 else 4
    copy_layout: gl.constexpr = gl.BlockedLayout([COPY_VECTOR], [64], [1], [0])
    x = gl.arange(0, COPY_SIZE, layout=copy_layout)


    if max((M - 1) * QS0 + (H - 1) * QS1 + 191 * QS2,
           (M - 1) * LS0 + 511 * LS1,
           (M - 1) * TS0 + 63 * TS1,
           M * H * 576, 575 * CS1) >= 2**31:
        row = row.to(gl.int64)
        x = x.to(gl.int64)
    latent = gl.load(Latent + (row * LS0 + x * LS1), x < 512, other=0)
    tail = gl.load(
        Tail + (row * TS0 + (x - 512) * TS1),
        (x >= 512) & (x < 576), other=0,
    )
    key = gl.where(x < 512, latent, tail)
    gl.store(Fresh + (row * 576 + x), key, x < 576)


    if M == 32:
        qx = gl.arange(
            0, COPY_SIZE, layout=gl.BlockedLayout([2], [64], [1], [0])
        )
        if max((M - 1) * QS0 + (H - 1) * QS1 + 191 * QS2,
               M * H * 576) >= 2**31:
            qx = qx.to(gl.int64)
        q_tail = gl.load(
            Q + (row * QS0 + (qx // 64) * QS1 + (128 + qx % 64) * QS2),
            qx < H * 64, other=0,
        )
        gl.store(
            Out + (row * H * 576 + (qx // 64) * 576 + 512 + qx % 64),
            q_tail, qx < H * 64,
        )


    if M != 32:
        slot = gl.load(Locations + row).to(gl.int64)
    write = slot > 0
    if slot == 0:
        if M == 32:
            location_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
            ids = gl.arange(0, 32, layout=location_layout)
            slots = gl.load(Locations + ids, ids < M, other=-1)


            zero_mask = gl.inline_asm_elementwise(
                "v_cmp_eq_u64 $0, 0, $1", "=s,v", [slots],
                dtype=gl.uint64, is_pure=True, pack=1,
            )
            first = gl.full((1,), 0, gl.int32, location_layout)
            zero_mask = gl.sum(gl.gather(zero_mask, first, 0), 0).to(gl.uint32)

            write = (zero_mask >> row) == 1
        else:
            ids = gl.arange(
                0, LOCATION_SIZE, layout=gl.BlockedLayout([1], [64], [1], [0])
            )
            slots = gl.load(Locations + ids, ids < M, other=-1)
            last = gl.max(gl.where((ids < M) & (slots == 0), ids, -1), 0)
            write = row == last
    if CACHE_OFFSETS_32:

        cache_offset = slot.to(gl.int32) * CS0 + x.to(gl.int32) * CS1
        gl.store(Cache + cache_offset, key, (x < 576) & write)
    else:
        gl.store(Cache + slot * CS0 + x * CS1, key, (x < 576) & write)

    if M != 32:
        q_tail = gl.load(
            Q + (row * QS0 + (x // 64) * QS1 + (128 + x % 64) * QS2),
            x < H * 64, other=0,
        )
        gl.store(
            Out + (row * H * 576 + (x // 64) * 576 + 512 + x % 64),
            q_tail, x < H * 64,
        )


@gluon.jit
def _m64_project_tile(
    Q, W, Out, tile,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    USE_ASYNC: gl.constexpr,
):
    ROW_TILES: gl.constexpr = triton.cdiv(M, 16)
    gl.assume(tile >= 0)
    gl.assume(tile < H * ROW_TILES * 16)


    if M == 64 and H == 12:
        local = 6 * (tile % 8) + tile // 128
        head = local // 4
        row_tile = local % 4
        col_tile = (tile // 8) % 16
    else:


        if (M == 32 and H % 4 == 0) or (M == 64 and H % 2 == 0):
            GROUP: gl.constexpr = 4 if M == 32 else 2
            head = tile % GROUP + ((tile // (GROUP * ROW_TILES)) % (H // GROUP)) * GROUP
            row_tile = (tile // GROUP) % ROW_TILES
        elif M >= 64:
            row_tile = tile % ROW_TILES
            head = (tile // ROW_TILES) % H
        else:
            head = tile % H
            row_tile = (tile // H) % ROW_TILES
        col_tile = tile // (H * ROW_TILES)
    gl.assume(head >= 0)
    gl.assume(head < H)
    gl.assume(row_tile >= 0)
    gl.assume(row_tile < ROW_TILES)
    gl.assume(col_tile >= 0)
    gl.assume(col_tile < 16)

    a_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [1, 0])
    b_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 1], [0, 1])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, 1]
    )
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    local_rows = gl.arange(0, 16, layout=gl.SliceLayout(1, a_layout))
    rows = row_tile * 16 + local_rows
    ak = gl.arange(0, 32, layout=gl.SliceLayout(0, a_layout))
    bk = gl.arange(0, 32, layout=gl.SliceLayout(1, b_layout))
    local_cols = gl.arange(0, 32, layout=gl.SliceLayout(0, b_layout))
    cols = col_tile * 32 + local_cols

    zero = gl.full((16, 32), 0, gl.float32, mfma_layout)
    if USE_ASYNC:
        a_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        b_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
        q_base = Q + (row_tile * 16 * QS0 + head * QS1)
        qa = local_rows[:, None] * QS0 + ak[None, :]
        w_base = W + (head * WS0 + col_tile * 32 * WS2)
        half_cols = gl.arange(0, 16, layout=gl.SliceLayout(0, b_layout))
        wb_half = bk[:, None] + half_cols[None, :] * WS2
        half_zero = gl.full((16, 16), 0, gl.float32, mfma_layout)

        bl0 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        bl1 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        bl2 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        bl3 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        a0 = gl.allocate_shared_memory(gl.bfloat16, [16, 32], a_shared_layout)
        a1 = gl.allocate_shared_memory(gl.bfloat16, [16, 32], a_shared_layout)
        a2 = gl.allocate_shared_memory(gl.bfloat16, [16, 32], a_shared_layout)
        a3 = gl.allocate_shared_memory(gl.bfloat16, [16, 32], a_shared_layout)
        br0 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        br1 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        br2 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)
        br3 = gl.allocate_shared_memory(gl.bfloat16, [32, 16], b_shared_layout)


        async_copy.buffer_load_to_shared(bl0, w_base, wb_half)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(a0, q_base, qa)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(br0, w_base, wb_half + 16 * WS2)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(bl1, w_base, wb_half + 32)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(a1, q_base, qa + 32)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(br1, w_base, wb_half + 32 + 16 * WS2)
        async_copy.commit_group()


        async_copy.wait_group(5)
        bl = async_copy.load_shared_relaxed(bl0, b_dot)
        async_copy.wait_group(4)
        av = async_copy.load_shared_relaxed(a0, a_dot)
        async_copy.wait_group(3)
        br = async_copy.load_shared_relaxed(br0, b_dot)
        async_copy.buffer_load_to_shared(bl2, w_base, wb_half + 64)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(a2, q_base, qa + 64)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(br2, w_base, wb_half + 64 + 16 * WS2)
        async_copy.commit_group()
        acc_l = half_zero + gl.amd.cdna4.mfma(av, bl, half_zero)
        acc_r = half_zero + gl.amd.cdna4.mfma(av, br, half_zero)

        async_copy.wait_group(5)
        bl = async_copy.load_shared_relaxed(bl1, b_dot)
        async_copy.wait_group(4)
        av = async_copy.load_shared_relaxed(a1, a_dot)
        async_copy.wait_group(3)
        br = async_copy.load_shared_relaxed(br1, b_dot)
        async_copy.buffer_load_to_shared(bl3, w_base, wb_half + 96)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(a3, q_base, qa + 96)
        async_copy.commit_group()
        async_copy.buffer_load_to_shared(br3, w_base, wb_half + 96 + 16 * WS2)
        async_copy.commit_group()
        acc_l = acc_l + gl.amd.cdna4.mfma(av, bl, half_zero)
        acc_r = acc_r + gl.amd.cdna4.mfma(av, br, half_zero)


        if M == 64:
            gl.inline_asm_elementwise(
                "", "=v,v", [acc_r], dtype=gl.float32, is_pure=False, pack=1
            )


        async_copy.wait_group(5)
        bl = async_copy.load_shared_relaxed(bl2, b_dot)
        async_copy.wait_group(4)
        av = async_copy.load_shared_relaxed(a2, a_dot)
        async_copy.wait_group(3)
        br = async_copy.load_shared_relaxed(br2, b_dot)
        acc_l = acc_l + gl.amd.cdna4.mfma(av, bl, half_zero)
        acc_r = acc_r + gl.amd.cdna4.mfma(av, br, half_zero)

        async_copy.wait_group(2)
        bl = async_copy.load_shared_relaxed(bl3, b_dot)
        async_copy.wait_group(1)
        av = async_copy.load_shared_relaxed(a3, a_dot)
        async_copy.wait_group(0)
        br = async_copy.load_shared_relaxed(br3, b_dot)
        acc_l = acc_l + gl.amd.cdna4.mfma(av, bl, half_zero)
        acc_r = acc_r + gl.amd.cdna4.mfma(av, br, half_zero)
    else:

        rows = rows.to(gl.int64)
        head = head.to(gl.int64)
        ak = ak.to(gl.int64)
        bk = bk.to(gl.int64)
        cols = cols.to(gl.int64)
        acc = zero
        for block in gl.static_range(4):
            a = gl.load(
                Q + (rows[:, None] * QS0 + head * QS1 + (ak[None, :] + block * 32) * QS2),
                rows[:, None] < M, other=0,
            )
            b = gl.load(
                W + (head * WS0 + (bk[:, None] + block * 32) * WS1 + cols[None, :] * WS2)
            )
            a = gl.convert_layout(a, a_dot)
            b = gl.convert_layout(b, b_dot)

            acc = acc + gl.amd.cdna4.mfma(a, b, zero)

    out_rows = row_tile * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, mfma_layout))
    if USE_ASYNC:

        out_base = Out + ((row_tile * 16 * H + head) * 576 + col_tile * 32)
        mr = gl.arange(0, 16, layout=gl.SliceLayout(1, mfma_layout))
        nc = gl.arange(0, 16, layout=gl.SliceLayout(0, mfma_layout))
        offsets = mr[:, None] * (H * 576) + nc[None, :]
        gl.store(out_base + offsets, acc_l.to(gl.bfloat16))
        gl.store(out_base + offsets + 16, acc_r.to(gl.bfloat16))
    else:
        out_cols = col_tile * 32 + gl.arange(0, 32, layout=gl.SliceLayout(0, mfma_layout))
        offsets = (out_rows[:, None].to(gl.int64) * H + head) * 576 + out_cols[None, :]
        gl.store(Out + offsets, acc.to(gl.bfloat16), out_rows[:, None] < M)


@gluon.jit
def _m64_fused(
    Q, Latent, Tail, W, Locations, Cache, Out, Fresh,
    M: gl.constexpr, H: gl.constexpr,
    QS0: gl.constexpr, QS1: gl.constexpr, QS2: gl.constexpr,
    LS0: gl.constexpr, LS1: gl.constexpr,
    TS0: gl.constexpr, TS1: gl.constexpr,
    WS0: gl.constexpr, WS1: gl.constexpr, WS2: gl.constexpr,
    CS0: gl.constexpr, CS1: gl.constexpr,
    COPY_SIZE: gl.constexpr, LOCATION_SIZE: gl.constexpr,
    USE_ASYNC: gl.constexpr,
    CACHE_OFFSETS_32: gl.constexpr,
):
    pid = gl.program_id(0)
    if pid < M:
        _m64_prepare_row(
            Q, Latent, Tail, Locations, Cache, Out, Fresh, pid, M, H,
            QS0, QS1, QS2, LS0, LS1, TS0, TS1, CS0, CS1,
            COPY_SIZE, LOCATION_SIZE, CACHE_OFFSETS_32,
        )
    else:
        _m64_project_tile(Q, W, Out, pid - M, M, H, QS0, QS1, QS2, WS0, WS1, WS2, USE_ASYNC)


def mla_kc_cache_m64(
    query: torch.Tensor,
    latent: torch.Tensor,
    key_tail: torch.Tensor,
    weight: torch.Tensor,
    locations: torch.Tensor,
    cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    m, h, _ = query.shape
    out = torch.empty((m, h, 576), device=query.device, dtype=query.dtype)
    fresh = torch.empty((m, 576), device=query.device, dtype=query.dtype)
    use_async = (
        m % 16 == 0 and query.stride(2) == 1 and weight.stride(1) == 1
        and query.storage_offset() % 8 == 0 and weight.storage_offset() % 8 == 0
        and m * h * 576 < 2**31
        and all(s % 8 == 0 for s in (*query.stride()[:2], weight.stride(0), weight.stride(2)))
        and 2 * ((m - 1) * query.stride(0) + (h - 1) * query.stride(1) + 127) < 2**31
        and 2 * ((h - 1) * weight.stride(0) + 511 * weight.stride(2) + 127) < 2**31
    )

    cache_offsets_32 = (
        m == 64
        and (cache.shape[0] - 1) * cache.stride(0) + 575 * cache.stride(1) < 2**31
    )
    _m64_fused[(m + h * triton.cdiv(m, 16) * 16,)](
        query, latent, key_tail, weight, locations, cache, out, fresh,
        m, h, *query.stride(), *latent.stride(), *key_tail.stride(),
        *weight.stride(), *cache.stride(),
        triton.next_power_of_2(max(576, h * 64)), triton.next_power_of_2(m),
        use_async, cache_offsets_32,
        num_warps=1,
    )
    return out, fresh
