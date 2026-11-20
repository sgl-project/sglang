# SPDX-License-Identifier: Apache-2.0
"""Step5 index normalization, compression, scoring and block selection."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.deep_select import topk

# Index Q/K normalization and RoPE.


@triton.jit(
    do_not_specialize=("ROWS",),
    do_not_specialize_on_alignment=("ROWS",),
)
def _indexer_norm_rope(
    Q,
    K,
    QW,
    KW,
    KB,
    Cache,
    Positions,
    QOut,
    KOut,
    ROWS,
    Q_ROW: tl.constexpr,
    K_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    real_row = token < ROWS
    channel = tl.arange(0, 256)
    if head < HEADS:
        x = tl.load(Q + token * Q_ROW + head * 256 + channel, real_row, other=0).to(
            tl.float32
        )
        w = tl.load(QW + channel).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / 256 + EPS)
        y = x * inverse_rms * (w + ZERO_CENTERED)
    else:
        x = tl.load(K + token * K_ROW + channel, real_row, other=0).to(tl.float32)
        w = tl.load(KW + channel).to(tl.float32)
        b = tl.load(KB + channel).to(tl.float32)
        mean = tl.sum(x, 0) / 256
        variance = tl.sum(x * x, 0) / 256 - mean * mean
        y = (x - mean) * tl.rsqrt(variance + EPS) * w + b

    # Normalization has its own BF16 boundary, including the rotated partner.
    normalized = y.to(tl.bfloat16).to(tl.float32)
    partner = tl.gather(normalized, channel ^ 16, axis=0)
    position = tl.load(Positions + token, real_row, other=0).to(tl.int64)
    cosine = tl.load(Cache + position * 32 + channel % 16, channel < 32, other=0).to(
        tl.float32
    )
    sine = tl.load(Cache + position * 32 + 16 + channel % 16, channel < 32, other=0).to(
        tl.float32
    )
    rotated = normalized * cosine + tl.where(channel < 16, -partner, partner) * sine
    result = tl.where(channel < 32, rotated, normalized).to(tl.bfloat16)
    if head < HEADS:
        tl.store(QOut + (token * HEADS + head) * 256 + channel, result)
    if head == HEADS:
        tl.store(KOut + token * 256 + channel, result, real_row)


def step5_indexer_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-centered RMSNorm on index Q, LayerNorm on index K, then RoPE on dims 0-31.

    Q is BF16 [T, 2 or 4, 256] and K is BF16 [T, 1, 256]; rows may be padded.
    """
    assert q.ndim == 3 and q.shape[1] in (2, 4) and q.shape[2] == 256
    tokens, heads, _ = q.shape
    assert k.shape == (tokens, 1, 256)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert q.stride(1) == 256 and q.stride(2) == 1 and k.stride(2) == 1
    assert cache.shape[1] == 32 and cache.is_contiguous()
    q_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    k_out = torch.empty(k.shape, dtype=torch.bfloat16, device=k.device)
    if tokens:
        _indexer_norm_rope[(tokens, heads + 1)](
            q,
            k,
            q_weight,
            k_weight,
            k_bias,
            cache,
            positions,
            q_out,
            k_out,
            tokens,
            q.stride(0),
            k.stride(0),
            heads,
            eps,
            1,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out


# Index-key compression: metadata, ring state and publication.


@triton.jit
def _count_request_owners(
    Slots, Live, Uses, ROWS: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(Slots + row, row < ROWS, other=0).to(tl.int64)
    live = tl.load(Live)
    slot = tl.minimum(tl.maximum(raw, 0), CAP - 1)
    tl.atomic_add(
        Uses + slot, 1, (row < ROWS) & (row < live) & (raw != 0), sem="relaxed"
    )


@triton.jit
def _build_request_metadata(
    Slots,
    Bases,
    Live,
    Mapping,
    Uses,
    SafeSlots,
    Starts,
    Sources,
    Lengths,
    Active,
    Admitted,
    CountValid,
    Writes,
    QuerySlots,
    QueryPositions,
    ROWS: tl.constexpr,
    CAP: tl.constexpr,
    TOKENS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    MAP_COL_STRIDE: tl.constexpr,
    PHYSICAL_LIMIT: tl.constexpr,
    CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    QBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    raw = tl.load(Slots + row).to(tl.int64)
    slot = tl.minimum(tl.maximum(raw, 0), CAP - 1)
    live = tl.load(Live).to(tl.int64)
    active = (row < live) & (raw != 0)
    count_ok = (live >= 0) & (live <= ROWS)
    base = tl.load(Bases + row).to(tl.int64)
    depth = tl.arange(0, QBLOCK)
    pos = base + depth
    safe_pos = tl.minimum(tl.maximum(pos, 0), TOKENS - 1)
    current = tl.load(Mapping + slot * MAP_STRIDE + safe_pos * MAP_COL_STRIDE)
    query_ok = (
        (pos >= 0)
        & (pos < tl.minimum(TOKENS, CONTEXT))
        & (current >= 64)
        & (current < PHYSICAL_LIMIT)
    )
    # Validate each completed R8 group once; current-token mappings above stay
    # per query because native FULL pages need not be physically adjacent.
    block_idx = tl.arange(0, RBLOCK)
    closing_depth = 7 - (base % 8 + 8) % 8 + block_idx * 8
    closing_pos = tl.minimum(
        tl.maximum(base + tl.minimum(closing_depth, WIDTH - 1), 0), TOKENS - 1
    )
    offset = tl.arange(0, 8)
    logical = closing_pos[:, None] // 8 * 8 + offset[None, :]
    mapped = tl.load(
        Mapping + slot * MAP_STRIDE + tl.minimum(logical, TOKENS - 1) * MAP_COL_STRIDE
    )
    first = tl.sum(tl.where(offset[None, :] == 0, mapped.to(tl.int64), 0), 1)
    block_ok = (
        (first >= 64)
        & (first < PHYSICAL_LIMIT)
        & (first % 8 == 0)
        & tl.min((mapped == first[:, None] + offset[None, :]).to(tl.int32), 1)
    )
    completed = (block_idx < BLOCKS) & (closing_depth < WIDTH)
    admitted = (
        count_ok
        & (slot == raw)
        & (tl.load(Uses + slot) == 1)
        & tl.min(tl.where(depth < WIDTH, query_ok, True).to(tl.int32), 0)
        & tl.min((~completed | block_ok).to(tl.int32), 0)
    )
    tl.store(SafeSlots + row, slot)
    tl.store(Starts + row, tl.minimum(tl.maximum(base, 0), TOKENS - 1))
    tl.store(Sources + row, row * WIDTH)
    tl.store(Lengths + row, WIDTH)
    tl.store(Active + row, active)
    tl.store(Admitted + row, admitted)
    if row == 0:
        tl.store(CountValid, count_ok)

    # Destinations are stored in [completed-block, request] order. An incomplete
    # block is never consumed but still gets a deterministic value for graph replay.
    tl.store(
        Writes + block_idx * ROWS + row,
        (first - (first % 8 + 8) % 8) // 8,
        block_idx < BLOCKS,
    )

    query_pos = safe_pos
    tl.store(QueryPositions + row * WIDTH + depth, query_pos, depth < WIDTH)
    tl.store(QuerySlots + row * WIDTH + depth, tl.where(active, raw, 0), depth < WIDTH)


def build_compression_metadata(
    *,
    request_slots: torch.Tensor,
    positions: torch.Tensor,
    valid_rows: torch.Tensor,
    req_to_token: torch.Tensor,
    compressed_index_key_pages: int,
    context_len: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Build the decode compression plan (one input token per request).

    Returns the plan fields plus int32 per-query request slots and positions.
    """
    device = req_to_token.device
    rows = request_slots.numel()
    num_slots, token_capacity = req_to_token.shape
    token_width = 1
    num_blocks = (7 + token_width) // 8
    result = dict(
        request_slots=torch.empty(rows, dtype=torch.int64, device=device),
        input_start_positions=torch.empty(rows, dtype=torch.int64, device=device),
        input_start_rows=torch.empty(rows, dtype=torch.int64, device=device),
        input_lengths=torch.empty(rows, dtype=torch.int64, device=device),
        request_active=torch.empty(rows, dtype=torch.bool, device=device),
        request_mapping_valid=torch.empty(rows, dtype=torch.bool, device=device),
        request_count_valid=torch.empty(1, dtype=torch.bool, device=device),
        compressed_key_write_indices=torch.empty(
            (num_blocks, rows), dtype=torch.int64, device=device
        ),
        num_input_tokens=rows * token_width,
    )
    query_slots = torch.empty(rows * token_width, dtype=torch.int32, device=device)
    query_positions = torch.empty(rows * token_width, dtype=torch.int32, device=device)
    uses = torch.zeros(num_slots, dtype=torch.int32, device=device)
    _count_request_owners[(triton.cdiv(rows, 128),)](
        request_slots, valid_rows, uses, rows, num_slots, 128
    )
    _build_request_metadata[(rows,)](
        request_slots,
        positions,
        valid_rows,
        req_to_token,
        uses,
        result["request_slots"],
        result["input_start_positions"],
        result["input_start_rows"],
        result["input_lengths"],
        result["request_active"],
        result["request_mapping_valid"],
        result["request_count_valid"],
        result["compressed_key_write_indices"],
        query_slots,
        query_positions,
        rows,
        num_slots,
        token_capacity,
        *req_to_token.stride(),
        compressed_index_key_pages * 64,
        context_len,
        token_width,
        triton.next_power_of_2(token_width),
        triton.next_power_of_2(num_blocks),
        num_blocks,
        num_warps=4,
    )
    return result, query_slots, query_positions


@triton.jit
def _fresh_product(weight, value):
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p; mul.rn.f32 p, $1, $2; add.rn.f32 $0, p, 0f00000000; }",
        constraints="=f,f,f",
        args=[weight, value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _compressed_index_key_bytes(index_keys, compression_logits):
    # raw payloads retain their BF16 bits, including unused NaN slots, until
    # all eight tokens are ready for the numerical reduction.
    maximum = tl.full((256,), -float("inf"), tl.float32)
    for token in tl.static_range(8):
        indices = tl.full((1, 256), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((256,))
        logit = logit.to(tl.bfloat16, bitcast=True).to(tl.float32)
        maximum = tl.maximum(maximum, logit, propagate_nan=tl.PropagateNan.NONE)
    block_max = tl.max(maximum, axis=0)
    numerator = tl.full((256,), 0.0, tl.float32)
    denominator = tl.full((256,), 0.0, tl.float32)
    for token in tl.static_range(8):
        indices = tl.full((1, 256), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((256,))
        value = tl.gather(index_keys, indices, axis=0).reshape((256,))
        logit = logit.to(tl.bfloat16, bitcast=True).to(tl.float32)
        value = value.to(tl.bfloat16, bitcast=True).to(tl.float32)
        weight = tl.exp2((logit - block_max) * 1.4426950408889634)
        if token == 0:
            numerator = _fresh_product(weight, value)
        else:
            numerator = tl.fma(weight, value, numerator)
        denominator = denominator + weight
    mean = tl.div_rn(
        numerator,
        tl.maximum(denominator, 1.0e-20, propagate_nan=tl.PropagateNan.NONE),
    )
    mean = tl.where(denominator > 0.0, mean, 0.0)
    rounded = mean.to(tl.bfloat16).to(tl.float32)
    packed = tl.inline_asm_elementwise(
        """{
            .reg .b16 pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, $1;
            cvt.u32.u16 $0, pair;
        }""",
        constraints="=r,f",
        args=[rounded],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8)


@triton.jit
def _admit_row(
    row,
    slots,
    bases,
    active,
    admitted,
    count_valid,
    tags,
    enabled,
    success,
    CAPACITY: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
):
    live = tl.load(active + row) != 0
    ok = tl.full((), False, tl.int1)
    if live & (tl.load(admitted + row) != 0) & (tl.load(count_valid) != 0):
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        pos = base // 8 * 8 + tl.arange(0, 8)
        actual = tl.load(tags + slot * CAPACITY + pos % CAPACITY)
        history_ok = tl.sum(((pos < base) & (actual != pos)).to(tl.int32), 0) == 0
        ok = history_ok
    tl.store(enabled + row, ok)
    row_ok = (tl.load(count_valid) != 0) & (~live | ok)
    tl.store(success + row, row_ok)
    if ASSERT_SUCCESS:
        tl.device_assert(row_ok, "Step5 CSA ring history failed")
    return ok


@triton.jit
def _admit(
    slots,
    bases,
    active,
    admitted,
    count_valid,
    tags,
    enabled,
    success,
    CAPACITY: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
):
    _admit_row(
        tl.program_id(0),
        slots,
        bases,
        active,
        admitted,
        count_valid,
        tags,
        enabled,
        success,
        CAPACITY,
        ASSERT_SUCCESS,
    )


@triton.jit
def _compress(
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    physical,
    enabled,
    index_key_ring,
    compression_logits_ring,
    paged,
    active,
    admitted,
    count_valid,
    tags,
    success,
    ADMIT: tl.constexpr,
    ASSERT_SUCCESS: tl.constexpr,
    CAPACITY: tl.constexpr,
    ROWS: tl.constexpr,
    Z_ROW: tl.constexpr,
):
    block_offset, row = tl.program_id(0), tl.program_id(1)
    if ADMIT:
        # Single-block launches admit in the same CTA; multi-block prefill admits
        # in a separate launch instead of repeating admission per block.
        ok = _admit_row(
            row,
            slots,
            bases,
            active,
            admitted,
            count_valid,
            tags,
            enabled,
            success,
            CAPACITY,
            ASSERT_SUCCESS,
        )
    else:
        ok = tl.load(enabled + row)
    if ok:
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        first = base // 8 * 8
        block_idx = first + block_offset * 8
        if block_idx + 8 <= end:
            slot = tl.load(slots + row).to(tl.int64)
            source = tl.load(sources + row).to(tl.int64)
            pos = block_idx + tl.arange(0, 8)
            ch = tl.arange(0, 256)
            history = pos < base
            ring_offset = (slot * CAPACITY + pos % CAPACITY)[:, None] * 256 + ch[
                None, :
            ]
            input_offset = (source + pos - base)[:, None] * 256 + ch[None, :]
            rk = tl.load(
                index_key_ring.to(tl.pointer_type(tl.uint16)) + ring_offset,
                history[:, None],
                0,
            )
            rz = tl.load(
                compression_logits_ring.to(tl.pointer_type(tl.uint16)) + ring_offset,
                history[:, None],
                0,
            )
            ik = tl.load(
                index_key.to(tl.pointer_type(tl.uint16)) + input_offset,
                ~history[:, None],
                0,
            )
            iz = tl.load(
                compression_logits.to(tl.pointer_type(tl.uint16))
                + (source + pos - base)[:, None] * Z_ROW
                + ch[None, :],
                ~history[:, None],
                0,
            )
            packed = _compressed_index_key_bytes(
                tl.where(history[:, None], rk, ik), tl.where(history[:, None], rz, iz)
            )
            dest = tl.load(physical + block_offset * ROWS + row).to(tl.int64)
            within = dest % 8
            sw128 = (
                (ch // 128) * 1024
                + within * 128
                + (((ch % 128) // 16) ^ within) * 16
                + ch % 16
            )
            tl.store(paged + dest // 8 * 2048 + sw128, packed)


@triton.jit
def _write_row(
    row,
    enabled,
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    index_key_ring,
    compression_logits_ring,
    tags,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    Z_ROW: tl.constexpr,
):
    if enabled:
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        source = tl.load(sources + row).to(tl.int64)
        # One lane per ring cell, selecting the newest input with that residue.
        cell = tl.arange(0, BLOCK)
        pos = end - 1 - (end - 1 + CAPACITY - cell) % CAPACITY
        write = (cell < CAPACITY) & (pos >= base)
        ch = tl.arange(0, 256)
        src = (source + pos - base)[:, None] * 256 + ch[None, :]
        dst = (slot * CAPACITY + cell)[:, None] * 256 + ch[None, :]
        ik = tl.load(index_key.to(tl.pointer_type(tl.uint16)) + src, write[:, None], 0)
        iz = tl.load(
            compression_logits.to(tl.pointer_type(tl.uint16))
            + (source + pos - base)[:, None] * Z_ROW
            + ch[None, :],
            write[:, None],
            0,
        )
        tl.store(
            index_key_ring.to(tl.pointer_type(tl.uint16)) + dst, ik, write[:, None]
        )
        tl.store(
            compression_logits_ring.to(tl.pointer_type(tl.uint16)) + dst,
            iz,
            write[:, None],
        )
        previous = tl.load(tags + slot * CAPACITY + cell, cell < CAPACITY, -1)
        # At a compression boundary no earlier input can be needed again.
        # Invalidate unwritten cells, including stale tags from a reused slot.
        tag = tl.where(write, pos, tl.where(base % 8 == 0, -1, previous))
        tl.store(tags + slot * CAPACITY + cell, tag, cell < CAPACITY)


@triton.jit
def _write(
    index_key,
    compression_logits,
    slots,
    bases,
    sources,
    lengths,
    enabled,
    index_key_ring,
    compression_logits_ring,
    tags,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    Z_ROW: tl.constexpr,
):
    row = tl.program_id(0)
    _write_row(
        row,
        tl.load(enabled + row),
        index_key,
        compression_logits,
        slots,
        bases,
        sources,
        lengths,
        index_key_ring,
        compression_logits_ring,
        tags,
        CAPACITY,
        BLOCK,
        Z_ROW,
    )


def append_ring(
    *,
    slots: torch.Tensor,
    bases: torch.Tensor,
    sources: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
    admitted: torch.Tensor,
    count_valid: torch.Tensor,
    physical_blocks: torch.Tensor,
    enabled: torch.Tensor,
    success: torch.Tensor,
    index_key: torch.Tensor,
    compression_logits: torch.Tensor,
    index_key_ring: torch.Tensor,
    compression_logits_ring: torch.Tensor,
    ring_positions: torch.Tensor,
    compressed_index_keys: torch.Tensor,
) -> None:
    """Compress completed R8 blocks into the SW128 FP8 index-key pages, then keep
    each request's newest K/Z inputs in its ring for the next forward."""
    assert index_key.is_contiguous() and compression_logits.stride(-1) == 1
    rows = slots.numel()
    max_blocks = physical_blocks.shape[0]
    capacity = index_key_ring.shape[1]
    # Ring history is read by _compress before _write overwrites it.
    if max_blocks != 1:
        _admit[(rows,)](
            slots,
            bases,
            active,
            admitted,
            count_valid,
            ring_positions,
            enabled,
            success,
            capacity,
            True,
            debug=True,
        )
    if max_blocks:
        _compress[(max_blocks, rows)](
            index_key,
            compression_logits,
            slots,
            bases,
            sources,
            lengths,
            physical_blocks,
            enabled,
            index_key_ring,
            compression_logits_ring,
            compressed_index_keys,
            active,
            admitted,
            count_valid,
            ring_positions,
            success,
            max_blocks == 1,
            True,
            capacity,
            rows,
            compression_logits.stride(0),
            num_warps=8,
            enable_fp_fusion=True,
            debug=True,
        )
    _write[(rows,)](
        index_key,
        compression_logits,
        slots,
        bases,
        sources,
        lengths,
        enabled,
        index_key_ring,
        compression_logits_ring,
        ring_positions,
        capacity,
        triton.next_power_of_2(capacity),
        compression_logits.stride(0),
        num_warps=8,
    )


# Indexer logits.


@triton.jit(
    do_not_specialize=("BLOCKS",),
    do_not_specialize_on_alignment=("BLOCKS",),
)
def _score_blocks(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    BLOCKS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    head = tl.arange(0, 16)
    channel = tl.arange(0, 256)
    request = tl.load(RowToReq + row).to(tl.int64)
    position = tl.load(Positions + row)
    query = tl.load(
        Q + (row * HEADS + head[:, None]) * 256 + channel[None, :],
        (request > 0) & (head[:, None] < HEADS),
        other=0.0,
    ).to(tl.bfloat16)
    weights = (
        tl.load(W + row * HEADS + head, (request > 0) & (head < HEADS), other=0).to(
            tl.float32
        )
        * WEIGHT_SCALE
    )
    last_tile = tl.cdiv(BLOCKS, BLOCK_R)
    for tile in range(first_tile, last_tile, GRID_TILES):
        num_blocks = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        if (request > 0) & (tile * BLOCK_R < position // 8):
            valid = num_blocks < position // 8
            token = tl.load(
                ReqToToken + request * MAP_STRIDE + num_blocks * 8,
                valid,
                other=0,
            ).to(tl.int64)
            page, slot = token // 64, token % 64 // 8
            # Each page contains two 128-byte channel atoms, each with eight
            # swizzled block rows. XOR uses the physical row, never logical ID.
            offset = (
                page[None, :] * 2048
                + (channel[:, None] // 128) * 1024
                + slot[None, :] * 128
                + (((channel[:, None] % 128) // 16) ^ slot[None, :]) * 16
                + channel[:, None] % 16
            )
            packed = tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)
            keys = packed.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            products = tl.dot(query, keys)
            score = tl.sum(tl.maximum(products, 0.0) * weights[:, None], axis=0)
            score = tl.where(valid, score, -float("inf"))
        else:
            score = tl.full((BLOCK_R,), -float("inf"), tl.float32)
        tl.store(Out + row * BLOCKS + num_blocks, score, num_blocks < BLOCKS)


def step5_indexer_decode_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Score each query row against its completed history blocks into FP32 ``out``.

    Q is BF16 [T, 2 or 4, 256] and weights are FP32 [T, heads], already scaled;
    future and padding columns of ``out`` are written as -inf.
    """
    assert q.ndim == 3 and q.shape[1] in (2, 4) and q.shape[2] == 256
    tokens, heads, _ = q.shape
    assert q.is_contiguous() and weights.is_contiguous()
    assert weights.shape == (tokens, heads) and weights.dtype == torch.float32
    assert compressed_index_keys.shape[1:] == (8, 1, 256)
    assert compressed_index_keys.dtype == torch.uint8
    assert out.shape[0] == tokens and out.dtype == torch.float32
    assert out.is_contiguous()
    # D256 gathers are register-heavy; 32-block tiles also give small batches more CTAs.
    blocks_per_tile = 32
    grid_tiles = min(triton.cdiv(out.shape[1], blocks_per_tile), 128)
    _score_blocks[(tokens, grid_tiles)](
        q.to(torch.float8_e4m3fn),
        weights,
        compressed_index_keys,
        req_to_token,
        row_to_req,
        positions,
        out,
        heads,
        req_to_token.stride(0),
        out.shape[1],
        blocks_per_tile,
        grid_tiles,
        1.0,
        num_warps=4,
    )


# Block top-k and packed attention addresses.


@triton.jit
def _canonicalize_zero(Scores, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(Scores + offset, offset < N, other=0.0).to(tl.uint32, bitcast=True)
    # FlashInfer's radix ordering distinguishes signed zero. Preserve all other
    # bits, including subnormals, so numerical ties prefer the smaller block ID.
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    tl.store(Out + offset, bits.to(tl.float32, bitcast=True), offset < N)


@triton.jit
def _pack_blocks(
    Indices,
    Positions,
    Requests,
    Out,
    Counts,
    Mapping,
    Packed,
    Lengths,
    K: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 512)
    position = tl.load(Positions + row).to(tl.int32)
    current = position // 8
    request = tl.load(Requests + row).to(tl.int64)
    live = request > 0
    count = tl.minimum(current, 512)
    if K > 0:
        ids = tl.load(Indices + row * K + lane, lane < K, other=2147483647).to(tl.int32)
        ids = tl.where((ids >= 0) & (ids < current), ids, 2147483647)
        ids = tl.sort(ids, descending=False)
    else:
        ids = lane
    result = tl.where(lane < count, ids, tl.where(lane == count, current, -1))
    tl.store(Out + row * 513 + lane, tl.where(live, result, -1))
    tl.store(Out + row * 513 + 512, tl.where(live & (count == 512), current, -1))
    tl.store(Counts + row, tl.where(live, count + 1, 0))

    if PACKED_WIDTH > 0:
        active = live & (lane <= count)
        logical = result.to(tl.int64) * 8
        physical = tl.load(
            Mapping + request * MAP_STRIDE + logical, active, other=0
        ).to(tl.int64)
        value = ((physical // 8) << 32) | logical
        tl.store(Packed + row * PACKED_WIDTH + lane, tl.where(active, value, -1))

        # The current block occupies slot 512 when all history slots are used.
        # Clear the remaining ABI padding for widths 513, 514 and 516 as well.
        tail = tl.arange(0, 4)
        active_tail = live & (count == 512) & (tail == 0)
        logical_tail = current.to(tl.int64) * 8
        physical_tail = tl.load(
            Mapping + request * MAP_STRIDE + logical_tail + tl.zeros((4,), tl.int64),
            active_tail,
            other=0,
        ).to(tl.int64)
        value_tail = ((physical_tail // 8) << 32) | logical_tail
        tl.store(
            Packed + row * PACKED_WIDTH + 512 + tail,
            tl.where(active_tail, value_tail, -1),
            512 + tail < PACKED_WIDTH,
        )
        tl.store(Lengths + row, tl.where(live, position + 1, 0))


def _launch_block_pack(
    *,
    indices,
    positions,
    row_to_req,
    block_indices,
    counts,
    selected_k,
    req_to_token=None,
    packed=None,
    causal_lengths=None,
):
    rows = block_indices.shape[0]
    _pack_blocks[(rows,)](
        indices,
        positions,
        row_to_req,
        block_indices,
        counts,
        req_to_token,
        packed,
        causal_lengths,
        selected_k,
        req_to_token.stride(0) if packed is not None else 0,
        packed.shape[1] if packed is not None else 0,
        num_warps=1 if packed is not None and rows >= 512 else 4,
    )


STEP5_HISTORY_BLOCKS = 512
# Preserve the score workspace layout while sharing the public top-k kernel.
_SCORE_STRIDE_ALIGNMENT = 256


def can_use_step5_topk_deepselect(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 10
    )


def deepselect_score_width(score_width: int) -> int:
    align = _SCORE_STRIDE_ALIGNMENT
    return (score_width + align - 1) // align * align


def step5_topk_blocks_deepselect(
    scores: torch.Tensor,
    *,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    history_lengths: torch.Tensor,
    req_to_token: torch.Tensor,
    topk_ids: torch.Tensor,
    block_indices: torch.Tensor,
    counts: torch.Tensor,
    packed: torch.Tensor,
    causal_lengths: torch.Tensor,
) -> None:
    """SM10x ``step5_topk_blocks`` that also writes the packed attention addresses.

    ``history_lengths`` is int32 positions // 8 per row, zero for padding rows.
    """
    topk(
        scores,
        STEP5_HISTORY_BLOCKS,
        end=history_lengths,
        output_idx=topk_ids,
        idx_oob_fill_value=-1,
    )
    _launch_block_pack(
        indices=topk_ids,
        positions=positions,
        row_to_req=row_to_req,
        block_indices=block_indices,
        counts=counts,
        selected_k=STEP5_HISTORY_BLOCKS,
        req_to_token=req_to_token,
        packed=packed,
        causal_lengths=causal_lengths,
    )


def step5_topk_blocks(
    scores: torch.Tensor,
    *,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    block_indices: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Select up to 512 history blocks per row, sorted, then append the current block.

    ``block_indices`` is int32 [T, 513] of logical block IDs padded with -1;
    ``counts`` is int32 [T] and includes the current block.
    """
    assert scores.ndim == 2 and scores.dtype == torch.float32
    assert scores.is_contiguous()
    rows, width = scores.shape
    assert block_indices.shape == (rows, 513) and block_indices.dtype == torch.int32
    assert counts.shape == (rows,) and counts.dtype == torch.int32
    selected_k = 0
    indices = block_indices
    if width > 512:
        import flashinfer

        # Numerical equality includes +0/-0; the library's radix key does not.
        # Keep the caller's scores intact, including during graph replay.
        ordered_scores = torch.empty_like(scores)
        size = scores.numel()
        _canonicalize_zero[(triton.cdiv(size, 1024),)](
            scores, ordered_scores, size, 1024, num_warps=4
        )
        _, indices = flashinfer.top_k(
            ordered_scores,
            512,
            sorted=False,
            deterministic=True,
            tie_break=1,
            dsa_graph_safe=True,
        )
        selected_k = 512

    _launch_block_pack(
        indices=indices,
        positions=positions,
        row_to_req=row_to_req,
        block_indices=block_indices,
        counts=counts,
        selected_k=selected_k,
    )


@triton.jit
def _pack_decode_blocks(
    Ids,
    Counts,
    Mapping,
    Requests,
    Positions,
    Packed,
    Lengths,
    MAP_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 1024)
    request = tl.load(Requests + row).to(tl.int64)
    count = tl.load(Counts + row)
    live = request > 0
    active = live & (lane < count) & (lane < 513)
    block_idx = tl.load(Ids + row * 513 + lane, active, other=0).to(tl.int64)
    logical_start = block_idx * 8
    physical_start = tl.load(
        Mapping + request * MAP_STRIDE + logical_start,
        active,
        other=0,
    ).to(tl.int64)
    value = ((physical_start // 8) << 32) | logical_start
    tl.store(Packed + row * WIDTH + lane, tl.where(active, value, -1), lane < WIDTH)
    position = tl.load(Positions + row).to(tl.int32)
    tl.store(Lengths + row, tl.where(live, position + 1, 0))


def step5_pack_decode_blocks(
    block_indices: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    packed_blocks: torch.Tensor,
    causal_lengths: torch.Tensor,
) -> None:
    """Map selected logical blocks to packed int64 attention addresses.

    Each output holds physical_start // 8 in the high word and logical_start in the
    low word; slots past ``counts`` are -1 and padding rows get causal length 0.
    """
    rows = block_indices.shape[0]
    assert block_indices.shape == (rows, 513) and block_indices.dtype == torch.int32
    assert packed_blocks.shape[0] == rows and packed_blocks.shape[1] in (513, 514, 516)
    assert packed_blocks.dtype == torch.int64 and packed_blocks.is_contiguous()
    assert req_to_token.stride(1) == 1
    _pack_decode_blocks[(rows,)](
        block_indices,
        counts,
        req_to_token,
        row_to_req,
        positions,
        packed_blocks,
        causal_lengths,
        req_to_token.stride(0),
        packed_blocks.shape[1],
        num_warps=4,
    )
