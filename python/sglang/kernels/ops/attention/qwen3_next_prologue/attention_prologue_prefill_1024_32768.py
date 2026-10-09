"""Fused 256-channel Q/K normalization, partial RoPE, and KV writes on CDNA4.

Query and KV CTAs have independent wave-local tiles and memory schedules.
Small query tiles share RoPE data and keep rotary partners in registers.
Per-role buffer I/O is enabled only when all reachable byte offsets fit int32.
BF16 rounding before/after RoPE and the FP32 cache-quotient boundary are preserved.
Medium-prefill tiles use independently cached, vectorized Q and KV I/O.
Large-prefill KV tiles use scalar pointer bases, and occupancy is shape-tuned.
Dense I/O, table indices, cache indices, and metadata strides have separate
address-width proofs, retaining narrow Q processing for oversized caches.
"""

from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store


class _Config(NamedTuple):
    q_block: int
    k_block: int
    warps: int
    q_vector: int
    k_vector: int
    q_lanes: int
    k_lanes: int
    group: int
    partitions: int
    key_first: bool
    io_policy: int
    store_cache: str
    q_prefetch: int
    k_prefetch: int
    q_aux_stage: int
    k_aux_stage: int
    output_layout: int
    output_gap: int


@gluon.jit
def _load(ptr, offset, mask, BUFFER: gl.constexpr, CACHE: gl.constexpr = ""):
    if BUFFER:
        return buffer_load(ptr, offset.to(gl.int32), mask, other=0, cache=CACHE)
    return gl.load(ptr + offset, mask, other=0, cache_modifier=CACHE)


@gluon.jit
def _load_index(
    ptr,
    offset,
    mask,
    BUFFER: gl.constexpr,
    NARROW: gl.constexpr,
    LOW_WORD: gl.constexpr,
):
    if LOW_WORD and NARROW and ptr.dtype.element_ty == gl.int64:
        # Valid indices fit uint32 when the complete table/cache byte span
        # fits int32. Read only the useful low word of int64 metadata.
        result = _load(ptr.to(gl.pointer_type(gl.uint32)), offset * 2, mask, BUFFER)
    else:
        result = _load(ptr, offset, mask, BUFFER)
    return result.to(gl.uint32 if NARROW else gl.int64)


@gluon.jit
def _load_scales(key_scale_ptr, value_scale_ptr, RECIPROCAL64: gl.constexpr):
    key_scale = gl.load(key_scale_ptr).to(gl.float32)
    value_scale = gl.load(value_scale_ptr).to(gl.float32)
    if RECIPROCAL64:
        return 1.0 / key_scale.to(gl.float64), 1.0 / value_scale.to(gl.float64)
    else:
        return key_scale, value_scale


@gluon.jit
def _scale_for_cache(value, scale, RECIPROCAL64: gl.constexpr):
    if RECIPROCAL64:
        # BF16 numerators and FP32 scales require this FP32 rounding before
        # cache conversion. A plain FP32 reciprocal can change FP8 midpoint ties.
        return (value.to(gl.float64) * scale).to(gl.float32)
    else:
        return value.to(gl.float32) / scale


@gluon.jit
def _store(ptr, offset, value, mask, BUFFER: gl.constexpr, CACHE: gl.constexpr):
    if BUFFER:
        buffer_store(
            value.to(ptr.dtype.element_ty), ptr, offset.to(gl.int32), mask, cache=CACHE
        )
    else:
        gl.store(ptr + offset, value, mask, cache_modifier=CACHE)


@gluon.jit
def _load_rope(
    ptr,
    position,
    channel,
    valid,
    ROTARY_DIM: gl.constexpr,
    BUFFER: gl.constexpr,
    JOINED: gl.constexpr,
):
    # KV tiles in selected buckets fetch the contiguous cosine/sine record
    # once, then distribute its halves with wave-local gathers.
    half: gl.constexpr = ROTARY_DIM // 2
    if JOINED:
        table = _load(
            ptr,
            position[:, None] * ROTARY_DIM + (channel[None, :] % ROTARY_DIM),
            valid[:, None],
            BUFFER,
        )
        block: gl.constexpr = table.type.shape[0]
        index = channel[None, :] % half + gl.full(
            (block, 1), 0, gl.int32, table.type.layout
        )
        cosine = gl.gather(table, index, axis=1).to(gl.float32)
        sine = gl.gather(table, index + half, axis=1).to(gl.float32)
    else:
        frequency = channel % half
        cosine = _load(
            ptr,
            position[:, None] * ROTARY_DIM + frequency[None, :],
            valid[:, None],
            BUFFER,
        ).to(gl.float32)
        sine = _load(
            ptr,
            position[:, None] * ROTARY_DIM + half + frequency[None, :],
            valid[:, None],
            BUFFER,
        ).to(gl.float32)
    return cosine, sine


@gluon.jit
def _rotate(
    normalized,
    channel,
    cosine,
    sine,
    ROTARY_DIM: gl.constexpr,
    COMPACT: gl.constexpr,
):
    layout: gl.constexpr = normalized.type.layout
    rank: gl.constexpr = len(normalized.type.shape)
    half: gl.constexpr = ROTARY_DIM // 2
    values = normalized.to(gl.float32)
    # Only the rotary prefix is consumed; XOR avoids a general indexed shuffle.
    if rank == 2:
        block: gl.constexpr = normalized.type.shape[0]
        channel_index = channel[None, :]
        partner_channel = channel ^ half
        partner_index = partner_channel[None, :] + gl.full(
            (block, 1), 0, gl.int32, layout
        )
    else:
        tokens: gl.constexpr = normalized.type.shape[0]
        heads: gl.constexpr = normalized.type.shape[1]
        channel_index = channel.expand_dims(0).expand_dims(0)
        partner_index = (channel_index ^ half) + gl.full(
            (tokens, heads, 1), 0, gl.int32, layout
        )
    if COMPACT:
        partner = gl.gather(normalized, partner_index, axis=rank - 1).to(gl.float32)
    else:
        partner = gl.gather(values, partner_index, axis=rank - 1)
    rotated = gl.where(
        channel_index < half,
        values * cosine - partner * sine,
        values * cosine + partner * sine,
    )
    if COMPACT:
        # The non-rotary tail is already BF16 and needs no round trip.
        return gl.where(channel_index < ROTARY_DIM, rotated.to(gl.bfloat16), normalized)
    return gl.where(channel_index < ROTARY_DIM, rotated, values).to(gl.bfloat16)


@gluon.jit
def _process_tile(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    locations_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    k_scale_ptr,
    v_scale_ptr,
    eps,
    first,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    LOC_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    USE_SCALE: gl.constexpr,
    IS_KEY: gl.constexpr,
    BLOCK: gl.constexpr,
    VECTOR: gl.constexpr,
    LANES: gl.constexpr,
    WARPS: gl.constexpr,
    FULL_TILES: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    PREFETCH: gl.constexpr,
    EARLY_AUX: gl.constexpr,
    STORE_KV: gl.constexpr,
):
    # Buffer-addressing bits separate packed reads, metadata/table reads,
    # returned-output stores, and scatter-cache stores.
    ROLE_BUFFER: gl.constexpr = (IO_POLICY >> 5) if IS_KEY else IO_POLICY
    PACKED_BUFFER: gl.constexpr = ROLE_BUFFER & 1
    META_BUFFER: gl.constexpr = ROLE_BUFFER & 2
    TABLE_BUFFER: gl.constexpr = ROLE_BUFFER & 4
    OUTPUT_BUFFER: gl.constexpr = ROLE_BUFFER & 8
    CACHE_BUFFER: gl.constexpr = ROLE_BUFFER & 16
    RECIPROCAL64: gl.constexpr = (IO_POLICY >> 10) & 1
    SOURCE_CACHE: gl.constexpr = (
        ".ca"
        if IS_KEY and ((IO_POLICY >> 18) & 1)
        else (".cg" if (IO_POLICY >> 11) & 1 else "")
    )
    LOW_WORD: gl.constexpr = (IO_POLICY >> 12) & 1
    COMPACT_ROPE: gl.constexpr = (IO_POLICY >> 13) & 1
    REBASE_KEY: gl.constexpr = IS_KEY and ((IO_POLICY >> 14) & 1)
    JOINED_TABLE: gl.constexpr = IS_KEY and ((IO_POLICY >> 16) & 1)
    POSITION_NARROW: gl.constexpr = NARROW and not ((IO_POLICY >> 21) & 1)
    SLOT_NARROW: gl.constexpr = NARROW and not ((IO_POLICY >> 22) & 1)
    # No reduction crosses a wave boundary, including half/quarter-wave heads.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VECTOR], [64 // LANES, LANES], [WARPS, 1], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    channel = gl.arange(0, 256, gl.SliceLayout(0, layout))
    item = first + gl.arange(0, BLOCK, row_layout)
    if IS_KEY:
        token = item
        head = gl.full((BLOCK,), HEADS, gl.int32, row_layout)
    else:
        token = item // HEADS
        head = item % HEADS
    if FULL_TILES:
        valid = gl.full((BLOCK,), True, gl.int1, row_layout)
    else:
        valid = token < ROWS
    if NARROW:
        token = token.to(gl.uint32)
    else:
        token = token.to(gl.int64)
    packed_offset = (
        token[:, None] * ((HEADS + 1) * 512) + head[:, None] * 512 + channel[None, :]
    )
    if REBASE_KEY:
        # Uniform token contributions belong in scalar 64-bit base pointers.
        # Scatter slots and RoPE positions keep their original runtime indices.
        packed_ptr += first.to(gl.int64) * ((HEADS + 1) * 512)
        packed_offset -= first * ((HEADS + 1) * 512)
        key_ptr += first.to(gl.int64) * KEY_STRIDE
        output_token = token - first
    else:
        output_token = token
    # Metadata storage strides and the indices stored there have independent
    # width proofs. A wide cache does not widen dense Q/K input/output offsets.
    position_offset = (
        token.to(gl.int64) if (IO_POLICY >> 19) & 1 else token
    ) * POS_STRIDE
    location_offset = (
        token.to(gl.int64) if (IO_POLICY >> 20) & 1 else token
    ) * LOC_STRIDE
    half: gl.constexpr = ROTARY_DIM // 2
    frequency = channel % half
    if PREFETCH & 17:
        position = _load_index(
            positions_ptr,
            position_offset,
            valid,
            META_BUFFER,
            POSITION_NARROW,
            LOW_WORD,
        )
    if PREFETCH & 1:
        if JOINED_TABLE:
            cosine, sine = _load_rope(
                cos_sin_ptr, position, channel, valid, ROTARY_DIM, TABLE_BUFFER, True
            )
        else:
            cosine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
            sine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + half + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
    if PREFETCH & 2 and (not IS_KEY or STORE_KV):
        companion = _load(
            packed_ptr, packed_offset + 256, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
        )
    if STORE_KV and IS_KEY and PREFETCH & 4:
        slot = _load_index(
            locations_ptr, location_offset, valid, META_BUFFER, SLOT_NARROW, LOW_WORD
        )
    if STORE_KV and IS_KEY and USE_SCALE and PREFETCH & 8:
        key_scale, value_scale = _load_scales(k_scale_ptr, v_scale_ptr, RECIPROCAL64)

    source = _load(
        packed_ptr, packed_offset, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
    ).to(gl.float32)
    weight = gl.load(weight_ptr + channel).to(gl.float32)
    if EARLY_AUX == 2:
        # Release the gate or V before forming normalized values.
        if (not IS_KEY or STORE_KV) and not PREFETCH & 2:
            companion = _load(
                packed_ptr,
                packed_offset + 256,
                valid[:, None],
                PACKED_BUFFER,
                SOURCE_CACHE,
            )
        if IS_KEY and STORE_KV:
            if not PREFETCH & 4:
                slot = _load_index(
                    locations_ptr,
                    location_offset,
                    valid,
                    META_BUFFER,
                    SLOT_NARROW,
                    LOW_WORD,
                )
            cache_offset = slot[:, None] * 256 + channel[None, :]
            if USE_SCALE:
                if not PREFETCH & 8:
                    key_scale, value_scale = _load_scales(
                        k_scale_ptr, v_scale_ptr, RECIPROCAL64
                    )
                cached_value = _scale_for_cache(companion, value_scale, RECIPROCAL64)
            else:
                cached_value = companion
            _store(
                value_cache_ptr,
                cache_offset,
                cached_value,
                valid[:, None],
                CACHE_BUFFER,
                CACHE,
            )
        elif not IS_KEY:
            output_offset = (
                output_token[:, None] * QUERY_STRIDE
                + head[:, None] * 256
                + channel[None, :]
            )
            _store(
                gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE
            )
    # The division expression and this BF16 boundary are part of the contract.
    variance = gl.sum(source * source, axis=1) / 256.0
    normalized = (
        source * gl.rsqrt(variance[:, None] + eps) * (1.0 + weight[None, :])
    ).to(gl.bfloat16)

    if not PREFETCH & 17:
        position = _load_index(
            positions_ptr,
            position_offset,
            valid,
            META_BUFFER,
            POSITION_NARROW,
            LOW_WORD,
        )
    if not PREFETCH & 1:
        if JOINED_TABLE:
            cosine, sine = _load_rope(
                cos_sin_ptr, position, channel, valid, ROTARY_DIM, TABLE_BUFFER, True
            )
        else:
            cosine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
            sine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + half + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
    if not PREFETCH & 2 and EARLY_AUX != 2 and (not IS_KEY or STORE_KV):
        companion = _load(
            packed_ptr, packed_offset + 256, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
        )

    if IS_KEY and STORE_KV:
        if not PREFETCH & 4 and EARLY_AUX != 2:
            slot = _load_index(
                locations_ptr,
                location_offset,
                valid,
                META_BUFFER,
                SLOT_NARROW,
                LOW_WORD,
            )
        cache_offset = slot[:, None] * 256 + channel[None, :]
        if EARLY_AUX != 2:
            if USE_SCALE:
                if not PREFETCH & 8:
                    key_scale, value_scale = _load_scales(
                        k_scale_ptr, v_scale_ptr, RECIPROCAL64
                    )
                cached_value = _scale_for_cache(companion, value_scale, RECIPROCAL64)
            else:
                cached_value = companion
    elif not IS_KEY:
        output_offset = (
            output_token[:, None] * QUERY_STRIDE
            + head[:, None] * 256
            + channel[None, :]
        )
        if EARLY_AUX == 1:
            _store(
                gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE
            )

    result = _rotate(normalized, channel, cosine, sine, ROTARY_DIM, COMPACT_ROPE)
    if IS_KEY:
        _store(
            key_ptr,
            output_token[:, None] * KEY_STRIDE + channel[None, :],
            result,
            valid[:, None],
            OUTPUT_BUFFER,
            CACHE,
        )
        if STORE_KV:
            if USE_SCALE:
                cached_key = _scale_for_cache(result, key_scale, RECIPROCAL64)
            else:
                cached_key = result
            _store(
                key_cache_ptr,
                cache_offset,
                cached_key,
                valid[:, None],
                CACHE_BUFFER,
                CACHE,
            )
            if not EARLY_AUX:
                _store(
                    value_cache_ptr,
                    cache_offset,
                    cached_value,
                    valid[:, None],
                    CACHE_BUFFER,
                    CACHE,
                )
    else:
        _store(query_ptr, output_offset, result, valid[:, None], OUTPUT_BUFFER, CACHE)
        if not EARLY_AUX:
            _store(
                gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE
            )


@gluon.jit
def _process_query_tokens(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_ptr,
    gate_ptr,
    eps,
    first,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    BLOCK: gl.constexpr,
    VECTOR: gl.constexpr,
    LANES: gl.constexpr,
    WARPS: gl.constexpr,
    FULL_TILES: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    EARLY_AUX: gl.constexpr,
):
    # Explicit [token, head, channel] ownership removes the head dimension
    # from RoPE loads, while every RMS reduction remains wave-local.
    tokens_per_tile: gl.constexpr = BLOCK // HEADS
    head_lanes: gl.constexpr = min(HEADS, 64 // LANES)
    head_waves: gl.constexpr = min(WARPS, HEADS // head_lanes)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, VECTOR],
        [64 // (head_lanes * LANES), head_lanes, LANES],
        [WARPS // head_waves, head_waves, 1],
        [2, 1, 0],
    )
    token_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    head_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    channel_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    table_layout: gl.constexpr = gl.SliceLayout(1, layout)
    token = first // HEADS + gl.arange(0, tokens_per_tile, token_layout)
    head = gl.arange(0, HEADS, head_layout)
    channel = gl.arange(0, 256, channel_layout)
    if FULL_TILES:
        valid = gl.full((tokens_per_tile,), True, gl.int1, token_layout)
    else:
        valid = token < ROWS
    token = token.to(gl.uint32 if NARROW else gl.int64)
    token3 = token.expand_dims(1).expand_dims(2)
    head3 = head.expand_dims(0).expand_dims(2)
    channel3 = channel.expand_dims(0).expand_dims(0)
    valid3 = valid.expand_dims(1).expand_dims(2)
    packed_offset = token3 * ((HEADS + 1) * 512) + head3 * 512 + channel3
    output_offset = token3 * QUERY_STRIDE + head3 * 256 + channel3
    PACKED_BUFFER: gl.constexpr = IO_POLICY & 1
    META_BUFFER: gl.constexpr = IO_POLICY & 2
    TABLE_BUFFER: gl.constexpr = IO_POLICY & 4
    OUTPUT_BUFFER: gl.constexpr = IO_POLICY & 8
    table_token = gl.convert_layout(token, gl.SliceLayout(1, table_layout))
    table_valid = gl.convert_layout(valid, gl.SliceLayout(1, table_layout))
    table_channel = gl.arange(0, 256, gl.SliceLayout(0, table_layout))
    position_offset = (
        table_token.to(gl.int64) if (IO_POLICY >> 19) & 1 else table_token
    ) * POS_STRIDE
    position = _load_index(
        positions_ptr,
        position_offset,
        table_valid,
        META_BUFFER,
        NARROW and not ((IO_POLICY >> 21) & 1),
        False,
    )
    cosine, sine = _load_rope(
        cos_sin_ptr,
        position,
        table_channel,
        table_valid,
        ROTARY_DIM,
        TABLE_BUFFER,
        False,
    )
    source = _load(packed_ptr, packed_offset, valid3, PACKED_BUFFER).to(gl.float32)
    weight = gl.load(weight_ptr + channel).to(gl.float32)
    if EARLY_AUX == 2:
        companion = _load(packed_ptr, packed_offset + 256, valid3, PACKED_BUFFER)
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)
    variance = gl.sum(source * source, axis=2) / 256.0
    normalized = (
        source
        * gl.rsqrt(variance.expand_dims(2) + eps)
        * (1.0 + weight.expand_dims(0).expand_dims(0))
    ).to(gl.bfloat16)
    if EARLY_AUX != 2:
        companion = _load(packed_ptr, packed_offset + 256, valid3, PACKED_BUFFER)
    if EARLY_AUX == 1:
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)
    result = _rotate(
        normalized,
        channel,
        cosine.expand_dims(1),
        sine.expand_dims(1),
        ROTARY_DIM,
        (IO_POLICY >> 13) & 1,
    )
    _store(query_ptr, output_offset, result, valid3, OUTPUT_BUFFER, CACHE)
    if not EARLY_AUX:
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)


@gluon.jit
def _fused_asymmetric(
    packed_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    locations_ptr,
    key_cache_ptr,
    value_cache_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    k_scale_ptr,
    v_scale_ptr,
    eps,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    LOC_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    USE_SCALE: gl.constexpr,
    Q_BLOCK: gl.constexpr,
    K_BLOCK: gl.constexpr,
    Q_VECTOR: gl.constexpr,
    K_VECTOR: gl.constexpr,
    Q_LANES: gl.constexpr,
    K_LANES: gl.constexpr,
    WARPS: gl.constexpr,
    GROUP: gl.constexpr,
    PARTITIONS: gl.constexpr,
    KEY_FIRST: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    Q_PREFETCH: gl.constexpr,
    K_PREFETCH: gl.constexpr,
    Q_EARLY: gl.constexpr,
    K_EARLY: gl.constexpr,
    STORE_KV: gl.constexpr,
):
    offset_type: gl.constexpr = gl.uint32 if NARROW else gl.int64
    pid = gl.program_id(0).to(offset_type)
    q_tiles: gl.constexpr = GROUP * HEADS // Q_BLOCK
    k_tiles: gl.constexpr = GROUP // K_BLOCK
    total: gl.constexpr = gl.cdiv(ROWS, GROUP) * (q_tiles + k_tiles)
    if PARTITIONS > 1:
        # Balanced partition transpose: the first total % PARTITIONS partitions
        # have one extra task. This stays bijective for arbitrary grid lengths.
        partition = pid % PARTITIONS
        pid = (
            partition * (total // PARTITIONS)
            + gl.minimum(partition, total % PARTITIONS)
            + pid // PARTITIONS
        )
    group = pid // (q_tiles + k_tiles)
    role = pid % (q_tiles + k_tiles)
    if KEY_FIRST:
        is_key = role < k_tiles
        k_first = group * GROUP + role * K_BLOCK
        q_first = group * GROUP * HEADS + (role - k_tiles) * Q_BLOCK
    else:
        is_key = role >= q_tiles
        k_first = group * GROUP + (role - q_tiles) * K_BLOCK
        q_first = group * GROUP * HEADS + role * Q_BLOCK
    full_tiles: gl.constexpr = ROWS % GROUP == 0
    # This branch is uniform across the entire CTA.
    if is_key:
        _process_tile(
            packed_ptr,
            k_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            locations_ptr,
            key_cache_ptr,
            value_cache_ptr,
            query_ptr,
            gate_ptr,
            key_ptr,
            k_scale_ptr,
            v_scale_ptr,
            eps,
            k_first,
            ROWS,
            HEADS,
            ROTARY_DIM,
            POS_STRIDE,
            LOC_STRIDE,
            QUERY_STRIDE,
            KEY_STRIDE,
            USE_SCALE,
            True,
            K_BLOCK,
            K_VECTOR,
            K_LANES,
            WARPS,
            full_tiles,
            IO_POLICY,
            NARROW,
            CACHE,
            K_PREFETCH,
            K_EARLY,
            STORE_KV,
        )
    else:
        if ((IO_POLICY >> 17) & 1) and Q_BLOCK >= HEADS:
            _process_query_tokens(
                packed_ptr,
                q_weight_ptr,
                positions_ptr,
                cos_sin_ptr,
                query_ptr,
                gate_ptr,
                eps,
                q_first,
                ROWS,
                HEADS,
                ROTARY_DIM,
                POS_STRIDE,
                QUERY_STRIDE,
                Q_BLOCK,
                Q_VECTOR,
                Q_LANES,
                WARPS,
                full_tiles,
                IO_POLICY,
                NARROW,
                CACHE,
                Q_EARLY,
            )
        else:
            _process_tile(
                packed_ptr,
                q_weight_ptr,
                positions_ptr,
                cos_sin_ptr,
                locations_ptr,
                key_cache_ptr,
                value_cache_ptr,
                query_ptr,
                gate_ptr,
                key_ptr,
                k_scale_ptr,
                v_scale_ptr,
                eps,
                q_first,
                ROWS,
                HEADS,
                ROTARY_DIM,
                POS_STRIDE,
                LOC_STRIDE,
                QUERY_STRIDE,
                KEY_STRIDE,
                USE_SCALE,
                False,
                Q_BLOCK,
                Q_VECTOR,
                Q_LANES,
                WARPS,
                full_tiles,
                IO_POLICY,
                NARROW,
                CACHE,
                Q_PREFETCH,
                Q_EARLY,
                STORE_KV,
            )


def _small_span(tensor: torch.Tensor) -> bool:
    """Prove signed 32-bit byte addressing, including holes in strided views."""
    return (
        all(s >= 0 for s in tensor.stride())
        and (sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1)
        * tensor.element_size()
        < 2**31
    )


def _config(rows):
    """Shape-only launch policy; Q blocks count heads, K blocks count tokens."""
    # I/O policy packs the Q mask in bits 0:5 and KV mask in bits 5:10.
    # Mask bits: packed=1, metadata=2, RoPE table=4, outputs=8, caches=16.
    # Bit 10 selects FP64 reciprocal products with explicit FP32 rounding.
    # Bit 11: source .cg loads; bit 12: bounded low-word metadata loads;
    # bit 13: BF16 rotary shuffles and retention of the BF16 non-rotary tail.
    # Bit 14: scalar rebasing of KV packed reads and returned-K stores.
    # Bit 16: joined cosine/sine table loads for KV.
    # Bit 17: token-tiled Q with head-shared RoPE data.
    # Bit 18: cache-all KV packed loads, independently of the query policy.
    # Bits 19/20: wide position/location metadata storage offsets.
    # Bits 21/22: wide RoPE-table/scatter-cache indices, independent of core I/O.
    # Prefetch bits: RoPE=1, gate/V=2, cache slot=4, cache scales=8,
    # position-only=16 (defer the table values until after normalization).
    # Auxiliary stores: 0=last, 1=before RoPE, 2=before the RMS reduction.
    if rows <= 1024:
        # Fetch KV RoPE early; defer V, scatter slots, and scales until needed.
        return _Config(
            8,
            2,
            2,
            2,
            2,
            16,
            64,
            128,
            4,
            False,
            0x2039C,
            ".wt",
            1,
            1,
            1,
            0,
            1,
            256,
        )
    if rows <= 2048:
        # Early gate stores shorten the live ranges of the eight-head Q tile.
        return _Config(
            8,
            2,
            1,
            8,
            2,
            8,
            32,
            64,
            8,
            True,
            0x13400,
            ".cs",
            1,
            3,
            2,
            0,
            0,
            4096,
        )
    if rows <= 4096:
        return _Config(
            4,
            2,
            1,
            8,
            2,
            16,
            32,
            128,
            8,
            False,
            0x1400,
            ".cs",
            0,
            12,
            1,
            0,
            0,
            512,
        )
    if rows <= 8192:
        return _Config(
            8,
            8,
            4,
            2,
            8,
            32,
            32,
            1024,
            8,
            True,
            0x41FFF,
            ".wt",
            16,
            10,
            2,
            2,
            0,
            0,
        )
    if rows <= 12288:
        return _Config(
            8,
            8,
            4,
            2,
            2,
            32,
            64,
            1024,
            8,
            True,
            0x3F3C,
            ".wt",
            3,
            0,
            2,
            0,
            1,
            128,
        )
    if rows <= 16384:
        return _Config(
            2,
            1,
            1,
            2,
            4,
            32,
            64,
            4096,
            8,
            True,
            0x3FF,
            ".cs",
            16,
            15,
            0,
            0,
            1,
            0,
        )
    if rows <= 24576:
        return _Config(
            2,
            1,
            1,
            2,
            2,
            32,
            64,
            1024,
            8,
            True,
            0,
            ".cs",
            0,
            0,
            0,
            2,
            0,
            0,
        )
    # Large groups plus compact BF16 rotary exchanges favor streaming prefill.
    return _Config(
        2,
        1,
        1,
        2,
        4,
        32,
        64,
        8192,
        8,
        True,
        0x6000,
        ".cs",
        1,
        15,
        0,
        0,
        1,
        0,
    )


def _allocate_outputs(packed, output_layout, gap):
    """Allocate planar or token-major Q/gate/K views of one backing buffer."""
    rows, width = packed.shape
    heads = (width - 512) // 512
    if output_layout == 0:
        q_size = rows * heads * 256
        backing = packed.new_empty((q_size * 2 + rows * 256 + 2 * gap,))
        query = backing[:q_size].view(rows, heads, 256)
        gate = backing[q_size + gap : 2 * q_size + gap].view(rows, heads * 256)
        key = backing[2 * q_size + 2 * gap :].view(rows, 1, 256)
    else:
        backing = packed.new_empty((rows, (2 * heads + 1) * 256 + gap))
        query = backing[:, : heads * 256].view(rows, heads, 256)
        gate = backing[:, heads * 256 : 2 * heads * 256]
        key = backing[:, 2 * heads * 256 : (2 * heads + 1) * 256].view(rows, 1, 256)
    # V is unchanged. Retaining its input-backed view avoids another full copy.
    value = packed[:, heads * 512 + 256 :].view(rows, 1, 256)
    return query, gate, key, value


def _launch(
    packed,
    q_weight,
    k_weight,
    positions,
    cos_sin,
    locations,
    key_cache,
    value_cache,
    outputs,
    eps,
    rotary_dim,
    k_scale,
    v_scale,
    config,
    store_kv,
):
    rows, width = packed.shape
    heads = (width - 512) // 512
    (
        qb,
        kb,
        warps,
        qv,
        kv,
        ql,
        kl,
        group,
        partitions,
        key_first,
        io_policy,
        cache,
        qp,
        kp,
        qe,
        ke,
        _output_layout,
        _gap,
    ) = config
    query, gate, key, _ = outputs
    packed_small = _small_span(packed)
    output_small = all(_small_span(t) for t in (query, gate, key))
    positions_small = _small_span(positions)
    locations_small = _small_span(locations)
    table_small = _small_span(cos_sin)
    cache_small = _small_span(key_cache) and _small_span(value_cache)
    narrow = packed_small and output_small
    # Keep each proven-safe buffer path. Use real int64 arithmetic only for
    # address spaces whose complete reachable byte span exceeds signed int32.
    if not packed_small:
        io_policy &= ~0x21
    if not output_small:
        io_policy &= ~0x108
    if not positions_small:
        io_policy = (io_policy & ~0x42) | (1 << 19)
    if not locations_small:
        io_policy = (io_policy & ~0x40) | (1 << 20)
    if not table_small:
        io_policy = (io_policy & ~0x84) | (1 << 21)
    if not cache_small:
        io_policy = (io_policy & ~0x200) | (1 << 22)
    if rows <= 1024:
        waves_per_eu = 8
    elif rows <= 2048:
        waves_per_eu = 4
    elif 12288 < rows <= 16384 or rows > 24576:
        waves_per_eu = 6
    else:
        waves_per_eu = 0
    programs = triton.cdiv(rows, group) * (group * heads // qb + group // kb)
    return _fused_asymmetric[(programs,)](
        packed,
        q_weight,
        k_weight,
        positions,
        cos_sin,
        locations,
        key_cache,
        value_cache,
        query,
        gate,
        key,
        packed if k_scale is None else k_scale,
        packed if v_scale is None else v_scale,
        eps,
        rows,
        heads,
        rotary_dim,
        positions.stride(0),
        locations.stride(0),
        query.stride(0),
        key.stride(0),
        k_scale is not None,
        qb,
        kb,
        qv,
        kv,
        ql,
        kl,
        warps,
        group,
        partitions,
        key_first,
        io_policy,
        narrow,
        cache,
        qp,
        kp,
        qe,
        ke,
        store_kv,
        num_warps=warps,
        waves_per_eu=waves_per_eu,
    )


def fused_attention_qk_norm_rope_kv_cache(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_locations: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    rotary_dim: int = 64,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    store_kv: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return Q, gate, K, V. Cache updates run only when store_kv is set."""
    config = _config(projected_qkv_gate.shape[0])
    outputs = _allocate_outputs(
        projected_qkv_gate, config.output_layout, config.output_gap
    )
    _launch(
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
        cache_locations,
        key_cache,
        value_cache,
        outputs,
        eps,
        rotary_dim,
        k_scale,
        v_scale,
        config,
        store_kv,
    )
    return (*outputs, key_cache, value_cache)
