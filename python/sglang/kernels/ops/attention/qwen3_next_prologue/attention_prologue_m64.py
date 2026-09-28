"""Single-launch attention prologue with asymmetric M16 query shards."""

import torch
import triton
import triton.language as tl


@triton.jit
def _process_head(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    rope_ptr,
    locations_ptr,
    key_cache_ptr,
    value_cache_ptr,
    output_ptr,
    k_scale_ptr,
    v_scale_ptr,
    eps,
    token,
    head,
    shard,
    ROWS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_KEY: tl.constexpr,
    USE_SCALE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    LOCATION_STRIDE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    LAYOUT_EXCHANGE: tl.constexpr,
    DO_ROPE: tl.constexpr = True,
    STORE_KV: tl.constexpr = False,
):
    if WIDE_INDEX:
        token = token.to(tl.int64)
        if not IS_KEY:
            head = head.to(tl.int64)
    width: tl.constexpr = (QUERY_HEADS + 1) * 512
    source_base = token * width + head * 512
    local_offsets = tl.arange(0, BLOCK)
    offsets = shard * BLOCK + local_offsets

    # Recomputing RMS in each shard exposes independent waves at decode sizes.
    source = tl.load(packed_ptr + source_base + tl.arange(0, 256))
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    auxiliary = tl.load(packed_ptr + source_base + 256 + offsets)
    half: tl.constexpr = ROTARY_DIM // 2
    if DO_ROPE:
        position = tl.load(positions_ptr + token * POSITION_STRIDE)
        if WIDE_INDEX:
            position = position.to(tl.int64)
        else:
            position = position.to(tl.int32)
        frequency = offsets % half
        cosine = tl.load(rope_ptr + position * ROTARY_DIM + frequency).to(tl.float32)
        sine = tl.load(rope_ptr + position * ROTARY_DIM + half + frequency).to(
            tl.float32
        )
    if IS_KEY and STORE_KV:
        slot = tl.load(locations_ptr + token * LOCATION_STRIDE)
        if WIDE_INDEX:
            slot = slot.to(tl.int64)
        else:
            slot = slot.to(tl.int32)
        if USE_SCALE:
            key_scale = tl.load(k_scale_ptr).to(tl.float32)
            value_scale = tl.load(v_scale_ptr).to(tl.float32)

    # Four-element native BF16 dot groups accumulate the RMS sum in FP32.
    pairs = tl.reshape(source, (64, 1, 4))
    squares = tl.dot(pairs, tl.trans(pairs, 0, 2, 1))
    variance = tl.sum(tl.reshape(squares, (64,)), 0) / 256.0
    inv_rms = tl.rsqrt(variance + eps)
    owned = tl.load(packed_ptr + source_base + offsets).to(tl.float32)
    # The BF16 rounding boundary must precede the FP32 RoPE arithmetic.
    normalized = (owned * inv_rms * (1.0 + weight)).to(tl.bfloat16).to(tl.float32)
    if DO_ROPE:
        if LAYOUT_EXCHANGE:
            transposed = tl.trans(tl.reshape(normalized, (BLOCK // 64, 2, 32)), 0, 2, 1)
            left, right = tl.split(transposed)
            partner = tl.reshape(tl.trans(tl.join(right, left), 0, 2, 1), (BLOCK,))
        else:
            partner = tl.gather(normalized, local_offsets ^ half, 0)
        rotated = tl.where(
            offsets < half,
            normalized * cosine - partner * sine,
            normalized * cosine + partner * sine,
        )
        result = tl.where(offsets < ROTARY_DIM, rotated, normalized).to(tl.bfloat16)
    else:
        result = normalized.to(tl.bfloat16)
    if IS_KEY:
        output_head = ROWS * QUERY_HEADS + token
    else:
        output_head = token * QUERY_HEADS + head
    tl.store(output_ptr + output_head * 256 + offsets, result, cache_modifier=".wt")

    if IS_KEY:
        if STORE_KV:
            if USE_SCALE:
                # Keep FP32 division before FP8 rounding, including midpoint cases.
                cache_key = result.to(tl.float32) / key_scale
                cache_value = auxiliary.to(tl.float32) / value_scale
            else:
                cache_key = result
                cache_value = auxiliary
            converted = tl.join(cache_key, cache_value).to(
                key_cache_ptr.dtype.element_ty
            )
            cache_key, cache_value = tl.split(converted)
            tl.store(
                key_cache_ptr + slot * 256 + offsets, cache_key, cache_modifier=".wt"
            )
            tl.store(
                value_cache_ptr + slot * 256 + offsets,
                cache_value,
                cache_modifier=".wt",
            )
    else:
        gate_ptr = output_ptr + ROWS * (QUERY_HEADS + 1) * 256
        tl.store(
            gate_ptr + output_head * 256 + offsets, auxiliary, cache_modifier=".wt"
        )


@triton.jit
def _process_query_pairs(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    rope_ptr,
    output_ptr,
    eps,
    token,
    head,
    shard,
    ROWS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    if WIDE_INDEX:
        token = token.to(tl.int64)
        head = head.to(tl.int64)
    width: tl.constexpr = (QUERY_HEADS + 1) * 512
    base = token * width + head * 512
    half: tl.constexpr = ROTARY_DIM // 2
    lane = tl.arange(0, BLOCK // 2)
    offsets = shard * BLOCK + (lane // half) * ROTARY_DIM + lane % half
    source = tl.load(packed_ptr + base + tl.arange(0, 256))
    left_weight = tl.load(weight_ptr + offsets).to(tl.float32)
    right_weight = tl.load(weight_ptr + offsets + half).to(tl.float32)
    left_gate = tl.load(packed_ptr + base + 256 + offsets)
    right_gate = tl.load(packed_ptr + base + 256 + offsets + half)
    position = tl.load(positions_ptr + token * POSITION_STRIDE)
    if WIDE_INDEX:
        position = position.to(tl.int64)
    else:
        position = position.to(tl.int32)
    cosine = tl.load(rope_ptr + position * ROTARY_DIM + lane % half).to(tl.float32)
    sine = tl.load(rope_ptr + position * ROTARY_DIM + half + lane % half).to(tl.float32)
    pairs = tl.reshape(source, (64, 1, 4))
    squares = tl.dot(pairs, tl.trans(pairs, 0, 2, 1))
    variance = tl.sum(tl.reshape(squares, (64,)), 0) / 256.0
    inv_rms = tl.rsqrt(variance + eps)
    left = tl.load(packed_ptr + base + offsets).to(tl.float32)
    right = tl.load(packed_ptr + base + offsets + half).to(tl.float32)
    left_norm = (left * inv_rms * (1.0 + left_weight)).to(tl.bfloat16).to(tl.float32)
    right_norm = (right * inv_rms * (1.0 + right_weight)).to(tl.bfloat16).to(tl.float32)
    left_rope = left_norm * cosine - right_norm * sine
    right_rope = right_norm * cosine + left_norm * sine
    left_result = tl.where(offsets < ROTARY_DIM, left_rope, left_norm).to(tl.bfloat16)
    right_result = tl.where(offsets < ROTARY_DIM, right_rope, right_norm).to(
        tl.bfloat16
    )
    output_base = (token * QUERY_HEADS + head) * 256
    tl.store(output_ptr + output_base + offsets, left_result, cache_modifier=".wt")
    tl.store(
        output_ptr + output_base + offsets + half, right_result, cache_modifier=".wt"
    )
    gate_ptr = output_ptr + ROWS * (QUERY_HEADS + 1) * 256
    tl.store(gate_ptr + output_base + offsets, left_gate, cache_modifier=".wt")
    tl.store(gate_ptr + output_base + offsets + half, right_gate, cache_modifier=".wt")


@triton.jit
def _dispatch_heads(
    packed_ptr,
    positions_ptr,
    rope_ptr,
    q_weight_ptr,
    k_weight_ptr,
    output_ptr,
    eps,
    locations_ptr,
    k_scale_ptr,
    v_scale_ptr,
    key_cache_ptr,
    value_cache_ptr,
    ROWS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    USE_SCALE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    LOCATION_STRIDE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    STORE_KV: tl.constexpr,
):
    program = tl.program_id(0).to(tl.uint32)
    QUERY_SHARDS: tl.constexpr = 256 // QUERY_BLOCK
    # Issue the longer KV path first; every cache shard has a unique writer.
    if program < ROWS * 4:
        if ROWS > 32:
            token = program % ROWS
            shard = program // ROWS
        else:
            token = program // 4
            shard = program % 4
        _process_head(
            packed_ptr,
            k_weight_ptr,
            positions_ptr,
            rope_ptr,
            locations_ptr,
            key_cache_ptr,
            value_cache_ptr,
            output_ptr,
            k_scale_ptr,
            v_scale_ptr,
            eps,
            token,
            QUERY_HEADS,
            shard,
            ROWS,
            QUERY_HEADS,
            ROTARY_DIM,
            64,
            True,
            USE_SCALE,
            POSITION_STRIDE,
            LOCATION_STRIDE,
            WIDE_INDEX,
            ROWS != 16 and ROWS <= 32,
            True,
            STORE_KV,
        )
    else:
        if ROWS == 16:
            # 64 rotary + 64 plain + 128 plain channels: one RMS reduction
            # less per query head, and 256 programs for the four-head M16 case.
            query_program = program - ROWS * 4
            token = query_program % ROWS
            head = (query_program // ROWS) % QUERY_HEADS
            piece = query_program // (ROWS * QUERY_HEADS)
            if piece == 0:
                _process_query_pairs(
                    packed_ptr,
                    q_weight_ptr,
                    positions_ptr,
                    rope_ptr,
                    output_ptr,
                    eps,
                    token,
                    head,
                    0,
                    ROWS,
                    QUERY_HEADS,
                    ROTARY_DIM,
                    64,
                    POSITION_STRIDE,
                    WIDE_INDEX,
                )
            elif piece == 1:
                _process_head(
                    packed_ptr,
                    q_weight_ptr,
                    positions_ptr,
                    rope_ptr,
                    locations_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    output_ptr,
                    k_scale_ptr,
                    v_scale_ptr,
                    eps,
                    token,
                    head,
                    1,
                    ROWS,
                    QUERY_HEADS,
                    ROTARY_DIM,
                    64,
                    False,
                    USE_SCALE,
                    POSITION_STRIDE,
                    LOCATION_STRIDE,
                    WIDE_INDEX,
                    False,
                    False,
                    STORE_KV,
                )
            else:
                _process_head(
                    packed_ptr,
                    q_weight_ptr,
                    positions_ptr,
                    rope_ptr,
                    locations_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    output_ptr,
                    k_scale_ptr,
                    v_scale_ptr,
                    eps,
                    token,
                    head,
                    1,
                    ROWS,
                    QUERY_HEADS,
                    ROTARY_DIM,
                    128,
                    False,
                    USE_SCALE,
                    POSITION_STRIDE,
                    LOCATION_STRIDE,
                    WIDE_INDEX,
                    False,
                    False,
                    STORE_KV,
                )
        else:
            query_program = program - ROWS * 4
            if ROWS <= 32:
                shard = query_program % QUERY_SHARDS
                token = (query_program // QUERY_SHARDS) % ROWS
            else:
                token = query_program % ROWS
                shard = (query_program // ROWS) % QUERY_SHARDS
            head = query_program // (ROWS * QUERY_SHARDS)
            _process_head(
                packed_ptr,
                q_weight_ptr,
                positions_ptr,
                rope_ptr,
                locations_ptr,
                key_cache_ptr,
                value_cache_ptr,
                output_ptr,
                k_scale_ptr,
                v_scale_ptr,
                eps,
                token,
                head,
                shard,
                ROWS,
                QUERY_HEADS,
                ROTARY_DIM,
                QUERY_BLOCK,
                False,
                USE_SCALE,
                POSITION_STRIDE,
                LOCATION_STRIDE,
                WIDE_INDEX,
                ROWS <= 32,
                True,
                STORE_KV,
            )


@triton.jit
def _attention_prologue_small(
    packed_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    rope_ptr,
    output_ptr,
    eps,
    locations_ptr,
    k_scale_ptr,
    v_scale_ptr,
    key_cache_ptr,
    value_cache_ptr,
    ROWS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    USE_SCALE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    LOCATION_STRIDE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    STORE_KV: tl.constexpr,
):
    _dispatch_heads(
        packed_ptr,
        positions_ptr,
        rope_ptr,
        q_weight_ptr,
        k_weight_ptr,
        output_ptr,
        eps,
        locations_ptr,
        k_scale_ptr,
        v_scale_ptr,
        key_cache_ptr,
        value_cache_ptr,
        ROWS,
        QUERY_HEADS,
        ROTARY_DIM,
        QUERY_BLOCK,
        USE_SCALE,
        POSITION_STRIDE,
        LOCATION_STRIDE,
        WIDE_INDEX,
        STORE_KV,
    )


@triton.jit
def _attention_prologue_regular(
    packed_ptr,
    positions_ptr,
    rope_ptr,
    q_weight_ptr,
    k_weight_ptr,
    output_ptr,
    eps,
    locations_ptr,
    k_scale_ptr,
    v_scale_ptr,
    key_cache_ptr,
    value_cache_ptr,
    ROWS: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    USE_SCALE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    LOCATION_STRIDE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    STORE_KV: tl.constexpr,
):
    _dispatch_heads(
        packed_ptr,
        positions_ptr,
        rope_ptr,
        q_weight_ptr,
        k_weight_ptr,
        output_ptr,
        eps,
        locations_ptr,
        k_scale_ptr,
        v_scale_ptr,
        key_cache_ptr,
        value_cache_ptr,
        ROWS,
        QUERY_HEADS,
        ROTARY_DIM,
        QUERY_BLOCK,
        USE_SCALE,
        POSITION_STRIDE,
        LOCATION_STRIDE,
        WIDE_INDEX,
        STORE_KV,
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
    rows, width = projected_qkv_gate.shape
    query_heads = (width - 512) // 512
    # Q, K, and gate are contiguous, disjoint planes of a single allocation.
    output = torch.empty(
        (rows * (2 * query_heads + 1), 256),
        dtype=projected_qkv_gate.dtype,
        device=projected_qkv_gate.device,
    )
    query = output.as_strided((rows, query_heads, 256), (query_heads * 256, 256, 1))
    key = output.as_strided((rows, 1, 256), (256, 256, 1), rows * query_heads * 256)
    gate = output.as_strided(
        (rows, query_heads * 256),
        (query_heads * 256, 1),
        rows * (query_heads + 1) * 256,
    )
    # Returned V is unscaled; only the copy written into the cache is scaled.
    value = projected_qkv_gate.as_strided(
        (rows, 1, 256),
        (width, 256, 1),
        projected_qkv_gate.storage_offset() + query_heads * 512 + 256,
    )
    query_block = 64 if rows <= 32 else 128
    # Narrow relative indices are safe only when all reachable spans fit.
    spans = [
        projected_qkv_gate.numel() * projected_qkv_gate.element_size(),
        cos_sin_cache.numel() * cos_sin_cache.element_size(),
        ((rows - 1) * positions.stride(0) + 1) * positions.element_size(),
    ]
    if store_kv:
        spans.append(key_cache.numel() * key_cache.element_size())
        spans.append(
            ((rows - 1) * cache_locations.stride(0) + 1)
            * cache_locations.element_size()
        )
    wide_index = max(spans) >= 2**31
    # M16 benefits from early weight-pointer preloads; preserve the established
    # argument ordering for the other shapes.
    if rows == 16:
        kernel = _attention_prologue_small
        operands = (
            projected_qkv_gate,
            q_norm_weight,
            k_norm_weight,
            positions,
            cos_sin_cache,
            output,
            eps,
        )
    else:
        kernel = _attention_prologue_regular
        operands = (
            projected_qkv_gate,
            positions,
            cos_sin_cache,
            q_norm_weight,
            k_norm_weight,
            output,
            eps,
        )
    query_shards = 3 if rows == 16 else 256 // query_block
    kernel[(rows * (query_heads * query_shards + 4),)](
        *operands,
        cache_locations,
        projected_qkv_gate if k_scale is None else k_scale,
        projected_qkv_gate if v_scale is None else v_scale,
        key_cache,
        value_cache,
        ROWS=rows,
        QUERY_HEADS=query_heads,
        ROTARY_DIM=rotary_dim,
        QUERY_BLOCK=query_block,
        USE_SCALE=k_scale is not None,
        POSITION_STRIDE=positions.stride(0),
        LOCATION_STRIDE=cache_locations.stride(0),
        WIDE_INDEX=wide_index,
        STORE_KV=store_kv,
        num_warps=1,
    )
    return query, gate, key, value, key_cache, value_cache
