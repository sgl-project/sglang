# SPDX-License-Identifier: Apache-2.0
"""Step5 index normalization, compression, scoring and block selection."""

import dataclasses
import math

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


def _check_tensor(value, *, shape, dtype, device, name, contiguous=True):
    if (
        value.layout != torch.strided
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        raise ValueError(f"Step5 indexer {name} requires {shape}/{dtype} on {device}")


def _overlaps(a, b):
    # Check addressed intervals, allowing disjoint views of a caller workspace.
    if not a.numel() or not b.numel():
        return False

    def interval(t):
        end = sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1
        return t.data_ptr(), t.data_ptr() + end * t.element_size()

    a0, a1 = interval(a)
    b0, b1 = interval(b)
    return max(a0, b0) < min(a1, b1)


def step5_indexer_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    eps: float,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K then rotate channels 0..31; outputs are caller-owned.

    Q[T,H,256] and K[T,1,256] may have padded row strides; other tensors must
    be contiguous. H is 2 or 4. Parameters/cache are BF16, statistics are FP32,
    normalized Q/K round to BF16 before FP32 rotation and BF16 output.
    Positions[T] must lie in [0, cache.shape[0]); the serving owner validates
    this bound. Inputs must be finite and output storage disjoint from inputs.
    Warm up before Graph capture. No allocation or host tensor-value reads.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 indexer requires NVIDIA CUDA")
    if q.ndim != 3 or q.shape[1] not in (2, 4) or q.shape[2] != 256:
        raise ValueError("Step5 index Q requires [tokens, 2 or 4, 256]")
    if not math.isfinite(eps) or eps <= 0 or not isinstance(zero_centered, bool):
        raise ValueError(
            "Step5 indexer requires positive finite eps and bool zero_centered"
        )
    tokens, heads, _ = q.shape
    for name, value, shape in (
        ("q", q, (tokens, heads, 256)),
        ("k", k, (tokens, 1, 256)),
    ):
        _check_tensor(
            value,
            shape=shape,
            dtype=torch.bfloat16,
            device=q.device,
            name=name,
            contiguous=False,
        )
        if (
            value.stride(2) != 1
            or value.stride(1) != 256
            or value.stride(0) < shape[1] * 256
        ):
            raise ValueError(
                "Step5 indexer requires contiguous heads and nonoverlapping rows"
            )
    for name, value in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("k_bias", k_bias),
    ):
        _check_tensor(
            value, shape=(256,), dtype=torch.bfloat16, device=q.device, name=name
        )
    if cache.ndim != 2 or cache.shape[1] != 32 or cache.shape[0] < 1:
        raise ValueError("Step5 indexer cache requires [positions, 32]")
    _check_tensor(
        cache,
        shape=tuple(cache.shape),
        dtype=torch.bfloat16,
        device=q.device,
        name="cache",
    )
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Step5 indexer positions must be int32 or int64")
    _check_tensor(
        positions,
        shape=(tokens,),
        dtype=positions.dtype,
        device=q.device,
        name="positions",
    )
    for name, out, shape in (
        ("q_out", q_out, tuple(q.shape)),
        ("k_out", k_out, tuple(k.shape)),
    ):
        _check_tensor(
            out, shape=shape, dtype=torch.bfloat16, device=q.device, name=name
        )
        if any(
            _overlaps(out, x)
            for x in (q, k, q_weight, k_weight, k_bias, cache, positions)
        ):
            raise ValueError("Step5 indexer outputs must not alias inputs")
    if _overlaps(q_out, k_out):
        raise ValueError("Step5 indexer outputs must not alias each other")
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
            int(zero_centered),
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
    NativePositions,
    SeqLens,
    OutLoc,
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
    QueryCount,
    NativeValid,
    ROWS: tl.constexpr,
    CAP: tl.constexpr,
    TOKENS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    MAP_COL_STRIDE: tl.constexpr,
    PHYSICAL_LIMIT: tl.constexpr,
    CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    NATIVE: tl.constexpr,
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
    # Validate each completed R8 group once, rather than loading its eight
    # physical mappings for every query. Current-token mappings above remain
    # per query: native FULL pages need not be physically adjacent.
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
        tl.store(QueryCount, live * WIDTH)

    # Store destinations directly in [completed-block, request] order.
    # A trailing, incomplete block is never consumed; match the last query's
    # mapping there so every output cell is deterministic across graph replays.
    tl.store(
        Writes + block_idx * ROWS + row,
        (first - (first % 8 + 8) % 8) // 8,
        block_idx < BLOCKS,
    )

    query_pos = safe_pos
    tl.store(QueryPositions + row * WIDTH + depth, query_pos, depth < WIDTH)
    tl.store(QuerySlots + row * WIDTH + depth, tl.where(active, raw, 0), depth < WIDTH)
    native_ok = True
    if NATIVE:
        native_pos = tl.load(
            NativePositions + row * WIDTH + depth, depth < WIDTH, other=0
        )
        out = tl.load(OutLoc + row * WIDTH + depth, depth < WIDTH, other=0)
        seq = tl.load(SeqLens + row).to(tl.int64)
        length_ok = (base == seq - 1) & (base < CONTEXT)
        native_ok = ~active | (
            length_ok
            & tl.min(
                tl.where(
                    depth < WIDTH, (native_pos == query_pos) & (out == current), True
                ).to(tl.int32),
                0,
            )
        )
    tl.store(NativeValid + row, native_ok)


def build_compression_metadata(
    *,
    request_slots,
    positions,
    valid_rows,
    req_to_token,
    compressed_index_key_pages,
    native_positions=None,
    seq_lens=None,
    out_cache_loc=None,
    context_len=None,
):
    """Build decode request admission, query layout and final block write addresses.

    ``positions`` names current-token positions. Native input validation is
    folded into mapping loads when the backend supplies positions/lengths/write
    locations. No CPU reads of dynamic values or mapping dtype conversions occur here.
    """
    token_width = 1
    device = req_to_token.device
    rows = request_slots.numel()
    if (
        device.type != "cuda"
        or req_to_token.ndim != 2
        or req_to_token.dtype not in (torch.int32, torch.int64)
        or req_to_token.shape[0] < 1
        or req_to_token.shape[1] < 8
        or compressed_index_key_pages < 2
    ):
        raise ValueError("CSA requires native CUDA request/FULL mapping")
    native = native_positions is not None
    values = [(request_slots, (rows,)), (positions, (rows,)), (valid_rows, (1,))]
    if native:
        values += [
            (native_positions, (rows * token_width,)),
            (seq_lens, (rows,)),
            (out_cache_loc, (rows * token_width,)),
        ]
    for value, shape in values:
        if (
            value is None
            or value.device != device
            or value.shape != shape
            or value.dtype not in (torch.int32, torch.int64)
            or not value.is_contiguous()
        ):
            raise ValueError("CSA requires contiguous device integer metadata")

    def alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    num_blocks = (7 + token_width) // 8
    result = dict(
        request_slots=alloc(rows, torch.int64),
        input_start_positions=alloc(rows, torch.int64),
        input_start_rows=alloc(rows, torch.int64),
        input_lengths=alloc(rows, torch.int64),
        request_active=alloc(rows, torch.bool),
        request_mapping_valid=alloc(rows, torch.bool),
        request_count_valid=alloc(1, torch.bool),
        compressed_key_write_indices=alloc((num_blocks, rows), torch.int64),
        num_input_tokens=rows * token_width,
    )
    query_slots = alloc(rows * token_width, torch.int32)
    query_positions = alloc(rows * token_width, torch.int32)
    query_count = alloc(1, torch.int32)
    native_valid = alloc(rows, torch.bool)
    uses = torch.zeros(req_to_token.shape[0], dtype=torch.int32, device=device)
    if rows:
        _count_request_owners[(triton.cdiv(rows, 128),)](
            request_slots, valid_rows, uses, rows, req_to_token.shape[0], 128
        )
        _build_request_metadata[(rows,)](
            request_slots,
            positions,
            valid_rows,
            req_to_token,
            uses,
            native_positions if native else positions,
            seq_lens if native else positions,
            out_cache_loc if native else positions,
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
            query_count,
            native_valid,
            rows,
            req_to_token.shape[0],
            req_to_token.shape[1],
            *req_to_token.stride(),
            compressed_index_key_pages * 64,
            context_len or req_to_token.shape[1],
            token_width,
            native,
            triton.next_power_of_2(token_width),
            triton.next_power_of_2(num_blocks),
            num_blocks,
            num_warps=4,
        )
    else:
        result["request_count_valid"].copy_((valid_rows == 0))
        query_count.zero_()
    return result, query_slots, query_positions, query_count, native_valid


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
        # Exactly one CTA per request: admission and compression share its
        # register result. Multi-block prefill admits and writes in
        # separate launches instead of racing/repeating them per block.
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


@dataclasses.dataclass(frozen=True)
class AppendRingPlan:
    """Execution plan for one batched append: one row per request span."""

    slots: torch.Tensor
    bases: torch.Tensor
    sources: torch.Tensor
    lengths: torch.Tensor
    active: torch.Tensor
    admitted: torch.Tensor
    count_valid: torch.Tensor
    physical_blocks: torch.Tensor
    max_blocks: int
    enabled: torch.Tensor
    success: torch.Tensor


def append_ring(
    *,
    plan: AppendRingPlan,
    index_key,
    compression_logits,
    index_key_ring,
    compression_logits_ring,
    positions,
    compressed_index_keys,
    assert_success=False,
):
    slots, bases, sources, lengths = (
        plan.slots,
        plan.bases,
        plan.sources,
        plan.lengths,
    )
    active, admitted, count_valid = plan.active, plan.admitted, plan.count_valid
    physical_blocks, max_blocks = plan.physical_blocks, plan.max_blocks
    enabled, success = plan.enabled, plan.success
    """Read history/current inputs before overwriting ring cells.

    The native page64 prefix cache resumes fresh requests at an R8 boundary;
    later chunks/verify read only [floor(base/8)*8, base). With C >= 7 + W,
    a candidate at p <= base + W - 1 overwrites p-C < floor(base/8)*8.
    Thus any legal accepted prefix retains every historical input it can need.
    Slot reuse is ordered by the scheduler's forward stream / WAR barrier;
    position tags diagnose missing inputs, not a second request identity.
    """
    rows = slots.numel()
    capacity = index_key_ring.shape[1]
    if rows:
        if max_blocks != 1:
            _admit[(rows,)](
                slots,
                bases,
                active,
                admitted,
                count_valid,
                positions,
                enabled,
                success,
                capacity,
                assert_success,
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
                positions,
                success,
                max_blocks == 1,
                assert_success,
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
            positions,
            capacity,
            triton.next_power_of_2(capacity),
            compression_logits.stride(0),
            num_warps=8,
        )
    return success


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
    CLEAR_FUTURE: tl.constexpr,
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
    # A 1M graph capture has 4096 capacity tiles while a 60K request has about
    # 235 live tiles. The SM10x selector consumes positions directly, so its
    # path can stop at live history. The portable FlashInfer fallback still
    # overwrites future tiles to make a full-width top-k replay safe.
    last_tile = tl.cdiv(BLOCKS, BLOCK_R)
    if not CLEAR_FUTURE:
        live_blocks = tl.maximum(position // 8, 0)
        last_tile = tl.minimum(tl.cdiv(live_blocks, BLOCK_R), last_tile)
        # Selecting every historical block does not consume indexer_logits.
        last_tile = tl.where(live_blocks <= 512, 0, last_tile)
        last_tile = tl.where(request > 0, last_tile, 0)
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


def _launch_triton_scores(
    q,
    weights,
    compressed_index_keys,
    req_to_token,
    row_to_req,
    positions,
    out,
    *,
    clear_future,
    weight_scale=1.0,
):
    # D256 gathers already consume substantial registers. A 32-block tile
    # reduces that pressure and gives small batches more independent CTAs.
    blocks_per_tile = 32
    grid_tiles = min(triton.cdiv(out.shape[1], blocks_per_tile), 128)
    _score_blocks[(q.shape[0], grid_tiles)](
        q,
        weights,
        compressed_index_keys,
        req_to_token,
        row_to_req,
        positions,
        out,
        q.shape[1],
        req_to_token.stride(0),
        out.shape[1],
        blocks_per_tile,
        grid_tiles,
        clear_future,
        weight_scale,
        num_warps=4,
    )


def _check(value, *, shape, dtype, device, name):
    if (
        value.layout != torch.strided
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"Step5 scoring {name} requires contiguous {shape}/{dtype} on {device}"
        )


def step5_indexer_decode_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    clear_future: bool = True,
) -> torch.Tensor:
    """Score completed historical blocks for flattened ragged query rows.

    Q is BF16 or prequantized unscaled E4M3. FP32 weights are already scaled by the
    index head count (``weight_scale=1``). BF16 weights are the unscaled GEMM
    output and use ``weight_scale=0.5`` (exact: ``4**-0.5``). CompressedIndexKeys is
    the existing SW128 byte backing.
    Request slot zero is padding. Valid slots/positions and compressed_index_keys
    publication are the backend's responsibility. Each valid req_to_token entry
    at logical offset 8*r must be an aligned physical start of a complete block.
    Native 64-token pages retain their contiguous logical offsets.
    Output width must cover all positions//8, including Graph replay inputs.
    With ``clear_future=True``, invalid/future output columns are always -inf.
    The SM10x live-length selector may set it false because it never reads those
    columns. Inputs are immutable; output must not share storage with them.
    Warm up before Graph capture.
    """
    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 scoring requires NVIDIA CUDA")
    if type(clear_future) is not bool:
        raise ValueError("Step5 scoring clear_future must be a bool")
    if q.ndim != 3 or q.shape[1] not in (2, 4) or q.shape[2] != 256:
        raise ValueError("Step5 scoring Q requires [tokens, 2 or 4, 256]")
    tokens, heads, _ = q.shape
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("Step5 scoring Q must be BF16 or E4M3")
    _check(q, shape=tuple(q.shape), dtype=q.dtype, device=q.device, name="q")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Step5 scoring weights must be BF16 or FP32")
    weight_scale = 0.5 if weights.dtype == torch.bfloat16 else 1.0
    _check(
        weights,
        shape=(tokens, heads),
        dtype=weights.dtype,
        device=q.device,
        name="weights",
    )
    if (
        compressed_index_keys.ndim != 4
        or compressed_index_keys.shape[1:] != (8, 1, 256)
        or compressed_index_keys.shape[0] < 1
    ):
        raise ValueError(
            "Step5 compressed_index_keys requires SW128 [pages, 8, 1, 256]"
        )
    _check(
        compressed_index_keys,
        shape=tuple(compressed_index_keys.shape),
        dtype=torch.uint8,
        device=q.device,
        name="compressed_index_keys",
    )
    if req_to_token.ndim != 2 or req_to_token.shape[0] < 1:
        raise ValueError("Step5 request map requires [request slots, token capacity]")
    for name, value, shape in (
        ("req_to_token", req_to_token, tuple(req_to_token.shape)),
        ("row_to_req", row_to_req, (tokens,)),
        ("positions", positions, (tokens,)),
    ):
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Step5 scoring {name} must be int32 or int64")
        _check(value, shape=shape, dtype=value.dtype, device=q.device, name=name)
    if (
        out.ndim != 2
        or not 1 <= out.shape[1] < 2**31
        or out.shape[1] > (req_to_token.shape[1] // 8 + 255) // 256 * 256
    ):
        raise ValueError(
            "Step5 score width must fit native request capacity and int32 indexing"
        )
    _check(
        out,
        shape=(tokens, out.shape[1]),
        dtype=torch.float32,
        device=q.device,
        name="out",
    )
    if out.numel():
        a, b = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for value in (
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
        ):
            lo = value.data_ptr()
            if max(a, lo) < min(b, lo + value.numel() * value.element_size()):
                raise ValueError("Step5 scoring output must not alias inputs")
    if tokens:
        _launch_triton_scores(
            q.to(torch.float8_e4m3fn),
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            clear_future=clear_future,
            weight_scale=weight_scale,
        )
    return out


# Block top-k and packed attention addresses.


@triton.jit
def _canonicalize_zero(Scores, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(Scores + offset, offset < N, other=0.0).to(tl.uint32, bitcast=True)
    # FlashInfer's radix ordering distinguishes signed zero. Preserve all other
    # bits, including subnormals, so numerical ties prefer the smaller block ID.
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    tl.store(Out + offset, bits.to(tl.float32, bitcast=True), offset < N)


def canonicalize_zero(scores, out):
    size = scores.numel()
    _canonicalize_zero[(triton.cdiv(size, 1024),)](scores, out, size, 1024, num_warps=4)


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
    indices,
    positions,
    requests,
    out,
    counts,
    *,
    selected_k,
    req_to_token=None,
    packed=None,
    causal_lengths=None,
):
    _pack_blocks[(out.shape[0],)](
        indices,
        positions,
        requests,
        out,
        counts,
        req_to_token,
        packed,
        causal_lengths,
        selected_k,
        req_to_token.stride(0) if packed is not None else 0,
        packed.shape[1] if packed is not None else 0,
        num_warps=1 if packed is not None and out.shape[0] >= 512 else 4,
    )


STEP5_HISTORY_BLOCKS = 512
# Preserve the score workspace layout while sharing the public top-k kernel.
_SCORE_STRIDE_ALIGNMENT = 256


@triton.jit
def _prepare_history_lengths(Positions, Requests, Lengths, ROWS: tl.constexpr):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    position = tl.load(Positions + row, row < ROWS, other=0)
    request = tl.load(Requests + row, row < ROWS, other=0)
    tl.store(Lengths + row, tl.where(request > 0, position // 8, 0), row < ROWS)


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
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    packed: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor,
    causal_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    topk_ids: torch.Tensor,
    block_indices: torch.Tensor,
    history_ready: bool = False,
) -> None:
    if not history_ready:
        _prepare_history_lengths[(triton.cdiv(positions.numel(), 128),)](
            positions, row_to_req, history_lengths, positions.numel(), num_warps=4
        )
    topk(
        scores,
        STEP5_HISTORY_BLOCKS,
        end=history_lengths,
        output_idx=topk_ids,
        idx_oob_fill_value=-1,
    )
    _launch_block_pack(
        topk_ids,
        positions,
        row_to_req,
        block_indices,
        counts,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select <=512 historical blocks, sort IDs, then append current exactly once.

    Inputs follow step5_indexer_decode_logits: finite history and -inf future scores.
    Width covers positions//8. Slot zero is padding. Output IDs are logical,
    ascending, int32 [T,513], padded with -1; counts are int32 [T]. Caller must
    keep outputs disjoint from inputs and each other. Warm FlashInfer outside
    capture and use its shared device scratch on the sequential forward stream.
    """
    if scores.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 selection requires NVIDIA CUDA")
    if (
        scores.ndim != 2
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or not 1 <= scores.shape[1] < 2**31
    ):
        raise ValueError(
            "Step5 selection requires contiguous FP32 scores [T,R], 1<=R<2**31"
        )
    rows, width = scores.shape
    for name, value, shape in (
        ("positions", positions, (rows,)),
        ("row_to_req", row_to_req, (rows,)),
        ("block_indices", block_indices, (rows, 513)),
        ("counts", counts, (rows,)),
    ):
        dtype_ok = (
            value.dtype in (torch.int32, torch.int64)
            if name in ("positions", "row_to_req")
            else value.dtype == torch.int32
        )
        if (
            value.device != scores.device
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or not dtype_ok
        ):
            raise ValueError(f"Step5 selection invalid {name} metadata")
    if rows == 0:
        return block_indices, counts
    for out, others in (
        (block_indices, (scores, positions, row_to_req, counts)),
        (counts, (scores, positions, row_to_req)),
    ):
        a = out.data_ptr()
        b = a + out.numel() * out.element_size()
        for value in others:
            lo = value.data_ptr()
            if max(a, lo) < min(b, lo + value.numel() * value.element_size()):
                raise ValueError(
                    "Step5 selection outputs must not alias inputs or each other"
                )
    selected_k = 0
    indices = block_indices
    if width > 512:
        import flashinfer

        # Numerical equality includes +0/-0; the library's radix key does not.
        # Keep the caller's scores intact, including during graph replay.
        ordered_scores = torch.empty_like(scores)
        canonicalize_zero(scores, ordered_scores)
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
        indices, positions, row_to_req, block_indices, counts, selected_k=selected_k
    )
    return block_indices, counts


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
    """Pack physical block and logical token start without copying raw K/V.

    Selection supplies sorted int32 IDs [T,513] and counts [T], including current.
    Request slot zero is padding and must have count zero. The native FULL pool
    uses page64; req_to_token[request, 8*block] is the aligned physical start.
    Its page16 view uses the same physical token numbering. Each int64 output
    stores physical_start/8 in its high word and logical_start in its low word.

    The caller owns disjoint output buffers and keeps all inputs alive until
    consumption. Output widths 513/514/516 support the source split policies.
    Slots beyond count become -1; padded rows get causal length zero. Logical
    IDs, live request slots, aligned mappings, and positions below the model's
    context limit are producer contracts, as in step5_indexer_decode_logits. No tensor
    data is read by Python, so changed-input Graph replay remains supported.
    """
    if block_indices.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 sparse metadata requires NVIDIA CUDA")
    if block_indices.ndim != 2 or block_indices.shape[1] != 513:
        raise ValueError("Step5 block_indices must have shape [T,513]")
    rows = block_indices.shape[0]
    for name, tensor, shape, dtypes in (
        ("block_indices", block_indices, (rows, 513), (torch.int32,)),
        ("counts", counts, (rows,), (torch.int32,)),
        ("row_to_req", row_to_req, (rows,), (torch.int32, torch.int64)),
        ("positions", positions, (rows,), (torch.int32, torch.int64)),
        ("causal_lengths", causal_lengths, (rows,), (torch.int32,)),
    ):
        if (
            tensor.device != block_indices.device
            or tensor.dtype not in dtypes
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Step5 sparse metadata invalid {name}")
    if (
        req_to_token.device != block_indices.device
        or req_to_token.dtype not in (torch.int32, torch.int64)
        or req_to_token.ndim != 2
        or req_to_token.stride(1) != 1
    ):
        raise ValueError("Step5 req_to_token requires CUDA integer [requests, context]")
    if (
        packed_blocks.device != block_indices.device
        or packed_blocks.dtype != torch.int64
        or packed_blocks.ndim != 2
        or packed_blocks.shape[0] != rows
        or packed_blocks.shape[1] not in (513, 514, 516)
        or not packed_blocks.is_contiguous()
    ):
        raise ValueError("Step5 packed_blocks requires int64 [T,513/514/516]")
    if rows:
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
