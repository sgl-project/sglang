# SPDX-License-Identifier: Apache-2.0
"""Step5 DSA indexer: index-key compression, scoring and region selection.

Unlike the DeepSeek DSA indexer (dsa/), Step5 scores 8-token regions instead of
single tokens. A region key is a per-dimension softmax(Z)-weighted sum of its
eight normalized, rotated index keys, stored as FP8, and each query keeps the
top 512 completed regions plus its own region. QSA (qsa_indexer.py) also scores
compressed blocks, but mean-pools raw keys and applies norm and RoPE afterwards.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.deep_select import topk

STEP5_PAGE_SIZE = 64

# Index Q/K normalization and RoPE.


@triton.jit(
    do_not_specialize=("ROWS",),
    do_not_specialize_on_alignment=("ROWS",),
)
def _indexer_norm_rope_kernel(
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
    D: tl.constexpr,
    ROPE: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    real_row = token < ROWS
    channel = tl.arange(0, D)
    if head < HEADS:
        x = tl.load(Q + token * Q_ROW + head * D + channel, real_row, other=0).to(
            tl.float32
        )
        w = tl.load(QW + channel).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * inverse_rms * (w + 1)
    else:
        x = tl.load(K + token * K_ROW + channel, real_row, other=0).to(tl.float32)
        w = tl.load(KW + channel).to(tl.float32)
        b = tl.load(KB + channel).to(tl.float32)
        mean = tl.sum(x, 0) / D
        variance = tl.sum(x * x, 0) / D - mean * mean
        y = (x - mean) * tl.rsqrt(variance + EPS) * w + b

    # Normalization has its own BF16 boundary, including the rotated partner.
    normalized = y.to(tl.bfloat16).to(tl.float32)
    half = ROPE // 2
    partner = tl.gather(normalized, channel ^ half, axis=0)
    position = tl.load(Positions + token, real_row, other=0).to(tl.int64)
    cosine = tl.load(
        Cache + position * ROPE + channel % half, channel < ROPE, other=0
    ).to(tl.float32)
    sine = tl.load(
        Cache + position * ROPE + half + channel % half, channel < ROPE, other=0
    ).to(tl.float32)
    rotated = normalized * cosine + tl.where(channel < half, -partner, partner) * sine
    result = tl.where(channel < ROPE, rotated, normalized).to(tl.bfloat16)
    if head < HEADS:
        tl.store(QOut + (token * HEADS + head) * D + channel, result)
    if head == HEADS:
        tl.store(KOut + token * D + channel, result, real_row)


def step5_dsa_indexer_norm_rope(
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
    """Zero-centered RMSNorm on index Q, LayerNorm on index K, then NeoX RoPE on the
    first ``cache.shape[1]`` dims.

    Q is BF16 [T, 2 or 4, D] and K is BF16 [T, 1, D]; rows may be padded.
    """
    assert q.ndim == 3 and q.shape[1] in (2, 4)
    tokens, heads, head_dim = q.shape
    assert k.shape == (tokens, 1, head_dim)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert q.stride(1) == head_dim and q.stride(2) == 1 and k.stride(2) == 1
    assert cache.is_contiguous()
    q_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    k_out = torch.empty(k.shape, dtype=torch.bfloat16, device=k.device)
    if tokens:
        _indexer_norm_rope_kernel[(tokens, heads + 1)](
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
            head_dim,
            cache.shape[1],
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out


# Index-key compression and ring state.


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
def _compressed_index_key_bytes(
    index_keys, compression_logits, REGION: tl.constexpr, D: tl.constexpr
):
    # raw payloads retain their BF16 bits, including unused NaN slots, until
    # all eight tokens are ready for the numerical reduction.
    maximum = tl.full((D,), -float("inf"), tl.float32)
    for token in tl.static_range(REGION):
        indices = tl.full((1, D), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((D,))
        logit = logit.to(tl.bfloat16, bitcast=True).to(tl.float32)
        maximum = tl.maximum(maximum, logit, propagate_nan=tl.PropagateNan.NONE)
    block_max = tl.max(maximum, axis=0)
    numerator = tl.full((D,), 0.0, tl.float32)
    denominator = tl.full((D,), 0.0, tl.float32)
    for token in tl.static_range(REGION):
        indices = tl.full((1, D), token, tl.int32)
        logit = tl.gather(compression_logits, indices, axis=0).reshape((D,))
        value = tl.gather(index_keys, indices, axis=0).reshape((D,))
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
):
    live = tl.load(active + row) != 0
    ok = tl.full((), False, tl.int1)
    if live & (tl.load(admitted + row) != 0) & (tl.load(count_valid) != 0):
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        pos = base // CAPACITY * CAPACITY + tl.arange(0, CAPACITY)
        actual = tl.load(tags + slot * CAPACITY + pos % CAPACITY)
        history_ok = tl.sum(((pos < base) & (actual != pos)).to(tl.int32), 0) == 0
        ok = history_ok
    tl.store(enabled + row, ok)
    row_ok = (tl.load(count_valid) != 0) & (~live | ok)
    tl.store(success + row, row_ok)
    tl.device_assert(row_ok, "Step5 DSA ring history failed")
    return ok


@triton.jit
def _admit_kernel(
    slots,
    bases,
    active,
    admitted,
    count_valid,
    tags,
    enabled,
    success,
    CAPACITY: tl.constexpr,
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
    )


@triton.jit
def _compress_kernel(
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
    CAPACITY: tl.constexpr,
    ROWS: tl.constexpr,
    Z_ROW: tl.constexpr,
    D: tl.constexpr,
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
        )
    else:
        ok = tl.load(enabled + row)
    if ok:
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        first = base // CAPACITY * CAPACITY
        block_idx = first + block_offset * CAPACITY
        if block_idx + CAPACITY <= end:
            slot = tl.load(slots + row).to(tl.int64)
            source = tl.load(sources + row).to(tl.int64)
            pos = block_idx + tl.arange(0, CAPACITY)
            ch = tl.arange(0, D)
            history = pos < base
            ring_offset = (slot * CAPACITY + pos % CAPACITY)[:, None] * D + ch[None, :]
            input_offset = (source + pos - base)[:, None] * D + ch[None, :]
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
                tl.where(history[:, None], rk, ik),
                tl.where(history[:, None], rz, iz),
                CAPACITY,
                D,
            )
            dest = tl.load(physical + block_offset * ROWS + row).to(tl.int64)
            tl.store(paged + dest * D + ch, packed)


@triton.jit
def _write_ring_kernel(
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
    D: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(enabled + row):
        slot = tl.load(slots + row).to(tl.int64)
        base = tl.load(bases + row)
        end = base + tl.load(lengths + row)
        source = tl.load(sources + row).to(tl.int64)
        # One lane per ring cell, selecting the newest input with that residue.
        cell = tl.arange(0, BLOCK)
        pos = end - 1 - (end - 1 + CAPACITY - cell) % CAPACITY
        write = (cell < CAPACITY) & (pos >= base)
        ch = tl.arange(0, D)
        src = (source + pos - base)[:, None] * D + ch[None, :]
        dst = (slot * CAPACITY + cell)[:, None] * D + ch[None, :]
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
        tag = tl.where(write, pos, tl.where(base % CAPACITY == 0, -1, previous))
        tl.store(tags + slot * CAPACITY + cell, tag, cell < CAPACITY)


def step5_dsa_append_ring(
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
    """Compress completed regions into the FP8 index-key pages, then keep
    each request's newest K/Z inputs in its ring for the next forward."""
    assert index_key.is_contiguous() and compression_logits.stride(-1) == 1
    rows = slots.numel()
    max_blocks = physical_blocks.shape[0]
    capacity = index_key_ring.shape[1]
    # _compress_kernel reads the ring before _write_ring_kernel overwrites it.
    if max_blocks != 1:
        _admit_kernel[(rows,)](
            slots,
            bases,
            active,
            admitted,
            count_valid,
            ring_positions,
            enabled,
            success,
            capacity,
            debug=True,
        )
    if max_blocks:
        _compress_kernel[(max_blocks, rows)](
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
            capacity,
            rows,
            compression_logits.stride(0),
            index_key.shape[-1],
            num_warps=8,
            enable_fp_fusion=True,
            debug=True,
        )
    _write_ring_kernel[(rows,)](
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
        index_key.shape[-1],
        num_warps=8,
    )


# Indexer logits.


@triton.jit(
    do_not_specialize=("BLOCKS",),
    do_not_specialize_on_alignment=("BLOCKS",),
)
def _score_blocks_kernel(
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
    D: tl.constexpr,
    REGION: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    head = tl.arange(0, 16)
    channel = tl.arange(0, D)
    request = tl.load(RowToReq + row).to(tl.int64)
    position = tl.load(Positions + row)
    query = tl.load(
        Q + (row * HEADS + head[:, None]) * D + channel[None, :],
        (request > 0) & (head[:, None] < HEADS),
        other=0.0,
    ).to(tl.bfloat16)
    weights = tl.load(
        W + row * HEADS + head, (request > 0) & (head < HEADS), other=0
    ).to(tl.float32)
    last_tile = tl.cdiv(BLOCKS, BLOCK_R)
    for tile in range(first_tile, last_tile, GRID_TILES):
        num_blocks = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        if (request > 0) & (tile * BLOCK_R < position // REGION):
            valid = num_blocks < position // REGION
            token = tl.load(
                ReqToToken + request * MAP_STRIDE + num_blocks * REGION,
                valid,
                other=0,
            ).to(tl.int64)
            offset = (token // REGION)[None, :] * D + channel[:, None]
            packed = tl.load(CompressedIndexKeys + offset, valid[None, :], other=0)
            keys = packed.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            products = tl.dot(query, keys)
            score = tl.sum(tl.maximum(products, 0.0) * weights[:, None], axis=0)
            score = tl.where(valid, score, -float("inf"))
        else:
            score = tl.full((BLOCK_R,), -float("inf"), tl.float32)
        tl.store(Out + row * BLOCKS + num_blocks, score, num_blocks < BLOCKS)


def _indexer_logits(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    region_size: int,
) -> None:
    """Score each query row against its completed history blocks into FP32 ``out``.

    Q is BF16 [T, 2 or 4, D] and weights are FP32 [T, heads], already scaled;
    future and padding columns of ``out`` are written as -inf.
    """
    assert q.ndim == 3 and q.shape[1] in (2, 4)
    tokens, heads, head_dim = q.shape
    assert q.is_contiguous() and weights.is_contiguous()
    assert weights.shape == (tokens, heads) and weights.dtype == torch.float32
    assert compressed_index_keys.shape[1:] == (
        STEP5_PAGE_SIZE // region_size,
        1,
        head_dim,
    )
    assert compressed_index_keys.dtype == torch.uint8
    assert out.shape[0] == tokens and out.dtype == torch.float32
    assert out.is_contiguous()
    # D256 gathers are register-heavy; 32-block tiles also give small batches more CTAs.
    blocks_per_tile = 32
    grid_tiles = min(triton.cdiv(out.shape[1], blocks_per_tile), 128)
    _score_blocks_kernel[(tokens, grid_tiles)](
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
        head_dim,
        region_size,
        num_warps=4,
    )


# Block top-k and packed attention addresses.


@triton.jit
def _pack_blocks_kernel(
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
    TOPK: tl.constexpr,
    REGION: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, TOPK)
    position = tl.load(Positions + row).to(tl.int32)
    current = position // REGION
    request = tl.load(Requests + row).to(tl.int64)
    live = request > 0
    count = tl.minimum(current, TOPK)
    if K > 0:
        ids = tl.load(Indices + row * K + lane, lane < K, other=2147483647).to(tl.int32)
        ids = tl.where((ids >= 0) & (ids < current), ids, 2147483647)
        ids = tl.sort(ids, descending=False)
    else:
        ids = lane
    result = tl.where(lane < count, ids, tl.where(lane == count, current, -1))
    tl.store(Out + row * (TOPK + 1) + lane, tl.where(live, result, -1))
    tl.store(
        Out + row * (TOPK + 1) + TOPK, tl.where(live & (count == TOPK), current, -1)
    )
    tl.store(Counts + row, tl.where(live, count + 1, 0))

    if PACKED_WIDTH > 0:
        active = live & (lane <= count)
        logical = result.to(tl.int64) * REGION
        physical = tl.load(
            Mapping + request * MAP_STRIDE + logical, active, other=0
        ).to(tl.int64)
        value = ((physical // REGION) << 32) | logical
        tl.store(Packed + row * PACKED_WIDTH + lane, tl.where(active, value, -1))

        # The current block occupies slot TOPK when all history slots are used.
        # Clear the remaining ABI padding for widths TOPK + 1, 2 and 4 as well.
        tail = tl.arange(0, 4)
        active_tail = live & (count == TOPK) & (tail == 0)
        logical_tail = current.to(tl.int64) * REGION
        physical_tail = tl.load(
            Mapping + request * MAP_STRIDE + logical_tail + tl.zeros((4,), tl.int64),
            active_tail,
            other=0,
        ).to(tl.int64)
        value_tail = ((physical_tail // REGION) << 32) | logical_tail
        tl.store(
            Packed + row * PACKED_WIDTH + TOPK + tail,
            tl.where(active_tail, value_tail, -1),
            TOPK + tail < PACKED_WIDTH,
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
    topk,
    region_size,
    req_to_token=None,
    packed=None,
    causal_lengths=None,
):
    rows = block_indices.shape[0]
    _pack_blocks_kernel[(rows,)](
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
        topk,
        region_size,
        num_warps=1 if packed is not None and rows >= 512 else 4,
    )


# Preserve the score workspace layout while sharing the public top-k kernel.
_SCORE_STRIDE_ALIGNMENT = 256


def deepselect_score_width(score_width: int) -> int:
    align = _SCORE_STRIDE_ALIGNMENT
    return (score_width + align - 1) // align * align


def step5_dsa_select_blocks(
    *,
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    tiles: tuple[tuple[int, int], ...],
    indexer_logits: torch.Tensor,
    ids: torch.Tensor,
    packed: torch.Tensor,
    counts: torch.Tensor,
    lengths: torch.Tensor,
    use_deepselect: bool,
    topk_lengths: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    topk_size: int,
    region_size: int,
) -> None:
    for start, end in tiles:
        rows = slice(start, end)
        if use_deepselect:
            _indexer_logits(
                q=index_q[rows],
                weights=index_weights[rows],
                compressed_index_keys=compressed_index_keys,
                req_to_token=req_to_token,
                row_to_req=row_to_req[rows],
                positions=positions[rows],
                out=indexer_logits[: end - start],
                region_size=region_size,
            )
            topk(
                indexer_logits[: end - start],
                topk_size,
                end=topk_lengths[rows],
                output_idx=topk_ids[: end - start],
                idx_oob_fill_value=-1,
            )
        _launch_block_pack(
            indices=topk_ids[: end - start] if use_deepselect else ids[: end - start],
            positions=positions[rows],
            row_to_req=row_to_req[rows],
            block_indices=ids[: end - start],
            counts=counts[rows],
            selected_k=topk_size if use_deepselect else 0,
            topk=topk_size,
            region_size=region_size,
            req_to_token=req_to_token,
            packed=packed[rows],
            causal_lengths=lengths[rows],
        )
