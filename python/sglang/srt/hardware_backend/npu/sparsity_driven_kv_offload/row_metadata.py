"""Triton row metadata for sparse-KV plans and MemFabric copies."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mtp_rows_kernel(
    requests,
    seq_lens,
    positions,
    locations,
    counts,
    slots_out,
    positions_out,
    valid_out,
    destination_out,
    N: tl.constexpr,
    OUT_N: tl.constexpr,
    Q: tl.constexpr,
    REQUEST_ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    HAS_LOC: tl.constexpr,
    DRAFT: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    PUBLISH: tl.constexpr,
    HOST_CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // Q
    req = tl.load(requests + row * REQ_STRIDE, i < N, other=0).to(tl.int64)
    length = tl.load(seq_lens + row * SEQ_STRIDE, i < N, other=0)
    pos = tl.load(positions + i * POS_STRIDE, i < N, other=-1).to(tl.int64)
    live = (req > 0) & (req < REQUEST_ROWS) & (length > 0)
    live = live & (pos >= 0) & (pos < CONTEXT)
    if DRAFT:
        count = tl.load(counts + row * COUNT_STRIDE, i < N, other=0)
        live = live & (i % Q < count)
    if HAS_LOC:
        loc = tl.load(locations + i * LOC_STRIDE, i < OUT_N, other=-1).to(tl.int64)
        live = live & (loc >= 0)
    if PUBLISH:
        live = live & (loc < HOST_CAPACITY) & (i < N)
        tl.store(slots_out + i, i, i < OUT_N)
        tl.store(destination_out + i, loc, i < OUT_N)
    else:
        tl.store(slots_out + i, req, i < OUT_N)
    tl.store(positions_out + i, pos, i < OUT_N)
    tl.store(valid_out + i, live, i < OUT_N)


def mtp_rows(
    batch,
    query_width,
    request_rows,
    context_len,
    *,
    publication_rows=None,
    host_capacity=0,
):
    """Materialize request IDs, positions and masks in a single device launch."""
    logical_tokens = batch.req_pool_indices.numel() * query_width
    shape = (
        (batch.req_pool_indices.numel(), query_width)
        if publication_rows is None
        else (publication_rows,)
    )
    slots = torch.empty(shape, dtype=torch.long, device=batch.positions.device)
    positions = torch.empty_like(slots)
    valid = torch.empty(shape, dtype=torch.bool, device=slots.device)
    destination = torch.empty_like(slots) if publication_rows is not None else None
    draft = batch.forward_mode.is_draft_extend_v2()
    counts = batch.spec_info.num_accept_tokens if draft else None
    input_positions = batch.positions.reshape(-1)
    locations = (
        batch.out_cache_loc.reshape(-1) if batch.out_cache_loc is not None else None
    )
    _mtp_rows_kernel[(triton.cdiv(slots.numel(), 256),)](
        batch.req_pool_indices,
        batch.seq_lens,
        input_positions,
        locations,
        counts,
        slots,
        positions,
        valid,
        destination,
        logical_tokens,
        slots.numel(),
        query_width,
        request_rows,
        context_len,
        locations is not None,
        draft,
        batch.req_pool_indices.stride(0),
        batch.seq_lens.stride(0),
        input_positions.stride(0),
        locations.stride(0) if locations is not None else 0,
        counts.stride(0) if counts is not None else 0,
        publication_rows is not None,
        host_capacity,
        256,
    )
    if publication_rows is not None:
        return slots, destination, positions, valid
    return slots, positions, valid


def mtp_publication_rows(batch, num_tokens, request_rows, context_len, host_capacity):
    """Fuse MTP masks, TP padding, physical bounds and publication indices."""
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
        mtp_row_geometry,
    )

    _, width, _ = mtp_row_geometry(batch, num_tokens)
    if batch.out_cache_loc is None or batch.out_cache_loc.numel() != num_tokens:
        raise ValueError("Sparse MTP publication requires one physical row per token")
    return mtp_rows(
        batch,
        width,
        request_rows,
        context_len,
        publication_rows=num_tokens,
        host_capacity=host_capacity,
    )


@triton.jit
def _prefill_plan_kernel(
    requests,
    lengths,
    table,
    source,
    destination,
    valid,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    LEN_STRIDE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    ids = tl.arange(0, BATCH_BLOCK)
    reqs = tl.load(requests + ids * REQ_STRIDE, ids < B, other=-1)
    lens = tl.load(lengths + ids * LEN_STRIDE, ids < B, other=0).to(tl.int64)
    lens = tl.where((reqs >= 0) & (lens > 0), lens, 0)
    start = tl.sum(tl.where(ids < r, lens, 0), 0)
    req = tl.load(requests + r * REQ_STRIDE).to(tl.int64)
    length = tl.load(lengths + r * LEN_STRIDE)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = (req >= 0) & (length > 0) & (pos < length)
    safe_req = tl.minimum(tl.maximum(req, 0), ROWS - 1)
    safe_pos = tl.minimum(pos, CONTEXT - 1)
    physical = tl.load(table + safe_req * S0 + safe_pos * S1, live, other=0)
    out = start + pos
    tl.store(source + out, physical.to(tl.int64), live)
    tl.store(destination + out, out, live)
    tl.store(valid + out, True, live)


def prefill_plan(batch, manager, total):
    """Expand ragged request rows directly through the physical token map."""
    table = manager.req_to_token_pool.req_to_token
    source = torch.empty(total, dtype=torch.long, device=table.device)
    destination = torch.empty_like(source)
    valid = torch.empty(total, dtype=torch.bool, device=table.device)
    if total:
        b = batch.req_pool_indices.numel()
        max_length = int(batch.seq_lens_cpu.max())
        _prefill_plan_kernel[(b, triton.cdiv(max_length, 256))](
            batch.req_pool_indices,
            batch.seq_lens,
            table,
            source,
            destination,
            valid,
            b,
            manager.actual_request_rows,
            manager.logical_max_context_len,
            table.stride(0),
            table.stride(1),
            batch.req_pool_indices.stride(0),
            batch.seq_lens.stride(0),
            triton.next_power_of_2(b),
            256,
        )
    return source, destination, valid


@triton.jit
def _mapping_init_kernel(
    counts, host_mask, CAPACITY: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(counts + i, 0, i < CAPACITY)
    tl.store(host_mask + i, True, i < TOTAL)


@triton.jit
def _mapping_count_kernel(
    metadata,
    locations,
    current_valid,
    counts,
    src_out,
    dst_out,
    valid_out,
    N: tl.constexpr,
    B: tl.constexpr,
    CAPACITY: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Metadata is O(B): current-row end, full-KV start, prefix, live-row end.
    rows = tl.arange(0, BATCH_BLOCK)
    ends = tl.load(metadata + rows, rows < B, other=0)
    owner = tl.sum(
        ((i[:, None] >= ends[None, :]) & (rows[None, :] < B)).to(tl.int32), 1
    )
    previous = tl.load(metadata + owner - 1, (i < N) & (owner > 0), other=0)
    start = tl.load(metadata + B + owner, i < N, other=0)
    prefix = tl.load(metadata + 2 * B + owner, i < N, other=0)
    live_end = tl.load(metadata + 3 * B + owner, i < N, other=0)
    eligible = i < live_end
    dst = tl.where(eligible, start + prefix + i - previous, 0).to(tl.int64)
    loc = tl.load(locations + i, i < N, other=-1).to(tl.int64)
    live = tl.load(current_valid + i, i < N, other=0) != 0
    live = live & eligible & (loc >= 0) & (loc < CAPACITY) & (i < N)
    tl.atomic_add(counts + tl.minimum(tl.maximum(loc, 0), CAPACITY - 1), 1, live)
    tl.store(src_out + i, i, i < N)
    tl.store(dst_out + i, dst, i < N)
    tl.store(valid_out + i, live, i < N)


@triton.jit
def _mapping_finish_kernel(
    source,
    locations,
    counts,
    destination,
    valid,
    host_mask,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(valid + i, i < N, other=0) != 0
    loc = tl.load(locations + i, i < N, other=0).to(tl.int64)
    safe = tl.minimum(tl.maximum(loc, 0), CAPACITY - 1)
    count = tl.load(counts + safe, i < N, other=0)
    dst = tl.load(destination + i, i < N, other=0)
    physical = tl.load(source + dst, i < N, other=-1)
    live = live & (count == 1) & (physical == loc)
    tl.store(valid + i, live, i < N)
    # CPU layout validation guarantees unique destinations for eligible rows.
    tl.store(host_mask + dst, False, (i < N) & live)


def prefill_current_mapping(
    plan, batch, current_valid, capacity, slots, requests, lengths, prefixes, extends
):
    """Build a D2D map while rejecting duplicate physical current rows."""
    locations = batch.out_cache_loc.reshape(-1).contiguous()
    current_valid = current_valid.reshape(-1).contiguous()
    ends, starts, live_ends = [], [], []
    row_end = full_start = 0
    for width, req, length, ext in zip(slots, requests, lengths, extends):
        starts.append(full_start)
        live_ends.append(row_end + ext if req > 0 and length > 0 else row_end)
        row_end += width
        ends.append(row_end)
        if req >= 0 and length > 0:
            full_start += length
    device = plan.source.device
    metadata = torch.tensor(
        ends + starts + prefixes + live_ends, device=device, dtype=torch.long
    )
    n, b = current_valid.numel(), len(slots)
    src = torch.empty(n, device=device, dtype=torch.long)
    dst = torch.empty_like(src)
    valid = torch.empty(n, device=device, dtype=torch.bool)
    host_mask = torch.empty(plan.total_tokens, device=device, dtype=torch.bool)
    counts = torch.empty(capacity, device=device, dtype=torch.int32)
    _mapping_init_kernel[(triton.cdiv(max(capacity, plan.total_tokens), 512),)](
        counts,
        host_mask,
        capacity,
        plan.total_tokens,
        512,
    )
    _mapping_count_kernel[(triton.cdiv(n, 128),)](
        metadata,
        locations,
        current_valid,
        counts,
        src,
        dst,
        valid,
        n,
        b,
        capacity,
        triton.next_power_of_2(b),
        128,
    )
    _mapping_finish_kernel[(triton.cdiv(n, 256),)](
        plan.source,
        locations,
        counts,
        dst,
        valid,
        host_mask,
        n,
        capacity,
        256,
    )
    return src, dst, valid, host_mask


@triton.jit
def _copy_rows_metadata_kernel(
    source_rows,
    destination_rows,
    valid,
    src,
    dst,
    lengths,
    count,
    source_bases,
    destination_bases,
    WIDTHS: tl.constexpr,
    N: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    VALID_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(valid + i * VALID_STRIDE, i < N, other=0) != 0
    s = tl.load(source_rows + i * SRC_STRIDE, i < N, other=0).to(tl.int64)
    d = tl.load(destination_rows + i * DST_STRIDE, i < N, other=0).to(tl.int64)
    s, d = tl.where(live, s, 0), tl.where(live, d, 0)
    for c in tl.static_range(len(WIDTHS)):
        tl.store(src + c * N + i, source_bases[c] + s * WIDTHS[c], i < N)
        tl.store(dst + c * N + i, destination_bases[c] + d * WIDTHS[c], i < N)
        tl.store(lengths + c * N + i, tl.where(live, WIDTHS[c], 0), i < N)
    if tl.program_id(0) == 0:
        tl.store(count, N * len(WIDTHS))


def copy_rows_metadata(sources, destinations, source_rows, destination_rows, valid):
    """Build component-major MF byte descriptors without touching host KV."""
    n = source_rows.numel()
    device = source_rows.device
    src = torch.empty(n * len(sources), dtype=torch.int64, device=device)
    dst = torch.empty_like(src)
    lengths = torch.empty(src.shape, dtype=torch.int32, device=device)
    count = torch.empty(1, dtype=torch.int32, device=device)
    _copy_rows_metadata_kernel[(triton.cdiv(n, 256),)](
        source_rows,
        destination_rows,
        valid,
        src,
        dst,
        lengths,
        count,
        tuple(s.data_ptr() for s in sources),
        tuple(d.data_ptr() for d in destinations),
        tuple(s.shape[-1] * s.element_size() for s in sources),
        n,
        source_rows.stride(0),
        destination_rows.stride(0),
        valid.stride(0),
        256,
    )
    return src, dst, lengths, count
