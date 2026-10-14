# SPDX-License-Identifier: Apache-2.0
"""Batched integer metadata for Step5 non-overlapping R8 compression.

One request kernel counts owners, then one CTA per request validates all query
mappings and writes final request/query/region layouts. Kernel launch ordering
provides the global barrier for owner counts; no cross-CTA spin waits are used.
Buffers are owned by the eager forward or captured graph, never shared globally.
"""

import torch
import triton
import triton.language as tl


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
    VERIFY: tl.constexpr,
    NATIVE: tl.constexpr,
    QBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    REGIONS: tl.constexpr,
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
    region = tl.arange(0, RBLOCK)
    closing_depth = 7 - (base % 8 + 8) % 8 + region * 8
    closing_pos = tl.minimum(
        tl.maximum(base + tl.minimum(closing_depth, WIDTH - 1), 0), TOKENS - 1
    )
    offset = tl.arange(0, 8)
    logical = closing_pos[:, None] // 8 * 8 + offset[None, :]
    mapped = tl.load(
        Mapping + slot * MAP_STRIDE + tl.minimum(logical, TOKENS - 1) * MAP_COL_STRIDE
    )
    first = tl.sum(tl.where(offset[None, :] == 0, mapped.to(tl.int64), 0), 1)
    region_ok = (
        (first >= 64)
        & (first < PHYSICAL_LIMIT)
        & (first % 8 == 0)
        & tl.min((mapped == first[:, None] + offset[None, :]).to(tl.int32), 1)
    )
    completed = (region < REGIONS) & (closing_depth < WIDTH)
    admitted = (
        count_ok
        & (slot == raw)
        & (tl.load(Uses + slot) == 1)
        & tl.min(tl.where(depth < WIDTH, query_ok, True).to(tl.int32), 0)
        & tl.min((~completed | region_ok).to(tl.int32), 0)
    )
    tl.store(SafeSlots + row, slot)
    tl.store(
        Starts + row, base if VERIFY else tl.minimum(tl.maximum(base, 0), TOKENS - 1)
    )
    tl.store(Sources + row, row * WIDTH)
    tl.store(Lengths + row, WIDTH)
    tl.store(Active + row, active)
    tl.store(Admitted + row, admitted)
    if row == 0:
        tl.store(CountValid, count_ok)
        tl.store(QueryCount, live * WIDTH)

    # Store destinations directly in [completed-region, request] order.
    # A trailing, incomplete region is never consumed; match the last query's
    # mapping there so every output cell is deterministic across graph replays.
    tl.store(
        Writes + region * ROWS + row,
        (first - (first % 8 + 8) % 8) // 8,
        region < REGIONS,
    )

    query_pos = pos if VERIFY else safe_pos
    tl.store(QueryPositions + row * WIDTH + depth, query_pos, depth < WIDTH)
    tl.store(QuerySlots + row * WIDTH + depth, tl.where(active, raw, 0), depth < WIDTH)
    native_ok = True
    if NATIVE:
        native_pos = tl.load(
            NativePositions + row * WIDTH + depth, depth < WIDTH, other=0
        )
        out = tl.load(OutLoc + row * WIDTH + depth, depth < WIDTH, other=0)
        seq = tl.load(SeqLens + row).to(tl.int64)
        if VERIFY:
            length_ok = (seq >= 0) & (seq <= CONTEXT - WIDTH)
        else:
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
    token_width=1,
    verify=False,
    native_positions=None,
    seq_lens=None,
    out_cache_loc=None,
    context_len=None,
):
    """Build request admission, query layout and final region write addresses.

    ``positions`` names accepted-prefix lengths for verify, current-token
    positions for decode. Native input validation is folded into mapping loads
    when the backend supplies positions/lengths/write locations. No CPU reads of
    dynamic values or mapping dtype conversions occur here.
    """
    device = req_to_token.device
    rows = request_slots.numel()
    if (
        device.type != "cuda"
        or req_to_token.ndim != 2
        or req_to_token.dtype not in (torch.int32, torch.int64)
        or req_to_token.shape[0] < 1
        or req_to_token.shape[1] < 8
        or compressed_index_key_pages < 2
        or token_width < 1
        or (not verify and token_width != 1)
    ):
        raise ValueError(
            "CSA requires native CUDA request/FULL mapping and chain width"
        )
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

    regions = (7 + token_width) // 8
    result = dict(
        request_slots=alloc(rows, torch.int64),
        input_start_positions=alloc(rows, torch.int64),
        input_start_rows=alloc(rows, torch.int64),
        input_lengths=alloc(rows, torch.int64),
        request_active=alloc(rows, torch.bool),
        request_mapping_valid=alloc(rows, torch.bool),
        request_count_valid=alloc(1, torch.bool),
        compressed_key_write_indices=alloc((regions, rows), torch.int64),
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
            verify,
            native,
            triton.next_power_of_2(token_width),
            triton.next_power_of_2(regions),
            regions,
            num_warps=4,
        )
    else:
        result["request_count_valid"].copy_((valid_rows == 0))
        query_count.zero_()
    return result, query_slots, query_positions, query_count, native_valid
