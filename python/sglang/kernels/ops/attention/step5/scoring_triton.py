# SPDX-License-Identifier: Apache-2.0
"""Paged Step5 scoring over existing SW128 E4M3 compressed_index_keys."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=("REGIONS",),
    do_not_specialize_on_alignment=("REGIONS",),
)
def _score_regions(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
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
    last_tile = tl.cdiv(REGIONS, BLOCK_R)
    if not CLEAR_FUTURE:
        live_regions = tl.maximum(position // 8, 0)
        last_tile = tl.minimum(tl.cdiv(live_regions, BLOCK_R), last_tile)
        # Selecting every historical region does not consume region_scores.
        last_tile = tl.where(live_regions <= 512, 0, last_tile)
        last_tile = tl.where(request > 0, last_tile, 0)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        if (request > 0) & (tile * BLOCK_R < position // 8):
            valid = regions < position // 8
            token = tl.load(
                ReqToToken + request * MAP_STRIDE + regions * 8,
                valid,
                other=0,
            ).to(tl.int64)
            page, slot = token // 64, token % 64 // 8
            # Each page contains two 128-byte channel atoms, each with eight
            # swizzled region rows. XOR uses the physical row, never logical ID.
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
        tl.store(Out + row * REGIONS + regions, score, regions < REGIONS)


@triton.jit(
    do_not_specialize=("ROWS", "REGIONS"),
    do_not_specialize_on_alignment=("ROWS", "REGIONS"),
)
def _score_four_rows_per_request(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    ROWS,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
):
    """Fill four MTP verifier rows with one fully occupied 16x256 dot."""
    group = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    group_rows = group * 4 + tl.arange(0, 4)
    matrix_row = tl.arange(0, 16)
    query_row = group * 4 + matrix_row // 4
    head = matrix_row % 4
    channel = tl.arange(0, 256)
    row_valid = group_rows < ROWS
    request = tl.load(RowToReq + group_rows, row_valid, other=0).to(tl.int64)
    position = tl.load(Positions + group_rows, row_valid, other=0)
    # Verify metadata guarantees four consecutive rows for one request. Padding
    # groups carry request slot zero in all four rows.
    base_request = tl.load(RowToReq + group * 4).to(tl.int64)
    query = tl.load(
        Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
        (query_row[:, None] < ROWS) & (head[:, None] < HEADS) & (base_request > 0),
        other=0.0,
    ).to(tl.bfloat16)
    weights = (
        tl.load(
            W + group_rows[:, None] * HEADS + tl.arange(0, 4)[None, :],
            row_valid[:, None]
            & (tl.arange(0, 4)[None, :] < HEADS)
            & (request[:, None] > 0),
            other=0.0,
        ).to(tl.float32)
        * WEIGHT_SCALE
    )
    live = tl.where(row_valid & (request > 0), tl.maximum(position // 8, 0), 0)
    max_live = tl.max(live, axis=0)
    last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
    last_tile = tl.where(max_live <= 512, 0, last_tile)
    for tile in range(first_tile, last_tile, GRID_TILES):
        regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        valid = regions < max_live
        token = tl.load(
            ReqToToken + base_request * MAP_STRIDE + regions * 8,
            valid & (base_request > 0),
            other=0,
        ).to(tl.int64)
        page, slot = token // 64, token % 64 // 8
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
        weighted = tl.maximum(products, 0.0) * tl.reshape(weights, (16, 1))
        region_scores = tl.sum(tl.reshape(weighted, (4, 4, BLOCK_R)), axis=1)
        output_valid = row_valid[:, None] & (regions[None, :] < live[:, None])
        tl.store(
            Out + group_rows[:, None] * REGIONS + regions[None, :],
            region_scores,
            output_valid,
        )


@triton.jit(
    do_not_specialize=("ROWS", "REGIONS"),
    do_not_specialize_on_alignment=("ROWS", "REGIONS"),
)
def _score_thirty_two_rows_per_request(
    Q,
    W,
    CompressedIndexKeys,
    ReqToToken,
    RowToReq,
    Positions,
    Out,
    HEADS: tl.constexpr,
    ROWS,
    MAP_STRIDE: tl.constexpr,
    REGIONS,
    BLOCK_R: tl.constexpr,
    GRID_TILES: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    group_base = group * 32
    channel = tl.arange(0, 256)
    matrix_row = tl.arange(0, 32)
    head = matrix_row % 4
    row_in_subgroup = tl.arange(0, 8)
    for subgroup in tl.static_range(0, 4):
        rows = group_base + subgroup * 8 + row_in_subgroup
        row_valid = rows < ROWS
        request = tl.load(RowToReq + rows, row_valid, other=0).to(tl.int64)
        position = tl.load(Positions + rows, row_valid, other=0)
        base_request = tl.load(RowToReq + group_base).to(tl.int64)
        query_row = group_base + subgroup * 8 + matrix_row // 4
        query = tl.load(
            Q + (query_row[:, None] * HEADS + head[:, None]) * 256 + channel[None, :],
            (query_row[:, None] < ROWS) & (head[:, None] < HEADS) & (base_request > 0),
            other=0.0,
        ).to(tl.bfloat16)
        weights = (
            tl.load(
                W + rows[:, None] * HEADS + tl.arange(0, 4)[None, :],
                row_valid[:, None]
                & (tl.arange(0, 4)[None, :] < HEADS)
                & (request[:, None] > 0),
                other=0.0,
            ).to(tl.float32)
            * WEIGHT_SCALE
        )
        live = tl.where(row_valid & (request > 0), tl.maximum(position // 8, 0), 0)
        max_live = tl.max(live, axis=0)
        last_tile = tl.minimum(tl.cdiv(max_live, BLOCK_R), tl.cdiv(REGIONS, BLOCK_R))
        last_tile = tl.where(max_live <= 512, 0, last_tile)
        for tile in range(first_tile, last_tile, GRID_TILES):
            regions = tile * BLOCK_R + tl.arange(0, BLOCK_R)
            valid = regions < max_live
            token = tl.load(
                ReqToToken + base_request * MAP_STRIDE + regions * 8,
                valid & (base_request > 0),
                other=0,
            ).to(tl.int64)
            page, slot = token // 64, token % 64 // 8
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
            weighted = tl.maximum(products, 0.0) * tl.reshape(weights, (32, 1))
            # Preserve group1's 16-lane reduction; compact reduction changes rounding.
            heads = tl.permute(tl.reshape(weighted, (8, 4, BLOCK_R)), 0, 2, 1)
            padded = tl.join(heads, tl.zeros_like(heads))
            padded = tl.reshape(tl.permute(padded, 0, 1, 3, 2), (8, BLOCK_R, 8))
            padded = tl.join(padded, tl.zeros_like(padded))
            padded = tl.reshape(tl.permute(padded, 0, 1, 3, 2), (8, BLOCK_R, 16))
            region_scores = tl.sum(padded, axis=2)
            output_valid = row_valid[:, None] & (regions[None, :] < live[:, None])
            tl.store(
                Out + rows[:, None] * REGIONS + regions[None, :],
                region_scores,
                output_valid,
            )


def launch(
    q,
    weights,
    compressed_index_keys,
    req_to_token,
    row_to_req,
    positions,
    out,
    *,
    clear_future,
    group_rows=1,
    weight_scale=1.0,
):
    # D256 gathers already consume substantial registers. A 32-region tile
    # reduces that pressure and gives small batches more independent CTAs.
    block_regions = 32
    grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 128)
    if group_rows == 32 and not clear_future:
        _score_thirty_two_rows_per_request[(triton.cdiv(q.shape[0], 32), grid_tiles)](
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            q.shape[1],
            q.shape[0],
            req_to_token.stride(0),
            out.shape[1],
            block_regions,
            grid_tiles,
            weight_scale,
            num_warps=4,
        )
        return
    if group_rows == 4 and not clear_future:
        if q.shape[0] <= 16 and torch.cuda.get_device_capability(q.device) == (10, 3):
            if q.shape[0] == 4:
                block_regions = 16
                grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 1024)
            else:
                grid_tiles = min(triton.cdiv(out.shape[1], block_regions), 512)

        _score_four_rows_per_request[(triton.cdiv(q.shape[0], 4), grid_tiles)](
            q,
            weights,
            compressed_index_keys,
            req_to_token,
            row_to_req,
            positions,
            out,
            q.shape[1],
            q.shape[0],
            req_to_token.stride(0),
            out.shape[1],
            block_regions,
            grid_tiles,
            weight_scale,
            num_warps=4,
        )
        return
    _score_regions[(q.shape[0], grid_tiles)](
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
        block_regions,
        grid_tiles,
        clear_future,
        weight_scale,
        num_warps=4,
    )
