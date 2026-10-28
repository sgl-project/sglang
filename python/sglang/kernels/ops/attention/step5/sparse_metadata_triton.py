# SPDX-License-Identifier: Apache-2.0
"""Lossless logical-to-physical metadata adaptation for Step5 sparse decode."""

import triton
import triton.language as tl


@triton.jit
def _pack_decode_regions(
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
    region = tl.load(Ids + row * 513 + lane, active, other=0).to(tl.int64)
    logical_start = region * 8
    physical_start = tl.load(
        Mapping + request * MAP_STRIDE + logical_start,
        active,
        other=0,
    ).to(tl.int64)
    value = ((physical_start // 8) << 32) | logical_start
    tl.store(Packed + row * WIDTH + lane, tl.where(active, value, -1), lane < WIDTH)
    position = tl.load(Positions + row).to(tl.int32)
    tl.store(Lengths + row, tl.where(live, position + 1, 0))


def pack_decode_regions(ids, counts, mapping, requests, positions, packed, lengths):
    _pack_decode_regions[(ids.shape[0],)](
        ids,
        counts,
        mapping,
        requests,
        positions,
        packed,
        lengths,
        mapping.stride(0),
        packed.shape[1],
        num_warps=4,
    )
