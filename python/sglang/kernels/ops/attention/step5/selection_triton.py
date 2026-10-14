# SPDX-License-Identifier: Apache-2.0
"""Sort Step5 region IDs, append the current region and optionally pack addresses."""

import triton
import triton.language as tl


@triton.jit
def _canonicalize_zero(Scores, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(Scores + offset, offset < N, other=0.0).to(tl.uint32, bitcast=True)
    # FlashInfer's radix ordering distinguishes signed zero. Preserve all other
    # bits, including subnormals, so numerical ties prefer the smaller region ID.
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    tl.store(Out + offset, bits.to(tl.float32, bitcast=True), offset < N)


def canonicalize_zero(scores, out):
    size = scores.numel()
    _canonicalize_zero[(triton.cdiv(size, 1024),)](scores, out, size, 1024, num_warps=4)


@triton.jit
def _pack_regions(
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

        # The current region occupies slot 512 when all history slots are used.
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


def launch(
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
    _pack_regions[(out.shape[0],)](
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
