"""Keep short KV contiguous in HBM and restore only missing logical ranges."""

import triton
import triton.language as tl
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace


@triton.jit
def _history_metadata(
    requests,
    lengths,
    first_positions,
    req_to_token,
    old_begin,
    old_end,
    next_begin,
    next_end,
    next_first,
    source,
    destination,
    valid,
    ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    C: tl.constexpr,
    STRIDE: tl.constexpr,
    Q: tl.constexpr,
    FULL_VISIBLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    req = tl.load(requests + row).to(tl.int32)
    length = tl.load(lengths + row).to(tl.int32)
    first = tl.load(first_positions + row * FIRST_STRIDE).to(tl.int32)
    active = (
        (req > 0) & (req < ROWS) & (length > 0) & (length <= CONTEXT) & (first >= 0)
    )
    safe = tl.where(active, req, 0)
    prefix = tl.maximum(length - Q, 0) // 128 * 128
    dense = length < C if FULL_VISIBLE else prefix < C
    begin = tl.where(dense, 0, tl.where(FULL_VISIBLE, -1, prefix)).to(tl.int32)
    previous_begin = tl.load(old_begin + safe)
    previous_end = tl.load(old_end + safe)
    same_layout = (previous_begin >= 0) & ((previous_begin == 0) == (begin == 0))
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    pos = tl.where(dense, offset, prefix + offset)
    present = same_layout & (pos >= previous_begin) & (pos < previous_end)
    restore = active & (begin >= 0) & (offset < STRIDE) & (pos < first) & ~present
    physical = tl.load(
        req_to_token + safe * REQUEST_STRIDE + pos, mask=restore, other=0
    ).to(tl.int64)
    restore = restore & (physical >= 0)
    dst = safe.to(tl.int64) * STRIDE + tl.where(dense, pos, C + pos % 256)
    index = row.to(tl.int64) * STRIDE + offset
    tl.store(source + index, tl.where(restore, physical, 0), mask=offset < STRIDE)
    tl.store(destination + index, tl.where(restore, dst, 0), mask=offset < STRIDE)
    tl.store(valid + index, restore, mask=offset < STRIDE)
    if tile == 0:
        tl.store(next_begin + safe, begin, mask=active)
        tl.store(next_end + safe, length, mask=active)
        tl.store(next_first + safe, first, mask=active)


@triton.jit
def _current_metadata(
    requests,
    positions,
    token_valid,
    next_begin,
    next_end,
    next_first,
    old_begin,
    old_end,
    destination,
    valid,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(requests + i, mask=i < N, other=0).to(tl.int32)
    pos = tl.load(positions + i, mask=i < N, other=-1).to(tl.int32)
    live = tl.load(token_valid + i, mask=i < N, other=0) != 0
    live = live & (req > 0) & (req < ROWS)
    safe = tl.where(live, req, 0)
    begin = tl.load(next_begin + safe)
    end = tl.load(next_end + safe)
    first = tl.load(next_first + safe)
    copy = live & (begin >= 0) & (pos >= begin) & (pos < end)
    dst = safe.to(tl.int64) * STRIDE + tl.where(begin == 0, pos, C + pos % 256)
    tl.store(destination + i, tl.where(copy, dst, 0), mask=i < N)
    tl.store(valid + i, copy, mask=i < N)
    # Historical copies have completed on this stream. One current token per
    # request commits the extent; subsequent kernels observe its payload too.
    commit = live & (pos == first)
    tl.store(old_begin + safe, begin, mask=commit)
    tl.store(old_end + safe, tl.where(begin >= 0, end, -1), mask=commit)


def stage_resident_kv(
    *,
    current,
    source_indices,
    host,
    host_address,
    cache,
    req_to_token,
    requests,
    lengths,
    first_positions,
    token_requests,
    token_positions,
    token_valid,
    old_begin,
    old_end,
    next_extents,
    history_workspace,
    current_workspace,
    capacity,
    context_len,
    query_width,
    full_visible,
):
    """Stage current KV and refill only absent short/ring history from host."""
    batch = requests.numel()
    stride = cache.shape[1]
    hs, hd, hv = history_workspace
    nb, ne, nf = next_extents
    _history_metadata[(triton.cdiv(stride, 128), batch)](
        requests,
        lengths,
        first_positions,
        req_to_token,
        old_begin,
        old_end,
        nb,
        ne,
        nf,
        hs,
        hd,
        hv,
        old_begin.numel(),
        context_len,
        req_to_token.stride(0),
        first_positions.stride(0),
        capacity,
        stride,
        query_width,
        full_visible,
        128,
    )
    history_count = batch * stride
    unidex_copy_inplace(
        host,
        cache,
        hs[:history_count],
        hd[:history_count],
        hv[:history_count],
        2,
        2,
        block_dim=48,
        src_ptr=host_address,
    )
    count = current.shape[0]
    cd, cv = current_workspace
    _current_metadata[(triton.cdiv(count, 128),)](
        token_requests,
        token_positions,
        token_valid,
        nb,
        ne,
        nf,
        old_begin,
        old_end,
        cd,
        cv,
        count,
        old_begin.numel(),
        capacity,
        stride,
        128,
    )
    unidex_copy_inplace(
        current,
        cache,
        source_indices,
        cd[:count],
        cv[:count],
        1,
        2,
        block_dim=48,
    )
