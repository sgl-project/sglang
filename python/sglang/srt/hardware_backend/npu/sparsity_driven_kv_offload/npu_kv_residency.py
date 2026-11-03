"""Keep short KV contiguous in HBM and restore only missing logical ranges."""

import triton
import triton.language as tl
from sgl_kernel_npu.sparsity_driven_kv_offload import unidex_copy_inplace


@triton.jit
def _restore_history(
    requests,
    lengths,
    first_positions,
    req_to_token,
    old_begin,
    old_end,
    next_begin,
    next_end,
    next_first,
    host_address,
    cache,
    ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    C: tl.constexpr,
    STRIDE: tl.constexpr,
    HOST_ROWS: tl.constexpr,
    FEATURES: tl.constexpr,
    FEATURE_BLOCK: tl.constexpr,
    OWNERS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    owner = tl.program_id(0)
    row = tl.program_id(1)
    req = tl.load(requests + row).to(tl.int32)
    length = tl.load(lengths + row).to(tl.int32)
    first = tl.load(first_positions + row * FIRST_STRIDE).to(tl.int32)
    active = (
        (req > 0) & (req < ROWS) & (length > 0) & (length <= CONTEXT) & (first >= 0)
    )
    safe = tl.where(active, req, 0)
    dense = length < C
    begin = tl.where(dense, 0, -1).to(tl.int32)
    if owner == 0:
        tl.store(next_begin + safe, begin, mask=active)
        tl.store(next_end + safe, length, mask=active)
        tl.store(next_first + safe, first, mask=active)
    # Evaluate on device every time, including graph replay.
    if active & dense:
        previous_begin = tl.load(old_begin + safe)
        previous_end = tl.load(old_end + safe)
        missing_start = tl.where(previous_begin == 0, tl.maximum(previous_end, 0), 0)
        feature = tl.arange(0, FEATURE_BLOCK)
        host = host_address.to(tl.pointer_type(cache.dtype.element_ty))
        for base in range(
            missing_start + owner * ROW_BLOCK,
            tl.minimum(first, STRIDE),
            OWNERS * ROW_BLOCK,
        ):
            pos = base + tl.arange(0, ROW_BLOCK)
            restore = (pos < STRIDE) & (pos < first)
            physical = tl.load(
                req_to_token + safe * REQUEST_STRIDE + pos, mask=restore, other=0
            ).to(tl.int64)
            restore = restore & (physical >= 0) & (physical < HOST_ROWS)
            src = physical[:, None] * FEATURES + feature[None, :]
            dst = (safe.to(tl.int64) * STRIDE + pos[:, None]) * FEATURES + feature[
                None, :
            ]
            mask = restore[:, None] & (feature[None, :] < FEATURES)
            value = tl.load(host + src, mask=mask, other=0)
            tl.store(cache + dst, value, mask=mask)


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
    dst = safe.to(tl.int64) * STRIDE + pos
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
):
    """Stage current KV and refill only absent short history from host."""
    batch = requests.numel()
    stride = cache.shape[1]
    nb, ne, nf = next_extents
    features = cache.numel() // (cache.shape[0] * stride)
    owners = max(1, triton.cdiv(48, batch))
    _restore_history[(owners, batch)](
        requests,
        lengths,
        first_positions,
        req_to_token,
        old_begin,
        old_end,
        nb,
        ne,
        nf,
        host_address,
        cache,
        old_begin.numel(),
        context_len,
        req_to_token.stride(0),
        first_positions.stride(0),
        capacity,
        stride,
        host.numel() // features,
        features,
        triton.next_power_of_2(features),
        owners,
        16,
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
