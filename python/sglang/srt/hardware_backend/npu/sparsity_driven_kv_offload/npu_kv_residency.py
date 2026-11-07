"""Keep short KV contiguous in HBM and restore only missing logical ranges."""

import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.npu_residency import (
    restore_short_kv_history_inplace,
)


@triton.jit
def _stage_current_kv(
    current,
    source_indices,
    cache,
    requests,
    positions,
    token_valid,
    next_begin,
    next_end,
    next_first,
    old_begin,
    old_end,
    ROWS: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    STRIDE: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    req = tl.load(requests + i).to(tl.int64)
    pos = tl.load(positions + i).to(tl.int64)
    live = (tl.load(token_valid + i) != 0) & (req > 0) & (req < ROWS)
    safe_req = tl.minimum(tl.maximum(req, 0), ROWS - 1)
    begin = tl.load(next_begin + safe_req)
    end = tl.load(next_end + safe_req)
    first = tl.load(next_first + safe_req)
    live = live & (req < TABLE_ROWS) & (first >= 0) & (end > 0)
    source = tl.load(source_indices + i).to(tl.int64)
    copy = (
        live
        & (begin >= 0)
        & (pos >= begin)
        & (pos < end)
        & (pos >= 0)
        & (pos < STRIDE)
        & (source >= 0)
        & (source < SOURCE_ROWS)
    )
    if copy:
        # Clamp before forming pointers, including for masked lanes.
        safe_source = tl.minimum(tl.maximum(source, 0), SOURCE_ROWS - 1)
        safe_pos = tl.minimum(tl.maximum(pos, 0), STRIDE - 1)
        feature = tl.arange(0, BLOCK)
        safe_feature = tl.minimum(feature, FEATURES - 1)
        value = tl.load(
            current + safe_source * SOURCE_STRIDE + safe_feature,
            mask=feature < FEATURES,
        )
        tl.store(
            cache + (safe_req * STRIDE + safe_pos) * FEATURES + safe_feature,
            value,
            mask=feature < FEATURES,
        )
    # Exactly one current token per request commits. All programs read immutable
    # next extents, so committing cannot race with another program's decisions.
    if live & (pos == first):
        tl.store(old_begin + safe_req, begin)
        tl.store(old_end + safe_req, tl.where(begin >= 0, end, -1))


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
    """Stage current KV and refill only absent short history from host.

    Real request IDs must be unique within a batch; padding ID 0 may repeat.
    """
    if current.dtype != cache.dtype or current.shape[1:] != cache.shape[2:]:
        raise ValueError(
            "Current KV rows must match the resident cache layout and dtype."
        )
    if requests.numel() == 0:
        return
    nb, ne, nf = next_extents
    restore_short_kv_history_inplace(
        host,
        cache,
        req_to_token,
        requests,
        lengths,
        first_positions,
        old_begin,
        old_end,
        nb,
        ne,
        nf,
        capacity,
        context_len,
        src_ptr=host_address,
        block_dim=48,
    )
    count = current.shape[0]
    if count == 0:
        return
    features = cache.shape[2] * cache.shape[3]
    _stage_current_kv[(count,)](
        current,
        source_indices,
        cache,
        token_requests,
        token_positions,
        token_valid,
        nb,
        ne,
        nf,
        old_begin,
        old_end,
        old_begin.numel(),
        req_to_token.shape[0],
        current.shape[0],
        current.stride(0),
        cache.shape[1],
        features,
        triton.next_power_of_2(features),
    )
