"""Validated sparse GQA operators migrated from the QSA reference branch."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    if is_batch_invariant_mode_enabled():
        return (32, 4, 2)
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _partition_cp_sparse_indices(
    indices,
    packed,
    offsets,
    cu_q,
    kv_lens,
    q_first_positions,
    prefix_lens,
    extend_starts,
    TOPK: tl.constexpr,
    GROUPS: tl.constexpr,
    SI_ROW: tl.constexpr,
    SI_GROUP: tl.constexpr,
    SI_COL: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    relative = tl.program_id(0)
    batch_group = tl.program_id(1)
    batch, group = batch_group // GROUPS, batch_group % GROUPS
    row = tl.load(cu_q + batch) + relative
    if row >= tl.load(cu_q + batch + 1):
        return
    cols = tl.arange(0, BLOCK)
    token = tl.load(
        indices + row * SI_ROW + group * SI_GROUP + cols * SI_COL,
        mask=cols < TOPK,
        other=-1,
    )
    prefix = tl.load(prefix_lens + batch)
    visible = tl.load(q_first_positions + batch) + relative * CP_SIZE + 1
    valid = (cols < TOPK) & (token >= 0) & (token < visible)
    valid &= token < tl.load(kv_lens + batch)
    owner = (tl.load(extend_starts + batch) + token - prefix) % CP_SIZE + 1
    owner = tl.where(token < prefix, 0, owner)
    output_base = (row * GROUPS + group) * TOPK
    offset_base = (row * GROUPS + group) * (CP_SIZE + 2)
    start = 0
    for partition in tl.static_range(CP_SIZE + 1):
        selected = valid & (owner == partition)
        position = tl.cumsum(selected.to(tl.int32)) - 1
        tl.store(packed + output_base + start + position, token, mask=selected)
        tl.store(offsets + offset_base + partition, start)
        start += tl.sum(selected.to(tl.int32), 0)
    tl.store(offsets + offset_base + CP_SIZE + 1, start)


def partition_cp_sparse_indices(
    indices,
    cu_q,
    kv_lens,
    q_first_positions,
    prefix_lens,
    extend_starts,
    *,
    cp_size,
    max_q,
):
    """Compact each query's selection into prefix and new-KV owner segments.

    Preserve logical token positions and selection order within each segment;
    ring steps then iterate only their selected keys, including empty segments.
    """
    if cp_size < 2 or indices.ndim not in (2, 3):
        raise ValueError("CP sparse selection requires CP > 1 and 2D/3D indices")
    rows, topk = indices.shape[0], indices.shape[-1]
    groups = indices.shape[1] if indices.ndim == 3 else 1
    packed = torch.empty(indices.shape, dtype=indices.dtype, device=indices.device)
    offsets = torch.empty(
        (rows, groups, cp_size + 2), dtype=torch.int32, device=indices.device
    )
    if rows and max_q:
        _partition_cp_sparse_indices[(max_q, (cu_q.numel() - 1) * groups)](
            indices,
            packed,
            offsets,
            cu_q,
            kv_lens,
            q_first_positions,
            prefix_lens,
            extend_starts,
            TOPK=topk,
            GROUPS=groups,
            SI_ROW=indices.stride(0),
            SI_GROUP=indices.stride(1) if indices.ndim == 3 else 0,
            SI_COL=indices.stride(-1),
            CP_SIZE=cp_size,
            BLOCK=triton.next_power_of_2(topk),
            num_warps=4,
        )
    return packed, offsets


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    lse,
    q_first_positions,
    prefix_lens,
    extend_starts,
    kv_owner,
    partition_offsets,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    PARTITION_GROUPS: tl.constexpr,
    PARTITIONED: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SLOT_INDICES: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_MODE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """One query row x one KV head group over its ``topk`` selected tokens.

    ``indices`` hold positions relative to row ``cu_k[batch]`` of ``k``/``v``
    (a packed context), or, with ``SLOT_INDICES``, rows of ``k``/``v`` itself
    (KV-pool slots; ``cu_k`` is unused)."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    if SLOT_INDICES:
        k_start = 0
    else:
        k_start = tl.load(cu_k + batch).to(tl.int64)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    if CP_MODE:
        visible = tl.load(q_first_positions + batch) + query_relative * CP_SIZE + 1
        prefix_len = tl.load(prefix_lens + batch)
    else:
        visible = query_relative + kv_len - (q_end - q_start) + 1
    selection_start = 0
    if PARTITIONED:
        partition = 0 if CP_MODE == 1 else kv_owner + 1
        offset_group = group if PARTITION_GROUPS > 1 else 0
        offset_base = (query * PARTITION_GROUPS + offset_group) * (CP_SIZE + 2)
        selection_start = tl.load(partition_offsets + offset_base + partition)
        selection_end = tl.load(partition_offsets + offset_base + partition + 1)
        row_topk = selection_end - selection_start
    else:
        row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g + selection_start * si_n
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        index_limit = row_topk if PARTITIONED else topk
        token = tl.load(idx_row + current * si_n, mask=current < index_limit, other=-1)
        valid = token >= 0
        if SLOT_INDICES:
            token = token.to(tl.int64)
        else:
            valid = (token >= 0) & (token < visible) & (token < kv_len)
            if CP_MODE == 1:
                valid &= token < prefix_len
            elif CP_MODE == 2:
                packed_token = tl.load(extend_starts + batch) + token - prefix_len
                valid &= (token >= prefix_len) & (packed_token % CP_SIZE == kv_owner)
                token = packed_token // CP_SIZE
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # Direct callers and the KV pool can be FP8; packing uses Q's dtype.
        # QSA stores unscaled K/V, so a plain cast supports both callers.
        keys = keys.to(q_values.dtype)
        values = values.to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.math.exp2(max_value - safe_max)
        probabilities = tl.math.exp2(scores - safe_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / tl.where(normalizer > 0, normalizer, 1.0)[:, None]
    if RETURN_LSE:
        row_lse = tl.where(
            normalizer > 0,
            (max_value + tl.log2(normalizer)) * 0.6931471805599453,
            -float("inf"),
        )
        tl.store(
            lse + query * (NUM_KV_HEADS * GROUP_SIZE) + group * GROUP_SIZE + offs_h,
            row_lse,
            mask=offs_h < GROUP_SIZE,
        )
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


@triton.jit(do_not_specialize=["TOTAL_K", "BS", "new_rank_stride"])
def _pack_qsa_prefill_kv(
    k,
    v,
    packed_k,
    packed_v,
    req_to_token,
    req_indices,
    cu_k,
    new_k,
    new_v,
    prefix_lens,
    extend_starts,
    new_token_indices,
    cache_loc,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SR0: tl.constexpr,
    SR1: tl.constexpr,
    SNK0: tl.constexpr,
    SNK1: tl.constexpr,
    SNK2: tl.constexpr,
    SNV0: tl.constexpr,
    SNV1: tl.constexpr,
    SNV2: tl.constexpr,
    NEW_CP_SIZE: tl.constexpr,
    new_rank_stride,
    NEW_TOKEN_INDICES: tl.constexpr,
    STORE_NEW_KV: tl.constexpr,
    CACHE_IS_FP8: tl.constexpr,
    CACHE_MAX: tl.constexpr,
    HAS_NEW_KV: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    TOTAL_K,
    BS,
    COMPACT: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (HEADS * DIM)
    if COMPACT:
        valid = token < TOTAL_K
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = lo + BS
        steps = BS
        while steps > 0:
            mid = (lo + hi + 1) // 2
            start = tl.load(cu_k + mid).to(tl.int64)
            after = token >= start
            lo = tl.where(after, mid, lo)
            hi = tl.where(after, hi, mid - 1)
            steps = steps // 2
        batch = lo
        start = tl.load(cu_k + lo, valid, 0).to(tl.int64)
        req = tl.load(req_indices + lo, valid, 0).to(tl.int64)
        token = token - start
        dst = offsets
    else:
        batch = tl.program_id(1)
        start = tl.load(cu_k + batch).to(tl.int64)
        end = tl.load(cu_k + batch + 1).to(tl.int64)
        valid = token < end - start
        req = tl.load(req_indices + batch).to(tl.int64)
        dst = start * HEADS * DIM + offsets
    head = offsets // DIM % HEADS
    dim = offsets % DIM
    cached = valid
    if HAS_NEW_KV:
        if COMPACT:
            prefix = tl.load(prefix_lens + batch, valid, 0)
            extend_start = tl.load(extend_starts + batch, valid, 0)
        else:
            prefix = tl.load(prefix_lens + batch)
            extend_start = tl.load(extend_starts + batch)
        cached &= token < prefix
    slot = tl.load(req_to_token + req * SR0 + token * SR1, cached, 0).to(tl.int64)
    keys = tl.load(k + slot * SK0 + head * SK1 + dim * SK2, cached, 0.0)
    values = tl.load(v + slot * SV0 + head * SV1 + dim * SV2, cached, 0.0)
    if HAS_NEW_KV:
        source = extend_start + token - prefix
        fresh = valid & ~cached
        if NEW_TOKEN_INDICES:
            source_row = tl.load(new_token_indices + source, fresh, 0).to(tl.int64)
        else:
            source_row = (
                source % NEW_CP_SIZE
            ) * new_rank_stride + source // NEW_CP_SIZE
        current_k = tl.load(
            new_k + source_row * SNK0 + head * SNK1 + dim * SNK2, fresh, 0.0
        )
        current_v = tl.load(
            new_v + source_row * SNV0 + head * SNV1 + dim * SNV2, fresh, 0.0
        )
        if STORE_NEW_KV:
            dst_slot = tl.load(cache_loc + source, fresh, 0).to(tl.int64)
            stored_k, stored_v = current_k, current_v
            if CACHE_IS_FP8:
                # Match the pool's unscaled FP8 clamp and cast.
                stored_k = tl.minimum(
                    tl.maximum(stored_k.to(tl.float32), -CACHE_MAX), CACHE_MAX
                )
                stored_v = tl.minimum(
                    tl.maximum(stored_v.to(tl.float32), -CACHE_MAX), CACHE_MAX
                )
            tl.store(k + dst_slot * SK0 + head * SK1 + dim * SK2, stored_k, fresh)
            tl.store(v + dst_slot * SV0 + head * SV1 + dim * SV2, stored_v, fresh)
        keys = tl.where(cached, keys, current_k)
        values = tl.where(cached, values, current_v)
    tl.store(packed_k + dst, keys, valid)
    tl.store(packed_v + dst, values, valid)


def pack_qsa_prefill_kv(
    *,
    k,
    v,
    req_to_token,
    req_indices,
    cu_k,
    total_k,
    max_k,
    output_dtype=None,
    new_kv=None,
    prefix_lens=None,
    extend_starts=None,
    new_kv_cp_size=1,
    new_kv_token_indices=None,
    cache_loc=None,
):
    """Pack prefix and fresh KV, optionally restoring CP order and storing cache.

    Fresh KV may be strided views of one rank-major all-gather buffer. The
    attention output keeps its original precision; cache writes follow pool
    dtype and saturation without a quantize/dequantize round trip.
    """
    heads, dim = k.shape[1:]
    if new_kv_cp_size < 1:
        raise ValueError("New KV CP size must be positive")
    if new_kv is None and (
        new_kv_cp_size != 1 or new_kv_token_indices is not None or cache_loc is not None
    ):
        raise ValueError("KV reordering and cache writes require fresh KV")
    if new_kv is not None:
        if len(new_kv) != 2 or new_kv[0].shape != new_kv[1].shape:
            raise ValueError("New K/V must have matching token/head dimensions")
        if prefix_lens is None or extend_starts is None:
            raise ValueError(
                "New KV packing requires per-request prefix/extend offsets"
            )
        if any(
            x.ndim != 3 or x.shape[1:] != k.shape[1:] or x.device != k.device
            for x in new_kv
        ):
            raise ValueError("New KV must match cached KV heads/device")
        if new_kv[0].shape[0] % new_kv_cp_size:
            raise ValueError("Rank-major KV must contain equally padded CP shards")
        for indices in (new_kv_token_indices, cache_loc):
            if indices is not None and (
                indices.ndim != 1
                or indices.dtype not in (torch.int32, torch.int64)
                or indices.device != k.device
                or not indices.is_contiguous()
            ):
                raise ValueError(
                    "KV source indices and cache slots must be contiguous integer vectors"
                )
    packed_k = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or k.dtype, device=k.device
    )
    packed_v = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or v.dtype, device=v.device
    )
    batch_size = req_indices.numel()
    blocks_per_request = triton.cdiv(max_k * heads * dim, 1024)
    compact = (
        max_k * batch_size > 2 * total_k and blocks_per_request * batch_size >= 4096
    )
    grid = (
        (triton.cdiv(total_k * heads * dim, 1024),)
        if compact
        else (blocks_per_request, batch_size)
    )
    _pack_qsa_prefill_kv[grid](
        k,
        v,
        packed_k,
        packed_v,
        req_to_token,
        req_indices,
        cu_k,
        new_kv[0] if new_kv is not None else None,
        new_kv[1] if new_kv is not None else None,
        prefix_lens,
        extend_starts,
        new_kv_token_indices,
        cache_loc,
        *k.stride(),
        *v.stride(),
        *req_to_token.stride(),
        *(new_kv[0].stride() if new_kv is not None else (0, 0, 0)),
        *(new_kv[1].stride() if new_kv is not None else (0, 0, 0)),
        NEW_CP_SIZE=new_kv_cp_size,
        new_rank_stride=new_kv[0].shape[0] // new_kv_cp_size
        if new_kv is not None
        else 0,
        NEW_TOKEN_INDICES=new_kv_token_indices is not None,
        STORE_NEW_KV=cache_loc is not None,
        CACHE_IS_FP8=k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2),
        CACHE_MAX=torch.finfo(k.dtype).max,
        HAS_NEW_KV=new_kv is not None,
        HEADS=heads,
        DIM=dim,
        BLOCK=1024,
        TOTAL_K=total_k,
        BS=batch_size,
        COMPACT=compact,
    )
    return packed_k, packed_v


def sparse_gqa_fwd_interface_triton_ck(
    q,
    k,
    v,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    max_q=None,
    *,
    cp_size=1,
    cp_mode=0,
    kv_owner=0,
    q_first_positions=None,
    prefix_lens=None,
    extend_starts=None,
    return_lse=False,
    partition_offsets=None,
):
    # CP modes: 0 = local Q/KV, 1 = prefix KV, 2 = owner-sharded KV,
    # 3 = interleaved Q with globally ordered KV. All retain causal masking.
    if cp_mode not in (0, 1, 2, 3) or cp_size < 1:
        raise ValueError("Invalid sparse CP shard mode or group size")
    if cp_mode and any(
        x is None for x in (q_first_positions, prefix_lens, extend_starts)
    ):
        raise ValueError("Sparse CP requires query positions and prefix/extend offsets")
    if partition_offsets is not None and cp_mode not in (1, 2):
        raise ValueError(
            "Partitioned selection requires prefix-only or owner-sharded KV"
        )
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if partition_offsets is not None:
        groups = indices.shape[1] if indices.ndim == 3 else 1
        if (
            partition_offsets.shape != (total_q, groups, cp_size + 2)
            or partition_offsets.dtype != torch.int32
            or partition_offsets.device != q.device
            or not partition_offsets.is_contiguous()
            or groups not in (1, num_kv_heads)
        ):
            raise ValueError("CP selection offsets must match query rows and KV groups")
    group_size = num_q_heads // num_kv_heads
    if max_q is None:
        # Host sync; callers that know the longest entry pass it instead.
        max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q, dtype=torch.float32 if return_lse else q.dtype)
    lse = (
        torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    if total_q == 0:
        return (out, lse) if return_lse else out
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        lse,
        q_first_positions,
        prefix_lens,
        extend_starts,
        kv_owner,
        partition_offsets,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        PARTITION_GROUPS=indices.shape[1] if indices.ndim == 3 else 1,
        PARTITIONED=partition_offsets is not None,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        SLOT_INDICES=False,
        CP_SIZE=cp_size,
        CP_MODE=cp_mode,
        RETURN_LSE=return_lse,
        num_warps=warps,
        num_stages=stages,
    )
    return (out, lse) if return_lse else out


def warmup_cp_sparse_attention(
    *, device, dtype, q_heads, kv_heads, head_dim, topk, cp_size, max_local_tokens
):
    """Compile gathered CP sparse attention and KV packing using scratch tensors."""
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    sizes = {min(int(limit), max_local_tokens) for limit, _ in table[:-1]}
    sizes.add(max_local_tokens)
    for rows in sorted(sizes):
        q = torch.zeros((rows, q_heads, head_dim), device=device, dtype=dtype)
        k = torch.zeros((1, kv_heads, head_dim), device=device, dtype=dtype)
        indices = torch.full((rows, topk), -1, device=device, dtype=torch.int32)
        cu_q = torch.tensor([0, rows], device=device, dtype=torch.int32)
        zero = torch.zeros(1, device=device, dtype=torch.int32)
        lengths = torch.ones(1, device=device, dtype=torch.int32)
        sparse_gqa_fwd_interface_triton_ck(
            q,
            k,
            k,
            indices,
            cu_q,
            zero,
            lengths,
            head_dim**-0.5,
            max_q=rows,
            cp_size=cp_size,
            cp_mode=3,
            q_first_positions=zero,
            prefix_lens=zero,
            extend_starts=zero,
        )
    for cache_dtype in (dtype, torch.float8_e4m3fn):
        cached = torch.zeros((2, kv_heads, head_dim), device=device, dtype=cache_dtype)
        new = torch.zeros((cp_size, 2, kv_heads, head_dim), device=device, dtype=dtype)
        for source_indices in (None, torch.zeros(1, device=device, dtype=torch.int64)):
            pack_qsa_prefill_kv(
                k=cached,
                v=cached,
                req_to_token=torch.tensor([[0, 1]], device=device, dtype=torch.int32),
                req_indices=torch.zeros(1, device=device, dtype=torch.int32),
                cu_k=torch.tensor([0, 2], device=device, dtype=torch.int32),
                total_k=2,
                max_k=2,
                output_dtype=dtype,
                new_kv=(new[:, 0], new[:, 1]),
                prefix_lens=torch.ones(1, device=device, dtype=torch.int32),
                extend_starts=torch.zeros(1, device=device, dtype=torch.int32),
                new_kv_cp_size=cp_size,
                new_kv_token_indices=source_indices,
                cache_loc=torch.ones(1, device=device, dtype=torch.int64),
            )


def sparse_gqa_packed_decode_triton(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """Run one packed sparse-attention row per request without a host sync.

    Reuses the chunk-prefill kernel at a fixed query length of one, so graph
    capture never hits the ``.item()`` that the general interface needs to
    derive ``max_q``.
    """

    return sparse_gqa_fwd_interface_triton_ck(
        q,
        k,
        v,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        max_q=1,
    )


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # One request per program: Triton caps a tile at 1M elements,
    # which [next_pow2(topk), next_pow2(batch)] exceeds at topk=2051, batch=512.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    pad_cols,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ZERO_FILL: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    # 64-bit element offsets: slot * heads * dim exceeds int32 once the pool holds
    # more than 2^31 / (heads * dim) tokens (~4.2M for 2 x 256), which an FP8 pool
    # on one GPU does reach.
    src = slots.to(tl.int64)[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (
        (pack_start + cols).to(tl.int64)[:, None] * heads * dim
        + head * dim
        + dims[None, :]
    )
    load_mask = valid[:, None] & (dims[None, :] < dim)
    if ZERO_FILL:
        # Strided (page-aligned) packing: the paged decode kernel reads whole pages,
        # so every slot in [valid_count, pad_cols) must hold zeros, never stale bytes.
        # `valid_count` here is the row's page-aligned stride, not its valid count, so
        # the store covers the full region while the load stays limited to valid rows.
        store_mask = (cols < pad_cols)[:, None] & (dims[None, :] < dim)
    else:
        store_mask = load_mask
    # Dequantize while gathering: the scratch is allocated in the query dtype, so an
    # FP8 pool is read as fp8 and stored as bf16. The QSA backend writes the pool
    # without per-tensor k/v scales (see set_kv_buffer calls in
    # qwen_sparse_attn_backend.py), so no scale is applied here either.
    out_dtype = out_k.dtype.element_ty
    tl.store(
        out_k + dst,
        tl.load(k + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )
    tl.store(
        out_v + dst,
        tl.load(v + src, mask=load_mask, other=0.0).to(out_dtype),
        mask=store_mask,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, without the packed cu_seqlens prefix sum."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    zero_fill_cols: int = 0,
):
    """Gather the selected K/V rows into ``out_k``/``out_v``.

    ``zero_fill_cols`` > 0 selects the strided (page-aligned) layout used by the paged
    decode kernel: row ``b`` owns ``[cu_k[b], cu_k[b] + zero_fill_cols)`` and every slot
    past its valid rows is zero-filled. Paged kernels read whole pages and multiply the
    masked probabilities into V, so stale or uninitialized bytes there (NaN/Inf bit
    patterns) would otherwise leak into the output. ``0`` keeps the compact layout for
    the varlen fallback, whose rows are packed back-to-back.

    ``out_k``/``out_v`` may use a wider dtype than the pool (bf16 scratch for an FP8
    pool); rows are converted while gathering.

    Both layouts assume the valid entries of each ``indices`` row are contiguous at
    the front (``expand_qsa_block_indices`` sorts them that way): ``valid_count`` is a
    count, not a mask, so a ``-1`` in the middle of a row would shift the packing.
    """
    _, heads, dim = k.shape
    block_topk = 16
    zero_fill = zero_fill_cols > 0
    num_cols = zero_fill_cols if zero_fill else topk
    _compact_kv[(batch, heads, triton.cdiv(num_cols, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        num_cols,
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        ZERO_FILL=zero_fill,
        num_warps=8,
    )


@triton.jit
def _resolve_prefill_slots(
    indices,
    slots,
    cu_q,
    slot_table,
    slot_rows,
    topk,
    si_m,
    ss_m,
    st_m,
    BLOCK: tl.constexpr,
):
    """``slots[row, j] = slot_table[entry % slot_rows, indices[row, j]]``, -1
    where the index is; one program per query row of an entry."""
    entry = tl.program_id(1)
    q_start = tl.load(cu_q + entry)
    row = q_start + tl.program_id(0)
    if row >= tl.load(cu_q + entry + 1):
        return
    row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    token = tl.load(indices + row * si_m + cols, mask=cols < topk, other=-1)
    valid = token >= 0
    table_row = slot_table + (entry % slot_rows).to(tl.int64) * st_m
    slot = tl.load(table_row + token, mask=valid, other=-1)
    tl.store(slots + row * ss_m + cols, slot, mask=cols < topk)


def sparse_gqa_paged_prefill_triton(
    q, k_cache, v_cache, slot_table, indices, cu_q, kv_lens, scale, *, max_q
):
    """Chunk-prefill sparse attention that reads K/V in place from the KV pool.

    Same entry geometry as ``sparse_gqa_fwd_interface_triton_ck`` (``cu_q`` and
    ``kv_lens`` per entry, ``indices`` are logical positions of the entry's
    request), but no packed copy of each request's context: entry ``e``'s
    position ``t`` is read from pool slot
    ``slot_table[e % slot_table.shape[0], t]``. Prefill CP passes two entries
    per request (zigzag prev blocks, then next blocks) over one table row per
    request. ``max_q`` is the longest entry's query count, from host lengths.
    The selected tokens, the current chunk's included, must already be in the
    pool.

    The slots are resolved in a pass of their own: looking them up inside the
    attention loop puts a second dependent load in front of every K/V load and
    made the kernel 1.5x slower."""
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    num_entries = cu_q.shape[0] - 1
    out = torch.empty_like(q)
    if total_q == 0 or num_entries == 0:
        return out
    indices = indices[:total_q]
    topk = indices.shape[-1]
    assert indices.ndim == 2 and indices.stride(-1) == 1
    assert slot_table.stride(-1) == 1
    slots = torch.empty((total_q, topk), dtype=slot_table.dtype, device=indices.device)
    _resolve_prefill_slots[(max_q, num_entries)](
        indices,
        slots,
        cu_q,
        slot_table,
        slot_table.shape[0],
        topk,
        indices.stride(0),
        slots.stride(0),
        slot_table.stride(0),
        BLOCK=triton.next_power_of_2(topk),
        num_warps=4,
    )
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    _sparse_gqa_chunk_prefill[(max_q, num_entries * num_kv_heads)](
        q,
        k_cache,
        v_cache,
        out,
        slots,
        cu_q,
        cu_q,
        kv_lens,
        scale,
        topk,
        None,
        None,
        None,
        None,
        0,
        None,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        slots.stride(0),
        0,
        slots.stride(1),
        PARTITION_GROUPS=1,
        PARTITIONED=False,
        CP_SIZE=1,
        CP_MODE=0,
        RETURN_LSE=False,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        SLOT_INDICES=True,
        num_warps=warps,
        num_stages=stages,
    )
    return out


__all__ = [
    "pack_qsa_prefill_kv",
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
    "sparse_gqa_packed_decode_triton",
    "sparse_gqa_paged_prefill_triton",
]
