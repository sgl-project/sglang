"""Fused sparse KV publication metadata and target KV layout on NPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _publication(
    reqs,
    seq,
    pos,
    loc,
    ext,
    prefix,
    source,
    dest,
    positions,
    valid,
    T: tl.constexpr,
    B: tl.constexpr,
    Q: tl.constexpr,
    REQ_ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    CAPACITY: tl.constexpr,
    REQ_S: tl.constexpr,
    SEQ_S: tl.constexpr,
    POS_S: tl.constexpr,
    LOC_S: tl.constexpr,
    EXT_S: tl.constexpr,
    PREFIX_S: tl.constexpr,
    PREFIX_N: tl.constexpr,
    PREFILL: tl.constexpr,
    COMPACT: tl.constexpr,
    PAD_WIDTH: tl.constexpr,
    B_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    t = tl.program_id(0) * T_BLOCK + tl.arange(0, T_BLOCK)
    in_bounds = t < T
    if PREFILL:
        if COMPACT:
            ids = tl.arange(0, B_BLOCK)
            lens = tl.load(ext + ids * EXT_S, ids < B, other=0)
            ends = tl.cumsum(lens, 0)
            row = tl.sum(
                tl.where((ids[None, :] < B) & (t[:, None] >= ends[None, :]), 1, 0), 1
            )
            start = tl.sum(tl.where(ids[None, :] < row[:, None], lens[None, :], 0), 1)
            offset = t - start
        else:
            row = t // PAD_WIDTH
            offset = t % PAD_WIDTH
        req = tl.load(reqs + row * REQ_S, (row < B) & in_bounds, other=0)
        length = tl.load(ext + row * EXT_S, (row < B) & in_bounds, other=0)
        pre = tl.load(prefix + row * PREFIX_S, (row < PREFIX_N) & in_bounds, other=0)
        position = (pre + offset).to(tl.int64)
        live = (req > 0) & (offset < length) & (position >= 0) & (position < CONTEXT)
    else:
        row = t // Q
        inside = (t < B * Q) & in_bounds
        req = tl.load(reqs + row * REQ_S, inside, other=0)
        length = tl.load(seq + row * SEQ_S, inside, other=0)
        position = tl.load(pos + t * POS_S, inside, other=-1).to(tl.int64)
        live = inside & (req > 0) & (req < REQ_ROWS) & (length > 0)
        live = live & (position >= 0) & (position < CONTEXT)
    location = tl.load(loc + t * LOC_S, in_bounds, other=-1).to(tl.int64)
    live = live & (location >= 0) & (location < CAPACITY)
    tl.store(source + t, t, in_bounds)
    tl.store(dest + t, location, in_bounds)
    tl.store(positions + t, position, in_bounds)
    tl.store(valid + t, live, in_bounds)


def prepare_sparse_kv_publication(
    k,
    k_rope,
    request_ids,
    seq_lens,
    positions,
    out_cache_loc,
    *,
    request_rows,
    context_len,
    host_capacity,
    query_width=1,
    extend_seq_lens=None,
    extend_prefix_lens=None,
    compact_prefill=False,
):
    """Prepare publication rows and, for target decode/verify, concatenate KV."""
    tokens, batch = k.shape[0], request_ids.numel()
    prefill = extend_seq_lens is not None
    source = torch.empty(tokens, dtype=torch.int64, device=k.device)
    destination = torch.empty_like(source)
    token_positions = torch.empty_like(source)
    valid = torch.empty(tokens, dtype=torch.bool, device=k.device)
    current = (k, k_rope) if prefill else torch.cat((k, k_rope), dim=-1)
    if tokens:
        token_block = 32
        _publication[(triton.cdiv(tokens, token_block),)](
            request_ids,
            seq_lens,
            positions if positions is not None else request_ids,
            out_cache_loc,
            extend_seq_lens if prefill else request_ids,
            extend_prefix_lens if prefill else request_ids,
            source,
            destination,
            token_positions,
            valid,
            tokens,
            batch,
            query_width,
            request_rows,
            context_len,
            host_capacity,
            request_ids.stride(0),
            seq_lens.stride(0),
            positions.stride(0) if positions is not None else 0,
            out_cache_loc.stride(0),
            extend_seq_lens.stride(0) if prefill else 0,
            extend_prefix_lens.stride(0) if prefill else 0,
            extend_prefix_lens.numel() if prefill else 0,
            prefill,
            compact_prefill,
            tokens // batch if prefill and not compact_prefill else 1,
            triton.next_power_of_2(batch),
            token_block,
        )
    return source, destination, token_positions, valid, current
