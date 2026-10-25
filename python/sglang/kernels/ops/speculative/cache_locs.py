from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    next_power_of_2,
)

_is_cpu = is_cpu()
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()

if _is_cpu:
    from sgl_kernel import (
        assign_draft_cache_locs_contiguous_cpu,
        assign_extend_cache_locs_cpu,
    )


def _get_oot_speculative_cache_locs_fn() -> Callable[..., torch.Tensor] | None:
    from sglang.srt.platforms import current_platform

    if not current_platform.is_out_of_tree():
        return None
    return current_platform.get_speculative_cache_locs_fn()


@triton.jit(do_not_specialize=["BS"])
def assign_draft_cache_locs_contiguous(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    positions=None,
    mrope=None,
    BS=0,
    WRITE_POSITIONS: tl.constexpr = False,
    WRITE_MROPE: tl.constexpr = False,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)

    if WRITE_POSITIONS:
        topk_offset = tl.arange(0, BLOCK_SIZE)
        tl.store(positions + pid * topk + topk_offset, kv_start, topk_offset < topk)
        if WRITE_MROPE:
            for axis in tl.static_range(3):
                tl.store(
                    mrope + axis * BS * topk + pid * topk + topk_offset,
                    kv_start,
                    topk_offset < topk,
                )


def assign_draft_cache_locs_contiguous_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,  # mutable
    topk: int,
    num_cols: int,
    positions: Optional[torch.Tensor] = None,  # mutable
    mrope: Optional[torch.Tensor] = None,  # mutable
):
    """Copy each request's ``topk * num_cols`` slots from ``seq_lens`` on, and
    write the draft positions when ``positions`` is given."""
    if _is_cpu:
        assert positions is None and mrope is None, "CPU writes no positions"
        assign_draft_cache_locs_contiguous_cpu(
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            req_to_token.shape[1],
            topk,
            num_cols,
        )
        return
    bs = seq_lens.shape[0]
    assign_draft_cache_locs_contiguous[(bs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.shape[1],
        topk,
        num_cols,
        positions=positions,
        mrope=mrope,
        BS=bs,
        WRITE_POSITIONS=positions is not None,
        WRITE_MROPE=mrope is not None,
    )


@triton.jit
def _translate_token_ids(ids, v2p, mask, page_size: tl.constexpr):
    """Virtual token ids -> physical ones through the page-level v2p table,
    as `translate_kv_loc` does. A negative id maps to the page-0 sink instead
    of loading before `v2p`."""
    i64 = ids.to(tl.int64)
    vpage = tl.where(i64 < 0, 0, i64 // page_size)
    phys = tl.load(v2p + vpage, mask=mask, other=0)
    return tl.where(i64 < 0, 0, tl.maximum(phys * page_size + i64 % page_size, 0))


@triton.jit
def _load_token_ids(
    row_ptr,
    pos,
    mask,
    v2p,
    page_size: tl.constexpr,
    ENTRY_PAGE_SIZE: tl.constexpr,
    TRANSLATE: tl.constexpr,
):
    """Token ids at positions ``pos`` of one read-source row: the row's own
    entries when token-granular, ``entry * ps + pos % ps`` when page-granular;
    with ``TRANSLATE``, virtual ids mapped through ``v2p`` on the way out."""
    if ENTRY_PAGE_SIZE == 1:
        ids = tl.load(row_ptr + pos, mask=mask)
    else:
        entry = tl.load(row_ptr + pos // ENTRY_PAGE_SIZE, mask=mask, other=0)
        ids = entry.to(tl.int64) * ENTRY_PAGE_SIZE + pos % ENTRY_PAGE_SIZE
    if TRANSLATE:
        ids = _translate_token_ids(ids, v2p, mask, page_size)
    return ids


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    # Runtime, not constexpr: the unified pool's read table is as wide as the
    # iteration's longest row, so a constexpr stride would recompile per width.
    row_stride,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
    window_size: tl.constexpr = 0,
    sink_size: tl.constexpr = 0,
    NUM_STEPS: tl.constexpr = 0,
    ENTRY_PAGE_SIZE: tl.constexpr = 1,
    # The source's page table (`KVIndexTable.v2p`), applied with TRANSLATE.
    v2p=None,
    TRANSLATE: tl.constexpr = False,
):
    # window_size > 0 restricts the draft (not the target) to sink_size prefix
    # tokens + the most-recent window_size; window_size == 0 is the identity.
    BLOCK_SIZE: tl.constexpr = 128 if NUM_STEPS == 0 else 512
    pid0 = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    if NUM_STEPS == 0:
        iters = pid0
        num_steps = tl.num_programs(axis=0)
        blk = 0
    else:
        iters = pid0 % NUM_STEPS
        blk = pid0 // NUM_STEPS
        num_steps = NUM_STEPS
        num_blk = tl.num_programs(axis=0) // NUM_STEPS
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    if window_size > 0:
        cap = window_size + sink_size
        seq_lens = tl.minimum(seq_lens, cap)
        seq_len_w = tl.minimum(seq_len, cap)
        s_eff = tl.minimum(sink_size, seq_len)
        recent_start = seq_len - (seq_len_w - s_eff)
    else:
        seq_len_w = seq_len
        s_eff = 0
        recent_start = 0
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len_w + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * row_stride

    num_loop = tl.cdiv(seq_len_w, BLOCK_SIZE)
    if NUM_STEPS != 0 and blk >= num_loop and blk > 0:
        return
    if NUM_STEPS == 0:
        copy_offset = tl.arange(0, BLOCK_SIZE)
        for _ in range(num_loop):
            mask = copy_offset < seq_len_w
            src = tl.where(
                copy_offset < s_eff,
                copy_offset,
                recent_start + copy_offset - s_eff,
            )
            data = _load_token_ids(
                token_pool_ptr, src, mask, v2p, page_size, ENTRY_PAGE_SIZE, TRANSLATE
            )
            tl.store(kv_ptr + copy_offset, data, mask=mask)
            copy_offset += BLOCK_SIZE
    else:
        for i in range(blk, num_loop, num_blk):
            copy_offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = copy_offset < seq_len_w
            src = tl.where(
                copy_offset < s_eff,
                copy_offset,
                recent_start + copy_offset - s_eff,
            )
            data = _load_token_ids(
                token_pool_ptr, src, mask, v2p, page_size, ENTRY_PAGE_SIZE, TRANSLATE
            )
            tl.store(kv_ptr + copy_offset, data, mask=mask)

    # Extension entries and kv_indptr belong to token block 0 alone; other
    # blocks neither compute nor store them.
    if blk == 0:
        extend_offset = tl.arange(0, iter_upper)
        if page_size == 1 or topk == 1:
            extend_data = _load_token_ids(
                token_pool_ptr,
                seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
                extend_offset < iters,
                v2p,
                page_size,
                ENTRY_PAGE_SIZE,
                TRANSLATE,
            )
        else:
            prefix_len = seq_len
            last_page_len = prefix_len % page_size
            num_new_pages_per_topk = (
                last_page_len + num_steps + page_size - 1
            ) // page_size
            prefix_base = seq_len // page_size * page_size
            start = (
                prefix_base
                + topk_id * num_new_pages_per_topk * page_size
                + last_page_len
            )
            extend_data = _load_token_ids(
                token_pool_ptr,
                start + extend_offset,
                extend_offset < iters,
                v2p,
                page_size,
                ENTRY_PAGE_SIZE,
                TRANSLATE,
            )

        tl.store(
            kv_ptr + seq_len_w + extend_offset,
            extend_data,
            mask=extend_offset < iters,
        )

        # Update kv_indptr
        bs_offset = tl.arange(0, num_tokens_upper)
        zid = bid * topk + topk_id
        if zid == 0:
            zid = num_seqs * topk
        pos_vals = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
        if window_size > 0:
            pos_vals = tl.minimum(pos_vals, window_size + sink_size)
        base = tl.sum(pos_vals)
        tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def generate_draft_decode_window_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    window_kv_indices,
    window_kv_indptr,
    positions,
    # Runtime, as in generate_draft_decode_kv_indices.
    row_stride,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
    window: tl.constexpr,
    ENTRY_PAGE_SIZE: tl.constexpr = 1,
    # The source's sliding-window page table (`KVIndexTable.v2p`).
    v2p=None,
    TRANSLATE: tl.constexpr = False,
):
    """Per-step sliding-window read rail of the multi-step draft.

    Step ``iters`` of branch ``(bid, topk_id)`` attends ``seq_len + iters``
    tokens: the committed prefix plus the draft's own chain in the slots
    `generate_draft_decode_kv_indices` emits. A window layer reads the last
    ``min(seq_len + iters, window)`` of them, so this emits that tail only,
    prefix ids first and chain ids after, from the sliding-window read source.
    """
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    window_kv_indices += kv_indices_stride * iters
    window_kv_indptr += kv_indptr_stride * iters
    iters += 1

    seq_len = tl.load(paged_kernel_lens + bid)
    total = seq_len + iters
    window_len = tl.minimum(total, window)
    # Chain position of the first token in the window; the prefix part covers
    # [start, seq_len), the chain part [max(start - seq_len, 0), iters).
    start = total - window_len

    # This branch's output offset: the window lengths of the branches before
    # it, each clipped like this one.
    zid = bid * topk + topk_id
    bs_offset = tl.arange(0, num_tokens_upper)
    in_front = bs_offset < zid
    prev_lens = tl.load(positions + bs_offset, mask=in_front, other=0)
    prev_window = tl.where(in_front, tl.minimum(prev_lens + iters, window), 0)
    base = tl.sum(prev_window)
    out_ptr = window_kv_indices + base
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * row_stride

    prefix_start = tl.minimum(start, seq_len)
    prefix_len = seq_len - prefix_start
    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < prefix_len
        data = _load_token_ids(
            token_pool_ptr,
            prefix_start + kv_offset,
            mask,
            v2p,
            page_size,
            ENTRY_PAGE_SIZE,
            TRANSLATE,
        )
        tl.store(out_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    chain_start = tl.maximum(start - seq_len, 0)
    extend_offset = tl.arange(0, iter_upper)
    chain_step = chain_start + extend_offset
    chain_mask = chain_step < iters
    # The same chain slots `generate_draft_decode_kv_indices` emits.
    if page_size == 1 or topk == 1:
        chain_pos = seq_len + topk_id * num_steps + chain_step
    else:
        last_page_len = seq_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        chain_base = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        chain_pos = chain_base + chain_step
    extend_data = _load_token_ids(
        token_pool_ptr,
        chain_pos,
        chain_mask,
        v2p,
        page_size,
        ENTRY_PAGE_SIZE,
        TRANSLATE,
    )
    tl.store(out_ptr + prefix_len + extend_offset, extend_data, mask=chain_mask)

    # Update window_kv_indptr: program 0 writes the total at [num_seqs * topk],
    # every other program its own start; [0] stays the buffer's zero.
    if zid == 0:
        zid = num_seqs * topk
        in_front = bs_offset < zid
        prev_lens = tl.load(positions + bs_offset, mask=in_front, other=0)
        prev_window = tl.where(in_front, tl.minimum(prev_lens + iters, window), 0)
        base = tl.sum(prev_window)
    tl.store(window_kv_indptr + zid, base)


@triton.jit
def rebuild_compact_draft_req_to_token(
    draft_req_to_token,
    target_req_to_token,
    req_pool_indices,
    suffix_start,
    draft_prefix_lens,
    verify_out_cache_loc,
    verify_loc_stride,
    draft_pool_len: tl.constexpr,
    target_pool_len: tl.constexpr,
    block_size: tl.constexpr,
):
    """Rebuild one request's draft-local compact req->token row in a single pass.

    Row layout written: [0, prefix_len) = the committed target suffix window
    (target_req_to_token[req, suffix_start : suffix_start + prefix_len]) and
    [prefix_len, prefix_len + block_size) = the verify block slots. Fixed grid,
    per-row data-dependent loop bound; no host reads, so the caller never syncs.
    """
    BLOCK: tl.constexpr = 256
    pid = tl.program_id(axis=0)
    req = tl.load(req_pool_indices + pid).to(tl.int64)
    start = tl.load(suffix_start + pid).to(tl.int64)
    prefix_len = tl.load(draft_prefix_lens + pid).to(tl.int64)
    total = prefix_len + block_size

    src_row = target_req_to_token + req * target_pool_len
    dst_row = draft_req_to_token + req * draft_pool_len
    verify_row = verify_out_cache_loc + pid * verify_loc_stride

    offs = tl.arange(0, BLOCK).to(tl.int64)
    num_loop = tl.cdiv(total, BLOCK)
    for i in range(num_loop):
        col = offs + i * BLOCK
        in_prefix = col < prefix_len
        in_block = (col >= prefix_len) & (col < total)
        src = tl.load(src_row + start + col, mask=in_prefix, other=0)
        blk = tl.load(verify_row + (col - prefix_len), mask=in_block, other=0)
        val = tl.where(in_prefix, src, blk)
        tl.store(dst_row + col, val, mask=in_prefix | in_block)


def rebuild_compact_draft_req_to_token_func(
    *,
    draft_req_to_token: torch.Tensor,
    target_req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    suffix_start: torch.Tensor,
    draft_prefix_lens: torch.Tensor,
    verify_out_cache_loc_2d: torch.Tensor,
    batch_size: int,
    block_size: int,
) -> None:
    rebuild_compact_draft_req_to_token[(batch_size,)](
        draft_req_to_token,
        target_req_to_token,
        req_pool_indices,
        suffix_start,
        draft_prefix_lens,
        verify_out_cache_loc_2d,
        verify_out_cache_loc_2d.stride(0),
        draft_req_to_token.shape[1],
        target_req_to_token.shape[1],
        block_size,
    )


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


@triton.jit
def assign_extend_cache_locs_uniform(
    req_pool_indices,
    req_to_token,
    start_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    draft_token_num: tl.constexpr,
    full_v2p=None,
    swa_v2p=None,
    full_out=None,
    swa_out=None,
    PAGE_SIZE: tl.constexpr = 1,
    num_v_pages=0,
    num_swa_pages=0,
    positions_out=None,
    anchor_src=None,
    anchor_out=None,
    anchor_stride=0,
):
    """Uniform-length variant of assign_extend_cache_locs: every row extends
    exactly draft_token_num tokens, so the end offset is start +
    draft_token_num (computed here, no end_offset tensor) and the output
    offset is pid * draft_token_num (no cross-row prefix-sum loads)."""
    BLOCK_SIZE: tl.constexpr = 64
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    out_cache_ptr = out_cache_loc + pid * draft_token_num
    if anchor_out is not None:
        tl.store(anchor_out + pid * anchor_stride, tl.load(anchor_src + pid))

    offs = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(draft_token_num, BLOCK_SIZE)
    for i in range(num_loop):
        o = offs + i * BLOCK_SIZE
        mask = o < draft_token_num
        data = tl.load(token_pool + kv_start + o, mask=mask)
        tl.store(out_cache_ptr + o, data, mask=mask)
        if positions_out is not None:
            tl.store(
                positions_out + pid * draft_token_num + o,
                kv_start.to(tl.int64) + o,
                mask=mask,
            )
        if full_v2p is not None:
            page = data // PAGE_SIZE
            valid = mask & (data >= 0) & (page < num_v_pages)
            physical = tl.load(
                full_v2p + tl.where(valid, page, 0), mask=valid, other=-1
            )
            full_loc = tl.maximum(physical * PAGE_SIZE + data % PAGE_SIZE, 0)
            tl.store(full_out + pid * draft_token_num + o, full_loc, mask=mask)
            if swa_v2p is not None:
                swa_valid = mask & (data >= 0) & (page < num_swa_pages)
                physical = tl.load(
                    swa_v2p + tl.where(swa_valid, page, 0), mask=swa_valid, other=-1
                )
                swa_loc = tl.maximum(physical * PAGE_SIZE + data % PAGE_SIZE, 0)
                tl.store(swa_out + pid * draft_token_num + o, swa_loc, mask=mask)


def assign_extend_cache_locs_uniform_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
    *,
    full_v2p: Optional[torch.Tensor] = None,
    swa_v2p: Optional[torch.Tensor] = None,
    full_out: Optional[torch.Tensor] = None,
    swa_out: Optional[torch.Tensor] = None,
    page_size: int = 1,
    positions_out: Optional[torch.Tensor] = None,
    anchor_src: Optional[torch.Tensor] = None,
    anchor_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """assign_extend_cache_locs for the uniform case (all rows extend exactly
    draft_token_num tokens, e.g. spec target-verify prep). Computes end
    offsets inside the kernel, removing the eager `seq_lens + draft_token_num`
    add from the host critical path. Optional v2p tables fill physical FULL
    and SWA write windows in the same launch; the returned ids stay virtual.
    Mapping tables and output buffers must be contiguous, with one output
    per gathered token. Translation is rank-local (no DCP widening).
    ``positions_out`` ([batch, draft_token_num] int64) gets start + column and
    ``anchor_out[:, 0]`` gets ``anchor_src``, in the same launch.
    """
    if full_v2p is not None:
        assert full_out is not None and full_out.numel() == batch_size * draft_token_num
        assert full_v2p.is_contiguous() and full_out.is_contiguous()
        assert swa_v2p is None or (
            swa_out is not None
            and swa_out.numel() == full_out.numel()
            and swa_out.is_contiguous()
            and swa_v2p.is_contiguous()
        )
    if _is_cuda or _is_hip or _is_musa or _is_xpu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs_uniform[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            out_cache_loc,
            req_to_token.shape[1],
            draft_token_num,
            full_v2p=full_v2p,
            swa_v2p=swa_v2p,
            full_out=full_out,
            swa_out=swa_out,
            PAGE_SIZE=page_size,
            num_v_pages=0 if full_v2p is None else full_v2p.numel(),
            num_swa_pages=0 if swa_v2p is None else swa_v2p.numel(),
            positions_out=positions_out,
            anchor_src=anchor_src,
            anchor_out=anchor_out,
            anchor_stride=0 if anchor_out is None else anchor_out.stride(0),
        )
        return out_cache_loc

    # NPU / CPU platforms: fall back to the end_offset-tensor path.
    assert full_v2p is None, "fused write translation requires a Triton device"
    if positions_out is not None:
        positions_out.copy_(
            start_offset.to(torch.int64).view(-1, 1)
            + torch.arange(draft_token_num, device=start_offset.device)
        )
    if anchor_out is not None:
        anchor_out[:, 0].copy_(anchor_src.view(-1))
    return assign_extend_cache_locs_func(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        start_offset=start_offset,
        end_offset=start_offset + draft_token_num,
        batch_size=batch_size,
        draft_token_num=draft_token_num,
        device=device,
    )


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    platform_fn = _get_oot_speculative_cache_locs_fn()
    if platform_fn is not None:
        return platform_fn(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            start_offset=start_offset,
            end_offset=end_offset,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
            device=device,
        )

    if _is_cuda or _is_hip or _is_musa or _is_xpu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc

    elif _is_cpu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )

        return out_cache_loc


def fill_front_out_cache_loc_func(
    num_front: torch.Tensor,
    out_cache_loc: torch.Tensor,
    front_out_cache_loc: torch.Tensor,  # mutable
):
    """Pack the first ``num_front[i]`` slots of request i's window in ``out_cache_loc``."""
    bs = num_front.numel()
    window = out_cache_loc.view(bs, -1)
    rows = torch.arange(bs, device=window.device)
    starts = torch.zeros_like(num_front)
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            rows,
            window.int(),
            starts,
            num_front,
            front_out_cache_loc,
            window.shape[1],
        )
        return
    assign_extend_cache_locs[(bs,)](
        rows,
        window,
        starts,
        num_front,
        front_out_cache_loc,
        window.shape[1],
        next_power_of_2(bs),
    )
