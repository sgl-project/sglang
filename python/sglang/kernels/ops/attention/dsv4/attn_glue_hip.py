"""Single-launch replacements for the torch glue around the DeepSeek-V4.1 sparse attention metadata
on HIP; every kernel reproduces the torch expression quoted in its docstring bit for bit."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_indices_by_length_kernel(
    idx_ptr,
    len_ptr,
    out_ptr,
    idx2_ptr,
    len2_ptr,
    out2_ptr,
    rows_per_len,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_SECOND: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)
    length_row = row // rows_per_len
    offs = tl.arange(0, BLOCK)
    if which == 0:
        length = tl.load(len_ptr + length_row)
        for w0 in tl.static_range(0, W1, BLOCK):
            w = w0 + offs
            m = w < W1
            v = tl.load(idx_ptr + row * W1 + w, mask=m, other=0)
            tl.store(out_ptr + row * W1 + w, tl.where(w < length, v, -1), mask=m)
    elif HAS_SECOND:
        length = tl.load(len2_ptr + length_row)
        for w0 in tl.static_range(0, W2, BLOCK):
            w = w0 + offs
            m = w < W2
            v = tl.load(idx2_ptr + row * W2 + w, mask=m, other=0)
            tl.store(out2_ptr + row * W2 + w, tl.where(w < length, v, -1), mask=m)


def _mask_shape(indices: torch.Tensor, lengths: torch.Tensor) -> Tuple[int, int, int]:
    assert indices.is_contiguous(), indices.stride()
    assert lengths.dim() == 1 and lengths.is_contiguous()
    w = indices.shape[-1]
    rows = indices.numel() // w if w else 0
    batch = lengths.shape[0]
    assert batch > 0 and rows % batch == 0, (indices.shape, lengths.shape)
    return rows, rows // batch, w


def mask_indices_by_length(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    indices2: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """torch.where(arange(w) < lengths.view(-1, 1, 1), indices, -1) for one or
    two [b, s, w] index tensors in one launch (the sparse decode kernel takes
    no per-row length and skips -1). lengths is [b]; every s row of a
    batch entry shares its length. Returns new tensors of the input dtypes."""
    rows, rows_per_len, w1 = _mask_shape(indices, lengths)
    out = torch.empty_like(indices)
    has_second = indices2 is not None
    if has_second:
        rows2, rows_per_len2, w2 = _mask_shape(indices2, lengths2)
        assert rows2 == rows and rows_per_len2 == rows_per_len, (
            indices.shape,
            indices2.shape,
        )
        out2 = torch.empty_like(indices2)
    else:
        w2 = w1
        indices2, lengths2, out2 = indices, lengths, out
    if rows == 0:
        return out, (out2 if has_second else None)
    block = min(1024, triton.next_power_of_2(max(w1, w2)))
    _mask_indices_by_length_kernel[(rows, 2 if has_second else 1)](
        indices,
        lengths,
        out,
        indices2,
        lengths2,
        out2,
        rows_per_len,
        W1=w1,
        W2=w2,
        BLOCK=block,
        HAS_SECOND=has_second,
        num_warps=4,
    )
    return out, (out2 if has_second else None)


@triton.jit
def _expand_index_page_table_kernel(
    pt_ptr,
    out_ptr,
    n_pages,
    pt_stride,
    BPP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < n_pages
    page = tl.load(pt_ptr + row * pt_stride + p, mask=m, other=0)
    base = page.to(tl.int64) * BPP
    k = tl.arange(0, BPP)
    expanded = base[:, None] + k[None, :]
    tl.store(
        out_ptr + row * (n_pages * BPP) + p[:, None] * BPP + k[None, :],
        expanded.to(tl.int32),
        mask=m[:, None],
    )


def expand_index_page_table(
    page_table: torch.Tensor, blocks_per_page: int
) -> torch.Tensor:
    """(page_table.to(int64) * bpp).unsqueeze(-1) + arange(bpp) reshaped to
    [bs, n * bpp] int32: the block table of a low-ratio indexer-K pool that
    pages at a fraction of a FULL page (candidate_indexer.expand_index_page_table)."""
    if blocks_per_page == 1:
        return page_table
    assert blocks_per_page & (blocks_per_page - 1) == 0, blocks_per_page
    assert page_table.dim() == 2 and page_table.stride(1) == 1
    bs, n = page_table.shape
    out = torch.empty(
        (bs, n * blocks_per_page), dtype=torch.int32, device=page_table.device
    )
    if bs == 0 or n == 0:
        return out
    block = 256
    _expand_index_page_table_kernel[(bs, triton.cdiv(n, block))](
        page_table,
        out,
        n,
        page_table.stride(0),
        BPP=blocks_per_page,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _low_ratio_compression_metadata_kernel(
    seq_lens_ptr,
    raw_out_loc_ptr,
    c1_out_loc_ptr,
    c1_clamp1_ptr,
    c2_out_loc_ptr,
    c2_clamp1_ptr,
    rows,
    num_write_tokens,
    HAS_C1: tl.constexpr,
    HAS_C2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < rows
    mw = r < num_write_tokens
    seq_len = tl.load(seq_lens_ptr + r, mask=m, other=0)
    raw_out_loc = tl.load(raw_out_loc_ptr + r, mask=mw, other=0).to(tl.int64)
    if HAS_C1:
        # ratio 1: every token completes a group, so out_loc is raw_out_loc itself
        tl.store(c1_out_loc_ptr + r, raw_out_loc, mask=mw)
        tl.store(c1_clamp1_ptr + r, tl.maximum(seq_len, 1), mask=m)
    if HAS_C2:
        completes = (seq_len % 2) == 0
        # >> 1 floors as torch's // does; Triton's // rounds toward zero on a negative loc
        tl.store(c2_out_loc_ptr + r, tl.where(completes, raw_out_loc >> 1, -1), mask=mw)
        tl.store(c2_clamp1_ptr + r, tl.maximum(seq_len // 2, 1), mask=m)


def low_ratio_compression_metadata(
    seq_lens_casual: torch.Tensor,
    raw_out_loc: torch.Tensor,
    low_ratios: Tuple[int, ...],
) -> Dict[str, torch.Tensor]:
    """_low_ratio_compression_metadata for ratios 1 and 2 in one launch: cR_out_loc =
    where(seq_lens[:nw] % R == 0, raw_out_loc // R, -1) (int64) and cR_topk_lengths_clamp1 =
    (seq_lens // R).clamp_min(1) (int32), keyed c{R}_... for the ratios present."""
    assert seq_lens_casual.dtype is torch.int32 and seq_lens_casual.is_contiguous()
    assert raw_out_loc.dim() == 1 and raw_out_loc.is_contiguous()
    assert set(low_ratios) <= {1, 2}, low_ratios
    rows = seq_lens_casual.shape[0]
    nw = raw_out_loc.shape[0]
    assert nw <= rows, (nw, rows)
    device = seq_lens_casual.device
    out = {}
    for ratio in (1, 2):
        if ratio in low_ratios:
            out[f"c{ratio}_out_loc"] = torch.empty(nw, dtype=torch.int64, device=device)
            out[f"c{ratio}_topk_lengths_clamp1"] = torch.empty(
                rows, dtype=torch.int32, device=device
            )
    if not out or rows == 0:
        return out
    placeholder = seq_lens_casual
    _low_ratio_compression_metadata_kernel[(triton.cdiv(rows, 1024),)](
        seq_lens_casual,
        raw_out_loc,
        out.get("c1_out_loc", placeholder),
        out.get("c1_topk_lengths_clamp1", placeholder),
        out.get("c2_out_loc", placeholder),
        out.get("c2_topk_lengths_clamp1", placeholder),
        rows,
        nw,
        HAS_C1=1 in low_ratios,
        HAS_C2=2 in low_ratios,
        BLOCK=1024,
        num_warps=4,
    )
    return out


@triton.jit
def _expand_same_length_kernel(
    seq_lens_ptr,
    req_pool_ptr,
    seq_lens_out_ptr,
    req_pool_out_ptr,
    positions_out_ptr,
    window_lens_out_ptr,
    num_tokens,
    qo_len,
    offset,
    window,
    BLOCK: tl.constexpr,
    WITH_POSITIONS: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = t < num_tokens
    r = t // qo_len
    seq_len = tl.load(seq_lens_ptr + r, mask=m, other=0)
    lens = seq_len + (offset - qo_len + 1) + t % qo_len
    tl.store(seq_lens_out_ptr + t, lens, mask=m)
    tl.store(req_pool_out_ptr + t, tl.load(req_pool_ptr + r, mask=m, other=0), mask=m)
    if WITH_POSITIONS:
        lens32 = lens.to(tl.int32)
        tl.store(positions_out_ptr + t, lens32 - 1, mask=m)
        tl.store(window_lens_out_ptr + t, tl.minimum(lens32, window), mask=m)


def expand_same_length(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    qo_len: int,
    offset: int = 0,
    window: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Every request's qo_len rows, one launch: row t of request r sees
    seq_lens[r] + offset - qo_len + 1 + t % qo_len keys (its causal length, in
    seq_lens' dtype) and reads req_pool_indices[r]. With ``window``, also each
    row's int32 position (length - 1) and its length clamped to the window."""
    bs = seq_lens.shape[0]
    assert req_pool_indices.shape[0] == bs and seq_lens.is_contiguous()
    num_tokens = bs * qo_len
    device = seq_lens.device
    seq_lens_out = torch.empty(num_tokens, dtype=seq_lens.dtype, device=device)
    req_pool_out = torch.empty(
        num_tokens, dtype=req_pool_indices.dtype, device=req_pool_indices.device
    )
    with_positions = window is not None
    positions = window_lens = seq_lens_out
    if with_positions:
        positions = torch.empty(num_tokens, dtype=torch.int32, device=device)
        window_lens = torch.empty(num_tokens, dtype=torch.int32, device=device)
    if num_tokens:
        block = 256
        _expand_same_length_kernel[(triton.cdiv(num_tokens, block),)](
            seq_lens,
            req_pool_indices.contiguous(),
            seq_lens_out,
            req_pool_out,
            positions,
            window_lens,
            num_tokens,
            qo_len,
            offset,
            window if with_positions else 0,
            BLOCK=block,
            WITH_POSITIONS=with_positions,
        )
    if with_positions:
        return seq_lens_out, req_pool_out, positions, window_lens
    return seq_lens_out, req_pool_out


@triton.jit
def _sparse_buffers_kernel(
    c4_clamp1_ptr,
    c4_raw_ptr,
    c4_sparse_ptr,
    c4_sparse_raw_ptr,
    c4_page_ptr,
    c1_clamp1_ptr,
    c1_sparse_ptr,
    c1_page_ptr,
    c1_raw_ptr,
    c2_clamp1_ptr,
    c2_sparse_ptr,
    c2_page_ptr,
    c2_raw_ptr,
    rows,
    topk,
    width,
    HAS_C4: tl.constexpr,
    HAS_C1: tl.constexpr,
    HAS_C2: tl.constexpr,
    C1_AT: tl.constexpr,
    C2_AT: tl.constexpr,
    C1R_AT: tl.constexpr,
    C2R_AT: tl.constexpr,
    BLOCK: tl.constexpr,
    FILL_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        # one program clamps the per-row lengths: a few ints per row
        for r0 in range(0, rows, BLOCK):
            r = r0 + tl.arange(0, BLOCK)
            m = r < rows
            if HAS_C4:
                c4_clamp1 = tl.load(c4_clamp1_ptr + r, mask=m, other=0)
                c4_raw = tl.load(c4_raw_ptr + r, mask=m, other=0)
                tl.store(c4_sparse_ptr + r, tl.minimum(c4_clamp1, topk), mask=m)
                tl.store(c4_sparse_raw_ptr + r, tl.minimum(c4_raw, topk), mask=m)
            if HAS_C1:
                c1_clamp1 = tl.load(c1_clamp1_ptr + r, mask=m, other=0)
                tl.store(c1_sparse_ptr + r, tl.minimum(c1_clamp1, topk), mask=m)
            if HAS_C2:
                c2_clamp1 = tl.load(c2_clamp1_ptr + r, mask=m, other=0)
                tl.store(c2_sparse_ptr + r, tl.minimum(c2_clamp1, topk), mask=m)
    else:
        # the rest fill the -1 buffers, one buffer per range of programs: c4's page
        # indices first, then c1's and c2's page (C1_AT, C2_AT) and raw (C1R_AT,
        # C2R_AT) indices
        fill = pid - 1
        n = rows * width
        per_buffer = tl.cdiv(n, FILL_BLOCK)
        which = fill // per_buffer
        i = (fill - which * per_buffer) * FILL_BLOCK + tl.arange(0, FILL_BLOCK)
        m = i < n
        if HAS_C4:
            if which == 0:
                tl.store(c4_page_ptr + i, -1, mask=m)
        if HAS_C1:
            if which == C1_AT:
                tl.store(c1_page_ptr + i, -1, mask=m)
            if which == C1R_AT:
                tl.store(c1_raw_ptr + i, -1, mask=m)
        if HAS_C2:
            if which == C2_AT:
                tl.store(c2_page_ptr + i, -1, mask=m)
            if which == C2R_AT:
                tl.store(c2_raw_ptr + i, -1, mask=m)


def sparse_buffers(
    *,
    c4_topk_lengths_clamp1: Optional[torch.Tensor],
    c4_topk_lengths_raw: Optional[torch.Tensor],
    c1_topk_lengths_clamp1: Optional[torch.Tensor],
    c2_topk_lengths_clamp1: Optional[torch.Tensor],
    index_topk: int,
    page_index_align: int,
) -> Dict[str, torch.Tensor]:
    """init_flashmla_related's tensors in one launch, keyed by attribute name: the top-k
    lengths clamped to index_topk and the -1-filled int32 page-index buffers padded to
    page_index_align, for the ratios whose clamp-1 lengths are given (ratio 4 with its raw
    lengths too); ratios 1 and 2 also get -1-filled raw-index buffers."""
    for t in (
        c4_topk_lengths_clamp1,
        c4_topk_lengths_raw,
        c1_topk_lengths_clamp1,
        c2_topk_lengths_clamp1,
    ):
        assert t is None or (
            t.dtype is torch.int32 and t.dim() == 1 and t.is_contiguous()
        ), t
    has_c4 = c4_topk_lengths_clamp1 is not None
    has_c1 = c1_topk_lengths_clamp1 is not None
    has_c2 = c2_topk_lengths_clamp1 is not None
    assert has_c4 == (c4_topk_lengths_raw is not None)
    lengths = [
        t
        for t in (
            c4_topk_lengths_clamp1,
            c1_topk_lengths_clamp1,
            c2_topk_lengths_clamp1,
        )
        if t is not None
    ]
    if not lengths:
        return {}
    rows = lengths[0].shape[0]
    assert all(t.shape[0] == rows for t in lengths)
    assert not has_c4 or c4_topk_lengths_raw.shape[0] == rows
    device = lengths[0].device
    width = triton.cdiv(index_topk, page_index_align) * page_index_align
    i32 = dict(dtype=torch.int32, device=device)
    out = {}
    if has_c4:
        out["c4_sparse_topk_lengths"] = torch.empty(rows, **i32)
        out["c4_sparse_topk_lengths_raw"] = torch.empty(rows, **i32)
        out["c4_sparse_page_indices"] = torch.empty((rows, width), **i32)
    for ratio, has in ((1, has_c1), (2, has_c2)):
        if has:
            out[f"c{ratio}_sparse_topk_lengths"] = torch.empty(rows, **i32)
            out[f"c{ratio}_sparse_page_indices"] = torch.empty((rows, width), **i32)
            out[f"c{ratio}_sparse_raw_indices"] = torch.empty((rows, width), **i32)
    if rows == 0:
        return out
    placeholder = lengths[0]
    fill_block = 1024
    per_buffer = triton.cdiv(rows * width, fill_block)
    n_buffers = int(has_c4) + 2 * int(has_c1) + 2 * int(has_c2)
    _sparse_buffers_kernel[(1 + per_buffer * n_buffers,)](
        c4_topk_lengths_clamp1 if has_c4 else placeholder,
        c4_topk_lengths_raw if has_c4 else placeholder,
        out.get("c4_sparse_topk_lengths", placeholder),
        out.get("c4_sparse_topk_lengths_raw", placeholder),
        out.get("c4_sparse_page_indices", placeholder),
        c1_topk_lengths_clamp1 if has_c1 else placeholder,
        out.get("c1_sparse_topk_lengths", placeholder),
        out.get("c1_sparse_page_indices", placeholder),
        out.get("c1_sparse_raw_indices", placeholder),
        c2_topk_lengths_clamp1 if has_c2 else placeholder,
        out.get("c2_sparse_topk_lengths", placeholder),
        out.get("c2_sparse_page_indices", placeholder),
        out.get("c2_sparse_raw_indices", placeholder),
        rows,
        index_topk,
        width,
        HAS_C4=has_c4,
        HAS_C1=has_c1,
        HAS_C2=has_c2,
        C1_AT=int(has_c4),
        C2_AT=int(has_c4) + int(has_c1),
        C1R_AT=int(has_c4) + int(has_c1) + int(has_c2),
        C2R_AT=int(has_c4) + 2 * int(has_c1) + int(has_c2),
        BLOCK=1024,
        FILL_BLOCK=fill_block,
        num_warps=4,
    )
    return out


@triton.jit
def _page_table_from_req_to_token_kernel(
    r2t_ptr,
    req_ptr,
    out_ptr,
    num_cols,
    r2t_stride,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    req = tl.load(req_ptr + row).to(tl.int64)
    col = blk * BLOCK + tl.arange(0, BLOCK)
    mask = col < num_cols
    v = tl.load(
        r2t_ptr + req * r2t_stride + col.to(tl.int64) * PAGE, mask=mask, other=0
    )
    # torch's floor division; the C-style quotient rounds toward zero
    q = v // PAGE
    q = tl.where((v < 0) & (v % PAGE != 0), q - 1, q)
    tl.store(out_ptr + row.to(tl.int64) * num_cols + col, q.to(tl.int32), mask=mask)


def page_table_from_req_to_token(
    req_to_token: torch.Tensor,
    req_rows: torch.Tensor,
    max_seq_len: int,
    page_size: int,
) -> torch.Tensor:
    """(req_to_token[req_rows, :max_seq_len:page_size] // page_size).to(torch.int32) in one
    launch (the gather, the floor division and the cast)."""
    assert req_to_token.dim() == 2 and req_to_token.stride(1) == 1
    assert req_rows.dim() == 1 and req_rows.stride(0) == 1
    n = req_rows.shape[0]
    num_cols = max(
        0, (min(max_seq_len, req_to_token.shape[1]) + page_size - 1) // page_size
    )
    out = torch.empty((n, num_cols), dtype=torch.int32, device=req_to_token.device)
    if n == 0 or num_cols == 0:
        return out
    block = min(1024, triton.next_power_of_2(num_cols))
    _page_table_from_req_to_token_kernel[(n, triton.cdiv(num_cols, block))](
        req_to_token,
        req_rows,
        out,
        num_cols,
        req_to_token.stride(0),
        PAGE=page_size,
        BLOCK=block,
        num_warps=4,
    )
    return out


@triton.jit
def _widen_pair_kernel(a_ptr, b_ptr, oa_ptr, ob_ptr, n, repeat, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    a = tl.load(a_ptr + i // repeat, mask=mask, other=0)
    tl.store(oa_ptr + i, a.to(tl.int64), mask=mask)
    tl.store(ob_ptr + i, tl.load(b_ptr + i, mask=mask, other=0).to(tl.int64), mask=mask)


def widen_pair_i64(
    a: torch.Tensor, b: torch.Tensor, repeat: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(a.repeat_interleave(repeat).to(torch.int64), b.to(torch.int64)) in one launch,
    b ``repeat`` times a's length: the request slots and positions the low-ratio
    indexer takes widened, a slot a row (decode) or a draft token (target verify)."""
    assert a.dim() == b.dim() == 1 and b.shape[0] == a.shape[0] * repeat, (
        a.shape,
        b.shape,
        repeat,
    )
    assert a.stride(0) == 1 and b.stride(0) == 1
    n = b.shape[0]
    oa = torch.empty(n, dtype=torch.int64, device=a.device)
    ob = torch.empty(n, dtype=torch.int64, device=a.device)
    if n:
        block = min(1024, triton.next_power_of_2(n))
        _widen_pair_kernel[(triton.cdiv(n, block),)](
            a, b, oa, ob, n, repeat, BLOCK=block, num_warps=4
        )
    return oa, ob


@triton.jit
def _gather_i32_kernel(table_ptr, idx_ptr, out_ptr, n, table_len, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    idx = tl.load(idx_ptr + i, mask=mask, other=0).to(tl.int64)
    # torch indexing: a negative index counts from the end
    idx = tl.where(idx < 0, idx + table_len, idx)
    tl.store(out_ptr + i, tl.load(table_ptr + idx, mask=mask).to(tl.int32), mask=mask)


def gather_i32(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table[idx].to(torch.int32) in one launch (a 1-D integer table and index vector)."""
    assert table.dim() == idx.dim() == 1 and table.stride(0) == 1, (
        table.shape,
        idx.shape,
    )
    idx = idx.contiguous()
    n = idx.shape[0]
    out = torch.empty(n, dtype=torch.int32, device=idx.device)
    if n:
        block = min(1024, triton.next_power_of_2(n))
        _gather_i32_kernel[(triton.cdiv(n, block),)](
            table, idx, out, n, table.shape[0], BLOCK=block, num_warps=4
        )
    return out
