from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.utils import is_hip

BLOCK_SIZE_K = 64
INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _preshuffle_tile() -> int:
    return (
        INDEXER_K_CACHE_PRESHUFFLE_TILE if aiter_can_use_preshuffle_paged_mqa() else 0
    )


@triton.jit
def _kpool_cache_k_offsets(
    page,
    slot,
    cols,
    PAGE_BYTES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
):
    if PRESHUFFLE_TILE:
        token_tile_id = slot // PRESHUFFLE_TILE
        token_in_tile = slot % PRESHUFFLE_TILE
        col_tile_id = cols // PRESHUFFLE_TILE
        col_in_tile = cols % PRESHUFFLE_TILE
        return (
            page * PAGE_BYTES
            + token_tile_id * (PRESHUFFLE_TILE * HEAD_DIM)
            + col_tile_id * (PRESHUFFLE_TILE * PRESHUFFLE_TILE)
            + token_in_tile * PRESHUFFLE_TILE
            + col_in_tile
        )
    return page * PAGE_BYTES + slot * HEAD_DIM + cols


def kpool_max_closed_pools(num_draft_tokens: int, pool_size: int) -> int:
    return (num_draft_tokens + pool_size - 1) // pool_size


def build_pooled_page_table_64(
    page_table_64: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    # Advanced indexing is required: a (1, 1) strided slice can remain non-unit-
    # stride even after contiguous(), which DeepGEMM rejects.
    assert (
        BLOCK_SIZE_K % pool_size == 0
    ), f"pool_size ({pool_size}) must divide page_size ({BLOCK_SIZE_K})"
    idx = torch.arange(
        0, page_table_64.shape[-1], pool_size, device=page_table_64.device
    )
    return page_table_64[..., idx]


def gather_index_k_scale_prefix_into(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    assert buf.dtype == torch.uint8
    assert page_indices.dtype in (torch.int32, torch.int64)
    assert k_out.dtype == torch.uint8
    assert scale_out.dtype == torch.float32
    assert pool.page_size == BLOCK_SIZE_K
    assert k_out.shape[0] >= seq_len
    assert k_out.shape[1] == INDEX_HEAD_DIM
    assert scale_out.shape[0] >= seq_len
    assert buf.is_contiguous()
    assert page_indices.is_contiguous()
    assert k_out.is_contiguous()
    assert scale_out.is_contiguous()
    if seq_len == 0:
        return

    _gather_index_k_scale_prefix_into_kernel[(seq_len,)](
        buf,
        buf.view(torch.float32),
        page_indices,
        k_out,
        scale_out,
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=pool.page_size * INDEX_HEAD_DIM,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _gather_index_k_scale_prefix_into_kernel(
    buf_u8_ptr,
    buf_fp32_ptr,
    page_indices_ptr,
    k_out_ptr,
    scale_out_ptr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    page_idx = token_id // PAGE_SIZE
    token_offset_in_page = token_id % PAGE_SIZE
    page = tl.load(page_indices_ptr + page_idx)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    src_k_offsets = _kpool_cache_k_offsets(
        page,
        token_offset_in_page,
        offs,
        BUF_NUMEL_PER_PAGE,
        HEAD_DIM,
        PRESHUFFLE_TILE,
    )
    dst_k_offsets = token_id * HEAD_DIM + offs
    k = tl.load(buf_u8_ptr + src_k_offsets, mask=mask)
    tl.store(k_out_ptr + dst_k_offsets, k, mask=mask)

    src_s_offset = (
        page * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + token_offset_in_page
    )
    scale = tl.load(buf_fp32_ptr + src_s_offset)
    tl.store(scale_out_ptr + token_id, scale)


def kpool_build_ragged_layout(
    full_page_table: torch.Tensor,
    cu_pages_excl: torch.Tensor,
    ragged_pool_pages: torch.Tensor,
    cu_q_len_excl: torch.Tensor,
    ragged_q_len: torch.Tensor,
    pooled_seq_lens_expanded: torch.Tensor,
    slots_per_page: int,
    total_pool_pages: int,
    total_q: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # One packed-cache page represents 64 pools, so its source is every
    # pool_size-th real token page.
    device = full_page_table.device
    n_rag = cu_pages_excl.shape[0]
    concat_page_table = torch.empty(
        (total_pool_pages,), dtype=full_page_table.dtype, device=device
    )
    q_ks = torch.empty((total_q,), dtype=torch.int32, device=device)
    q_ke = torch.empty((total_q,), dtype=torch.int32, device=device)
    if n_rag == 0:
        return concat_page_table, q_ks, q_ke

    max_pool_pages = full_page_table.shape[1]
    _kpool_build_ragged_layout_kernel[(n_rag,)](
        full_page_table,
        cu_pages_excl,
        ragged_pool_pages,
        cu_q_len_excl,
        ragged_q_len,
        pooled_seq_lens_expanded,
        concat_page_table,
        q_ks,
        q_ke,
        max_pool_pages,
        slots_per_page,
        pool_size,
        BLOCK_PAGE=128,
        BLOCK_Q=128,
    )
    return concat_page_table, q_ks, q_ke


@triton.jit
def _kpool_build_ragged_layout_kernel(
    full_page_table_ptr,
    cu_pages_excl_ptr,
    ragged_pool_pages_ptr,
    cu_q_len_excl_ptr,
    ragged_q_len_ptr,
    pooled_seq_lens_ptr,
    concat_page_table_ptr,
    q_ks_ptr,
    q_ke_ptr,
    MAX_POOL_PAGES,
    SLOTS_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    k = tl.program_id(0)
    page_start = tl.load(cu_pages_excl_ptr + k)
    n_pages = tl.load(ragged_pool_pages_ptr + k)
    q_start = tl.load(cu_q_len_excl_ptr + k)
    q_count = tl.load(ragged_q_len_ptr + k)
    ks_val = page_start * SLOTS_PER_PAGE

    for p_off in tl.range(0, BLOCK_PAGE * tl.cdiv(n_pages, BLOCK_PAGE), BLOCK_PAGE):
        p_offs = p_off + tl.arange(0, BLOCK_PAGE)
        p_mask = p_offs < n_pages
        source_cols = p_offs * POOL_SIZE
        pages = tl.load(
            full_page_table_ptr + k * MAX_POOL_PAGES + source_cols,
            mask=p_mask,
            other=0,
        )
        tl.store(concat_page_table_ptr + page_start + p_offs, pages, mask=p_mask)

    for q_off in tl.range(0, BLOCK_Q * tl.cdiv(q_count, BLOCK_Q), BLOCK_Q):
        q_offs = q_off + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < q_count
        plen = tl.load(pooled_seq_lens_ptr + q_start + q_offs, mask=q_mask, other=0)
        ke_val = tl.minimum(
            ks_val + plen,
            (page_start + n_pages) * SLOTS_PER_PAGE,
        )
        tl.store(
            q_ks_ptr + q_start + q_offs,
            tl.full([BLOCK_Q], ks_val, tl.int32),
            mask=q_mask,
        )
        tl.store(q_ke_ptr + q_start + q_offs, ke_val, mask=q_mask)


def _prep_update_kpool_write_plan_launch(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out: Optional[torch.Tensor],
    seqlens_per_q_out: Optional[torch.Tensor],
    *,
    pool_size: int,
    num_draft_tokens: int,
    slots_per_page: int,
):
    # Eager and captured replay must share one launch spec, including
    # N > pool_size where one write closes multiple pools.
    max_closed_pools = kpool_max_closed_pools(num_draft_tokens, pool_size)
    bs = write_start.shape[0]

    assert write_loc_out.shape == (bs, max_closed_pools), write_loc_out.shape
    assert write_loc_out.stride(1) == 1, write_loc_out.stride()

    has_per_q_outputs = pool_seqlens_per_q_out is not None
    assert has_per_q_outputs == (
        seqlens_per_q_out is not None
    ), "pool_seqlens_per_q_out and seqlens_per_q_out must be both set or both None"
    per_q_dummy = (
        pool_seqlens_per_q_out
        if has_per_q_outputs
        else torch.empty(1, dtype=torch.int32, device=write_start.device)
    )

    args = (
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out if has_per_q_outputs else per_q_dummy,
        seqlens_per_q_out if has_per_q_outputs else per_q_dummy,
        real_page_table.stride(0),
        real_page_table.shape[1],
        write_loc_out.stride(0),
    )
    constexprs = dict(
        POOL_SIZE=pool_size,
        N=num_draft_tokens,
        SLOTS_PER_PAGE=slots_per_page,
        MAX_CLOSED_POOLS=max_closed_pools,
        HAS_PER_Q=has_per_q_outputs,
    )
    return bs, args, constexprs


def update_kpool_write_plan_cuda_graph(
    write_start: torch.Tensor,
    req_pool_indices: torch.Tensor,
    real_page_table: torch.Tensor,
    req_out: torch.Tensor,
    write_start_out: torch.Tensor,
    tail_logical_start_out: torch.Tensor,
    write_loc_out: torch.Tensor,
    pool_seqlens_per_q_out: Optional[torch.Tensor],
    seqlens_per_q_out: Optional[torch.Tensor],
    *,
    pool_size: int,
    num_draft_tokens: int,
    slots_per_page: int,
) -> None:
    if write_start.shape[0] == 0:
        return
    bs, args, constexprs = _prep_update_kpool_write_plan_launch(
        write_start,
        req_pool_indices,
        real_page_table,
        req_out,
        write_start_out,
        tail_logical_start_out,
        write_loc_out,
        pool_seqlens_per_q_out,
        seqlens_per_q_out,
        pool_size=pool_size,
        num_draft_tokens=num_draft_tokens,
        slots_per_page=slots_per_page,
    )
    _update_kpool_write_plan_kernel[(bs,)](*args, **constexprs)


@triton.jit
def _update_kpool_write_plan_kernel(
    write_start_ptr,
    req_pool_indices_ptr,
    real_page_table_ptr,
    req_out_ptr,
    write_start_out_ptr,
    tail_logical_start_out_ptr,
    write_loc_out_ptr,
    pool_seqlens_per_q_out_ptr,
    seqlens_per_q_out_ptr,
    real_page_table_stride_0,
    real_page_table_cols,
    write_loc_out_stride_0,
    POOL_SIZE: tl.constexpr,
    N: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
    HAS_PER_Q: tl.constexpr,
):
    b = tl.program_id(0)
    ws = tl.load(write_start_ptr + b).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + b)
    base_pool = ws // POOL_SIZE

    if HAS_PER_Q:
        for k in tl.static_range(0, N):
            row = b * N + k
            seqlen_per_q = ws + k + 1
            tl.store(seqlens_per_q_out_ptr + row, seqlen_per_q)
            tl.store(pool_seqlens_per_q_out_ptr + row, seqlen_per_q // POOL_SIZE)

    tl.store(req_out_ptr + b, req)
    tl.store(write_start_out_ptr + b, ws)
    tl.store(tail_logical_start_out_ptr + b, (base_pool * POOL_SIZE).to(tl.int32))
    for p in tl.static_range(0, MAX_CLOSED_POOLS):
        pool_id = base_pool + p
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = pool_page_group * POOL_SIZE
        token_page_row = tl.minimum(
            tl.maximum(token_page_row, 0), real_page_table_cols - 1
        )
        # Promote the row term before multiplication; 1M-context strides
        # overflow int32 around row 2048.
        packed_page = tl.load(
            real_page_table_ptr
            + (b * N).to(tl.int64) * real_page_table_stride_0
            + token_page_row
        ).to(tl.int64)
        write_loc = packed_page * SLOTS_PER_PAGE + (pool_id % SLOTS_PER_PAGE)
        tl.store(
            write_loc_out_ptr + b * write_loc_out_stride_0 + p,
            write_loc.to(tl.int64),
        )


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    pool_page_group = torch.div(pool_ids, BLOCK_SIZE_K, rounding_mode="floor")
    token_page_row = pool_page_group * pool_size
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * BLOCK_SIZE_K + torch.remainder(
        pool_ids, BLOCK_SIZE_K
    )


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pooled_groups_to_topk(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        assert page_table.shape[0] == group_ids.shape[0]
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_kpool_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1
    assert seq_lens.shape[0] == topk_result.shape[0]
    assert pool_lens.shape[0] == topk_result.shape[0]

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    if page_table is None:
        page_table = topk_result
        has_page_table = False
        page_table_cols = 1
    else:
        assert page_table.ndim == 2
        has_page_table = True
        page_table_cols = page_table.shape[1]

    if topk_offsets is None:
        topk_offsets = seq_lens
        has_topk_offsets = False
    else:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        has_topk_offsets = True

    block_cols = triton.next_power_of_2(out_cols)
    _append_kpool_tail_to_topk_kernel[(rows,)](
        topk_result,
        seq_lens,
        pool_lens,
        page_table,
        topk_offsets,
        out,
        topk_result.stride(0),
        topk_result.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        N_COLS=n_cols,
        OUT_COLS=out_cols,
        PAGE_TABLE_COLS=page_table_cols,
        POOL_SIZE=pool_size,
        HAS_PAGE_TABLE=has_page_table,
        HAS_TOPK_OFFSETS=has_topk_offsets,
        BLOCK_COLS=block_cols,
    )
    return out


@triton.jit
def _append_kpool_tail_to_topk_kernel(
    topk_ptr,
    seq_lens_ptr,
    pool_lens_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    out_ptr,
    topk_stride_0,
    topk_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    out_stride_0,
    out_stride_1,
    N_COLS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    PAGE_TABLE_COLS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < OUT_COLS

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * POOL_SIZE
    history_len = tl.minimum(tail_start, N_COLS)
    tail_count = seq_len % POOL_SIZE

    is_history = cols < history_len
    safe_history_cols = tl.minimum(cols, N_COLS - 1)
    history_value = tl.load(
        topk_ptr + row * topk_stride_0 + safe_history_cols * topk_stride_1,
        mask=mask & is_history,
        other=-1,
    )

    tail_offset = cols - history_len
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = tail_start + tail_offset
    tail_value = tail_raw
    if HAS_PAGE_TABLE:
        safe_tail = tl.minimum(tl.maximum(tail_raw, 0), PAGE_TABLE_COLS - 1)
        # Promote the row term before multiplication; 1M-context strides
        # overflow int32 around row 2048.
        tail_value = tl.load(
            page_table_ptr
            + row.to(tl.int64) * page_table_stride_0
            + safe_tail * page_table_stride_1,
            mask=mask & is_tail,
            other=-1,
        ).to(tl.int32)
    if HAS_TOPK_OFFSETS:
        offset = tl.load(topk_offsets_ptr + row).to(tl.int32)
        tail_value = tail_raw + offset

    value = tl.where(is_history, history_value, -1)
    value = tl.where(is_tail, tail_value, value)
    tl.store(out_ptr + row * out_stride_0 + cols * out_stride_1, value, mask=mask)


def topk_from_pooled_history_logits(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    assert logits.ndim == 2
    assert group_lengths.ndim == 1
    assert logits.shape[0] == group_lengths.shape[0]
    assert topk > 0
    assert topk % pool_size == 0
    assert out_rows is None or out_rows >= logits.shape[0]
    assert page_table_row_index is None or page_table is not None

    _, cols = logits.shape
    group_topk = history_group_budget_for_topk(topk, pool_size)
    if topk_offsets is not None and topk_offsets.ndim == 2:
        assert topk_offsets.shape[1] == 1
        topk_offsets = topk_offsets.squeeze(1)

    if group_topk not in (128, 160, 192, 224, 256, 512, 2048):
        raise NotImplementedError(
            "index_kpool topk only supports pooled group_topk in "
            f"(128, 160, 192, 224, 256, 512, 2048), got {group_topk} "
            f"(topk={topk}, pool_size={pool_size})."
        )
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise NotImplementedError(
            "index_kpool topk requires CUDA float32 logits; PyTorch topk fallback "
            f"is disabled. Got device={logits.device}, dtype={logits.dtype}."
        )

    # The JIT kernel covers group_topk <= 512 on CUDA/HIP; HIP 2048 still
    # requires the legacy AOT fallback.
    if is_hip() and group_topk == 2048:
        return _topk_from_pooled_history_logits_unfused(
            logits=logits,
            group_lengths=group_lengths,
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    if group_topk in (128, 160, 192, 224, 256, 512):
        from sglang.kernels.ops.moe.kpool_topk_transform import (
            fast_kpool_topk_transform_fused,
        )

        result = fast_kpool_topk_transform_fused(
            score=logits,
            lengths=group_lengths.to(torch.int32),
            pool_size=pool_size,
            topk=topk,
            page_table=page_table,
            topk_indices_offset=topk_offsets,
            row_starts=row_starts,
            seq_lens=seq_lens.to(torch.int32) if seq_lens is not None else None,
            page_table_row_index=page_table_row_index,
        )
        if out_rows is None or out_rows == result.shape[0]:
            return result
        padded = torch.full(
            (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
        )
        padded[: result.shape[0]] = result
        return padded

    assert (
        page_table_row_index is None
    ), "page_table_row_index requires the fused fast_kpool group_topk path"

    from sgl_kernel import fast_topk_v2

    selected_groups = fast_topk_v2(
        logits,
        group_lengths.to(torch.int32),
        group_topk,
        row_starts=row_starts,
    )

    rank = torch.arange(group_topk, device=logits.device, dtype=torch.int32)
    max_valid_groups = min(cols, group_topk)
    valid_counts = torch.minimum(
        group_lengths.to(torch.int32),
        torch.full_like(group_lengths.to(torch.int32), max_valid_groups),
    )
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)
    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        result = expanded
    else:
        result = append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table,
            topk_offsets=topk_offsets,
        )
    if out_rows is None or out_rows == result.shape[0]:
        return result
    padded = torch.full(
        (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
    )
    padded[: result.shape[0]] = result
    return padded


def _topk_from_pooled_history_logits_unfused(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import _topk_unfused

    group_topk = history_group_budget_for_topk(topk, pool_size)
    selected_groups = _topk_unfused(
        logits,
        group_lengths,
        group_topk,
        row_starts=row_starts,
        topk_op=torch.topk,
        topk_op_kwargs={"dim": -1},
    )
    group_valid = selected_groups >= 0

    page_table_for_rows = page_table
    if page_table_row_index is not None:
        assert page_table is not None
        page_table_for_rows = page_table.index_select(
            0, page_table_row_index.to(dtype=torch.int64, device=page_table.device)
        )

    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table_for_rows,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        result = expanded
    else:
        result = append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table_for_rows,
            topk_offsets=topk_offsets,
        )
    if out_rows is None or out_rows == result.shape[0]:
        return result
    padded = torch.full(
        (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
    )
    padded[: result.shape[0]] = result
    return padded


def kpool_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
    return_compressed: bool = False,
    write_cache: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == INDEX_HEAD_DIM
    assert slot_k.dtype == torch.bfloat16
    assert slot_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.dtype == torch.float32
    assert buf.dtype == torch.uint8
    assert pool.page_size == BLOCK_SIZE_K
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert loc.dtype == torch.int64
    assert write_cache or return_compressed

    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        assert not return_compressed
        write_mask = write_mask.contiguous()
        has_write_mask = True

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, slot_k.shape[2]),
                    dtype=torch.float8_e4m3fn,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], slot_k.shape[2]),
            dtype=torch.float8_e4m3fn,
            device=slot_k.device,
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32
    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=slot_k.shape[2],
        S_OFFSET_NBYTES_IN_PAGE=pool.page_size * pool.index_head_dim,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(slot_k.shape[2]),
    )
    if return_compressed:
        return compressed_k, compressed_scale
    return None


def kpool_decode_update_and_maybe_write_cache(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool = False,
) -> None:
    assert tail_k.ndim == 3
    assert tail_score.shape == tail_k.shape
    assert tail_k.shape[1] == pool.index_kpool + pool.tail_extra_slots
    assert tail_k.shape[2] == INDEX_HEAD_DIM
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert slot_score.shape == key.shape
    assert ape.shape == (pool.index_kpool, INDEX_HEAD_DIM)
    assert tail_k.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert slot_score.dtype == tail_score.dtype
    assert ape.dtype == torch.float32
    assert buf.dtype == torch.uint8
    assert pool.page_size == BLOCK_SIZE_K
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert tail_k.is_contiguous()
    assert tail_score.is_contiguous()

    batch = key.shape[0]
    if batch == 0:
        return

    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    assert req_pool_indices.shape[0] >= batch
    assert positions.shape[0] >= batch
    assert seq_lens.shape[0] >= batch
    assert out_cache_loc.shape[0] >= batch
    assert block_tables.ndim == 2
    assert block_tables.shape[0] >= batch

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    _kpool_decode_update_and_maybe_write_cache_kernel[(batch,)](
        buf_fp8,
        buf_fp32,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        REQ_POOL_SIZE=tail_k.shape[0],
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=tail_k.shape[2],
        BLOCK_TABLE_COLS=block_tables.shape[1],
        S_OFFSET_NBYTES_IN_PAGE=pool.slots_per_page * pool.index_head_dim,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(tail_k.shape[2]),
        SLOTS_PER_PAGE=pool.slots_per_page,
    )


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        max_score = tl.maximum(max_score, score)

    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)

    if ROUND_SCALE:
        log_val = tl.log2(absmax * fp8_max_inv)
        scale = tl.exp2(tl.ceil(log_val))
    else:
        scale = absmax * fp8_max_inv

    quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, fp8_min), fp8_max)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = _kpool_cache_k_offsets(
            loc_page_index,
            loc_token_offset_in_page,
            offs,
            BUF_NUMEL_PER_PAGE,
            HEAD_DIM,
            PRESHUFFLE_TILE,
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )

        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)
    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            quantized,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


@triton.jit
def _kpool_decode_update_and_maybe_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_k_ptr,
    tail_score_ptr,
    key_ptr,
    slot_score_ptr,
    ape_ptr,
    block_tables_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    REQ_POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TABLE_COLS: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < REQ_POOL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), REQ_POOL_SIZE - 1)

    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)

    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % TAIL_SIZE

    key = tl.load(
        key_ptr + row * key_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    if pos_valid & (slot == POOL_SIZE - 1):
        pool_logical_start = safe_pos - slot
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            k_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + phys * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_current, key, k_buf)
            acc += k * prob

        x = (acc / denom).to(tl.bfloat16).to(tl.float32)
        x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1.0 / fp8_max
        absmax = tl.max(tl.abs(x), axis=0)
        absmax = tl.maximum(absmax, 1e-4)

        if ROUND_SCALE:
            log_val = tl.log2(absmax * fp8_max_inv)
            scale = tl.exp2(tl.ceil(log_val))
        else:
            scale = absmax * fp8_max_inv

        quantized = x / scale
        quantized = tl.minimum(tl.maximum(quantized, fp8_min), fp8_max)

        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = pool_page_group * POOL_SIZE
        token_page_row = tl.minimum(tl.maximum(token_page_row, 0), BLOCK_TABLE_COLS - 1)
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1,
        )
        loc_page_index = packed_page.to(tl.int64)
        loc_token_offset_in_page = pool_id % SLOTS_PER_PAGE
        out_k_offsets = _kpool_cache_k_offsets(
            loc_page_index,
            loc_token_offset_in_page,
            offs,
            BUF_NUMEL_PER_PAGE,
            HEAD_DIM,
            PRESHUFFLE_TILE,
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )

        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale)

    tail_k_offset = req * tail_k_stride_0 + phys_slot * tail_k_stride_1 + offs
    tail_score_offset = (
        req * tail_score_stride_0 + phys_slot * tail_score_stride_1 + offs
    )
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


@triton.jit
def _hadamard_quantize_fp8(acc, denom, ROUND_SCALE: tl.constexpr):
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_max_inv = 1.0 / 448.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return quantized, scale


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_WRITE_MASK:
        if not tl.load(write_mask_ptr + row):
            return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)
    tail_base = tl.load(tail_logical_base_ptr + row)

    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            phys = (tail_base + slot) % TAIL_SIZE
            off = req * tail_stride_0 + phys * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(tail_k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        else:
            off = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(chunk_k_ptr + off, mask=mask, other=0.0).to(tl.float32)

        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        new_m = tl.maximum(m, score)
        rescale = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        denom = denom * rescale + prob
        acc = acc * rescale + k * prob
        m = new_m

    quantized, scale = _hadamard_quantize_fp8(acc, denom, ROUND_SCALE)

    loc = tl.load(loc_ptr + row)
    loc_page_index = loc // SLOTS_PER_PAGE
    loc_token_offset_in_page = loc % SLOTS_PER_PAGE
    out_k_offsets = _kpool_cache_k_offsets(
        loc_page_index,
        loc_token_offset_in_page,
        offs,
        BUF_NUMEL_PER_PAGE,
        HEAD_DIM,
        PRESHUFFLE_TILE,
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
    tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_assemble_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
) -> None:
    pool_size = pool.index_kpool
    n_pools = req_pool_idx.shape[0]
    if n_pools == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    slots_per_page = pool.slots_per_page

    _kpool_assemble_softmax_rotate_write_cache_kernel[(n_pools,)](
        buf_fp8,
        buf_fp32,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
    )


def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    pool_size = pool.index_kpool
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    _scatter_kpool_tail_updates_kernel[(n_rows, pool_size)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


@triton.jit
def _scatter_kpool_tail_updates_kernel(
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    dst_logical_start_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)

    n_w = tl.load(n_write_ptr + row)
    if slot >= n_w:
        return

    req = tl.load(req_pool_idx_ptr + row)
    dst_logical_start = tl.load(dst_logical_start_ptr + row)
    src_off = tl.load(chunk_src_start_ptr + row) + slot

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    k = tl.load(chunk_k_ptr + src_off * chunk_stride_0 + offs, mask=mask)
    s = tl.load(chunk_score_ptr + src_off * chunk_stride_0 + offs, mask=mask)

    dst = (
        req * tail_stride_0
        + ((dst_logical_start + slot) % TAIL_SIZE) * tail_stride_1
        + offs
    )
    tl.store(tail_k_ptr + dst, k, mask=mask)
    tl.store(tail_score_ptr + dst, s, mask=mask)


@triton.jit
def _pack_pool_slots_to_payload_kernel(
    buf_ptr,
    locs_ptr,
    payload_ptr,
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,
    scale_region_off: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    src = _kpool_cache_k_offsets(
        page,
        slot,
        offs,
        page_bytes,
        head_dim,
        PRESHUFFLE_TILE,
    )
    val = tl.load(buf_ptr + src, mask=mask, other=0).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + offs, val, mask=mask)

    s_offs = tl.arange(0, 4)
    s_src = page_base + scale_region_off + slot * 4 + s_offs
    s_val = tl.load(buf_ptr + s_src).to(tl.uint8)
    tl.store(payload_ptr + row * payload_bytes + head_dim + s_offs, s_val)


@triton.jit
def _select_and_scatter_pool_slots_kernel(
    recv_ptr,
    owner_ptr,
    locs_ptr,
    buf_ptr,
    cp_rank: tl.constexpr,
    payload_bytes: tl.constexpr,
    slots_per_page: tl.constexpr,
    head_dim: tl.constexpr,
    page_bytes: tl.constexpr,
    scale_region_off: tl.constexpr,
    n_total: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    owner = tl.load(owner_ptr + row).to(tl.int64)
    if owner == cp_rank:
        return

    loc = tl.load(locs_ptr + row).to(tl.int64)
    page = loc // slots_per_page
    slot = loc % slots_per_page
    page_base = page * page_bytes
    recv_row_base = (owner * n_total + row) * payload_bytes

    offs = tl.arange(0, BLOCK_D)
    mask = offs < head_dim
    val = tl.load(recv_ptr + recv_row_base + offs, mask=mask, other=0).to(tl.uint8)
    dst = _kpool_cache_k_offsets(
        page,
        slot,
        offs,
        page_bytes,
        head_dim,
        PRESHUFFLE_TILE,
    )
    tl.store(buf_ptr + dst, val, mask=mask)

    s_offs = tl.arange(0, 4)
    s_val = tl.load(recv_ptr + recv_row_base + head_dim + s_offs).to(tl.uint8)
    s_dst = page_base + scale_region_off + slot * 4 + s_offs
    tl.store(buf_ptr + s_dst, s_val)


def all_gather_and_scatter_pool_slots(
    buf: torch.Tensor,
    local_locs: torch.Tensor,
    owner_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    slots_per_page: int,
) -> None:
    from sglang.srt.layers.dp_attention import attn_cp_all_gather_into_tensor

    assert buf.is_contiguous()
    n_total = local_locs.shape[0]
    if n_total == 0 or cp_size <= 1:
        return

    head_dim = INDEX_HEAD_DIM
    payload_bytes = head_dim + 4
    scale_region_off = slots_per_page * head_dim
    page_bytes = buf.shape[1]
    device = buf.device

    send_payload = torch.empty(
        (n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    _pack_pool_slots_to_payload_kernel[(n_total,)](
        buf,
        local_locs,
        send_payload,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    recv = torch.empty(
        (cp_size, n_total, payload_bytes), dtype=torch.uint8, device=device
    )
    attn_cp_all_gather_into_tensor(
        recv.view(cp_size * n_total, payload_bytes), send_payload
    )

    _select_and_scatter_pool_slots_kernel[(n_total,)](
        recv,
        owner_rank,
        local_locs,
        buf,
        cp_rank=cp_rank,
        payload_bytes=payload_bytes,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
        page_bytes=page_bytes,
        scale_region_off=scale_region_off,
        n_total=n_total,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _kpool_write_tail_and_maybe_compress_kernel(
    key_ptr,
    score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    ape_ptr,
    req_pool_indices_ptr,
    write_start_ptr,
    tail_logical_start_ptr,
    write_loc_ptr,
    out_cache_loc_ptr,
    effective_n_ptr,
    buf_fp8_ptr,
    buf_fp32_ptr,
    key_stride_0,
    score_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    write_loc_stride_0,
    N: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    PRESHUFFLE_TILE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_EFFECTIVE_N: tl.constexpr,
    MAX_CLOSED_POOLS: tl.constexpr,
):
    b = tl.program_id(0)
    cache_loc_0 = tl.load(out_cache_loc_ptr + b * N)
    if cache_loc_0 == 0:
        return

    req = tl.load(req_pool_indices_ptr + b)
    write_start = tl.load(write_start_ptr + b)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for i_n in tl.static_range(0, N):
        row = b * N + i_n
        k = tl.load(key_ptr + row * key_stride_0 + offs, mask=dim_mask)
        s = tl.load(score_ptr + row * score_stride_0 + offs, mask=dim_mask)
        phys = (write_start + i_n) % TAIL_SIZE
        dst = req * tail_stride_0 + phys * tail_stride_1 + offs
        tl.store(tail_k_ptr + dst, k, mask=dim_mask)
        tl.store(tail_score_ptr + dst, s, mask=dim_mask)

    if HAS_EFFECTIVE_N:
        gate_n = tl.load(effective_n_ptr + b).to(tl.int32)
    else:
        gate_n = N
    base_pool = write_start // POOL_SIZE
    n_pool = (write_start + gate_n) // POOL_SIZE - base_pool
    if n_pool == 0:
        return

    base0 = tl.load(tail_logical_start_ptr + b)
    for p in tl.static_range(0, MAX_CLOSED_POOLS):
        if p < n_pool:
            base = base0 + p * POOL_SIZE
            m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for slot in tl.static_range(0, POOL_SIZE):
                phys = (base + slot) % TAIL_SIZE
                off = req * tail_stride_0 + phys * tail_stride_1 + offs
                score = tl.load(tail_score_ptr + off, mask=dim_mask, other=0.0).to(
                    tl.float32
                )
                k_ld = tl.load(tail_k_ptr + off, mask=dim_mask, other=0.0).to(
                    tl.float32
                )
                score += tl.load(
                    ape_ptr + slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                new_m = tl.maximum(m, score)
                rescale = tl.exp(m - new_m)
                prob = tl.exp(score - new_m)
                denom = denom * rescale + prob
                acc = acc * rescale + k_ld * prob
                m = new_m

            quantized, scale = _hadamard_quantize_fp8(acc, denom, ROUND_SCALE)
            loc = tl.load(write_loc_ptr + b * write_loc_stride_0 + p)
            loc_page_index = loc // SLOTS_PER_PAGE
            loc_token_offset_in_page = loc % SLOTS_PER_PAGE
            out_k_offsets = _kpool_cache_k_offsets(
                loc_page_index,
                loc_token_offset_in_page,
                offs,
                BUF_NUMEL_PER_PAGE,
                HEAD_DIM,
                PRESHUFFLE_TILE,
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_write_tail_and_maybe_compress(
    pool,
    buf: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    round_scale: bool,
    effective_n_per_batch: Optional[torch.Tensor] = None,
) -> None:
    assert num_draft_tokens > 0
    assert key.dim() == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert score.shape == key.shape
    assert tail_k.shape == tail_score.shape
    assert tail_k.shape[1] == pool.index_kpool + pool.tail_extra_slots
    assert tail_k.shape[2] == INDEX_HEAD_DIM
    assert key.dtype == torch.bfloat16
    assert score.dtype in KPOOL_SCORE_DTYPES
    assert tail_k.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.dtype == torch.float32

    bn = key.shape[0]
    if bn == 0:
        return
    assert bn % num_draft_tokens == 0
    bs = bn // num_draft_tokens
    max_closed_pools = kpool_max_closed_pools(num_draft_tokens, pool.index_kpool)
    assert write_loc.shape == (bs, max_closed_pools), write_loc.shape
    assert write_loc.stride(1) == 1, write_loc.stride()

    key = key.contiguous()
    score = score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    write_start = write_start.contiguous()
    tail_logical_start = tail_logical_start.contiguous()
    write_loc = write_loc.contiguous()
    out_cache_loc = out_cache_loc.contiguous()
    if effective_n_per_batch is not None:
        effective_n_per_batch = effective_n_per_batch.contiguous()

    slots_per_page = pool.slots_per_page
    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    _kpool_write_tail_and_maybe_compress_kernel[(bs,)](
        key,
        score,
        tail_k,
        tail_score,
        ape,
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        effective_n_per_batch,
        buf_fp8,
        buf_fp32,
        key.stride(0),
        score.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        write_loc.stride(0),
        N=num_draft_tokens,
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * pool.index_head_dim,
        PRESHUFFLE_TILE=_preshuffle_tile(),
        ROUND_SCALE=round_scale,
        HAS_EFFECTIVE_N=effective_n_per_batch is not None,
        MAX_CLOSED_POOLS=max_closed_pools,
    )
