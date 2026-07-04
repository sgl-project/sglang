import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_len",
    ]
)
def _fused_dsa_decode_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_len,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        seq_i32 = seq.to(tl.int32)
        dsa_seq = tl.minimum(seq_i32, dsa_index_topk)

        cu = tl.cumsum(seq_i32, 0)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(cache_seqlens + offs_b, seq_i32, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)
        tl.store(dsa_cache_seqlens + offs_b, dsa_seq, mask=mask_b)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_b, dsa_cu, mask=mask_b)
        return

    num_col_blocks = tl.cdiv(max_len, BLOCK_N)
    page_pid = pid - 1
    row = page_pid // num_col_blocks
    col_block = page_pid - row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row < bs) & (offs_n < max_len)

    req_idx = tl.load(
        req_pool_indices + row * req_pool_indices_stride,
        mask=row < bs,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1 + row * page_table_stride_0 + offs_n * page_table_stride_1,
        vals,
        mask=mask,
    )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + row * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // real_page_size,
            mask=real_mask,
        )


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    real_page_size: int,
) -> None:
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert page_table_1.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        real_page_table = page_table_1

    block_bs = triton.next_power_of_2(bs)
    block_n = 128
    num_col_blocks = triton.cdiv(max_len, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        max_len,
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        BLOCK_BS=block_bs,
        BLOCK_N=block_n,
    )


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "max_seqlen_k",
    ]
)
def _fused_dsa_target_verify_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    paged_mqa_ctx_lens_2d,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    paged_mqa_ctx_lens_stride_0: tl.constexpr,
    paged_mqa_ctx_lens_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_seqlen_k,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    next_n: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    HAS_PAGED_MQA_CTX_LENS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    expanded_size: tl.constexpr = bs * next_n

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32) + next_n
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < expanded_size
        req_row = offs_e // next_n
        draft_off = offs_e - req_row * next_n
        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        expanded_seq = base_seq + draft_off + 1
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)

        if HAS_PAGED_MQA_CTX_LENS:
            tl.store(
                paged_mqa_ctx_lens_2d
                + req_row * paged_mqa_ctx_lens_stride_0
                + draft_off * paged_mqa_ctx_lens_stride_1,
                base_seq + next_n,
                mask=mask_e,
            )
        return

    num_col_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)
    page_pid = pid - 1
    out_row = page_pid // num_col_blocks
    col_block = page_pid - out_row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (out_row < expanded_size) & (offs_n < max_seqlen_k)

    req_row = out_row // next_n
    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=out_row < expanded_size,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1 + out_row * page_table_stride_0 + offs_n * page_table_stride_1,
        vals,
        mask=mask,
    )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + out_row * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // real_page_size,
            mask=real_mask,
        )


def fused_dsa_target_verify_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
    next_n: int,
    paged_mqa_ctx_lens_2d: torch.Tensor = None,
) -> None:
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert page_table_1.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert next_n > 0

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        real_page_table = page_table_1

    has_paged_mqa_ctx_lens = paged_mqa_ctx_lens_2d is not None
    if has_paged_mqa_ctx_lens:
        assert paged_mqa_ctx_lens_2d.is_cuda
        assert paged_mqa_ctx_lens_2d.dtype == torch.int32
        assert paged_mqa_ctx_lens_2d.dim() == 2
        assert paged_mqa_ctx_lens_2d.size(0) == bs
        assert paged_mqa_ctx_lens_2d.size(1) == next_n
    else:
        paged_mqa_ctx_lens_2d = page_table_1

    expanded_size = bs * next_n
    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(expanded_size)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    grid = (1 + expanded_size * num_col_blocks,)

    _fused_dsa_target_verify_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        paged_mqa_ctx_lens_2d,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        paged_mqa_ctx_lens_2d.stride(0) if has_paged_mqa_ctx_lens else 0,
        paged_mqa_ctx_lens_2d.stride(1) if has_paged_mqa_ctx_lens else 0,
        bs,
        max_seqlen_k,
        dsa_index_topk,
        real_page_size,
        next_n,
        has_real_page_table,
        has_paged_mqa_ctx_lens,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_N=block_n,
    )


@triton.jit(
    do_not_specialize=[
        "page_table_stride_0",
        "real_page_table_stride_0",
        "total_len",
        "max_seqlen_k",
    ]
)
def _fused_dsa_draft_extend_metadata_kernel(
    seq_lens,
    extend_seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    seqlens_expanded,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    extend_seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    total_len,
    max_seqlen_k,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    STATIC_EXTEND_LEN: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_EXPANDED: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        cache_seq = seq.to(tl.int32)
        cu = tl.cumsum(cache_seq, 0)

        tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

        offs_e = tl.arange(0, BLOCK_EXPANDED)
        mask_e = offs_e < total_len
        if STATIC_EXTEND_LEN:
            static_qo_len = tl.load(extend_seq_lens).to(tl.int32)
            req_row = offs_e // static_qo_len
            local_off = offs_e - req_row * static_qo_len
            qo_len_for_row = tl.zeros((BLOCK_EXPANDED,), tl.int32) + static_qo_len
        else:
            req_row = tl.full((BLOCK_EXPANDED,), 0, tl.int32)
            local_off = tl.full((BLOCK_EXPANDED,), 0, tl.int32)
            qo_len_for_row = tl.full((BLOCK_EXPANDED,), 1, tl.int32)
            prefix = tl.full((), 0, tl.int32)

            for i in tl.range(0, bs):
                qo_len = tl.load(extend_seq_lens + i * extend_seq_lens_stride).to(
                    tl.int32
                )
                in_row = (offs_e >= prefix) & (offs_e < prefix + qo_len)
                req_row = tl.where(in_row, i, req_row)
                local_off = tl.where(in_row, offs_e - prefix, local_off)
                qo_len_for_row = tl.where(in_row, qo_len, qo_len_for_row)
                prefix += qo_len

        base_seq = tl.load(
            seq_lens + req_row * seq_lens_stride,
            mask=mask_e,
            other=0,
        ).to(tl.int32)
        expanded_seq = base_seq - qo_len_for_row + local_off + 1
        expanded_seq = tl.where(mask_e, expanded_seq, 0)
        dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
        tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)
        return

    num_col_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)
    page_pid = pid - 1
    req_row = page_pid // num_col_blocks
    col_block = page_pid - req_row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)

    qo_len = tl.load(
        extend_seq_lens + req_row * extend_seq_lens_stride,
        mask=req_row < bs,
        other=0,
    ).to(tl.int32)
    if STATIC_EXTEND_LEN:
        prefix = req_row * qo_len
    else:
        prefix = tl.full((), 0, tl.int32)
        for i in tl.range(0, bs):
            prev_qo_len = tl.load(extend_seq_lens + i * extend_seq_lens_stride).to(
                tl.int32
            )
            prefix += tl.where(i < req_row, prev_qo_len, 0)
    offs_r = tl.arange(0, BLOCK_ROWS)
    out_rows = prefix + offs_r
    row_mask = (req_row < bs) & (offs_r < qo_len) & (out_rows < total_len)
    col_mask = offs_n < max_seqlen_k
    has_rows = (req_row < bs) & (qo_len > 0)
    mask = row_mask[:, None] & col_mask[None, :]

    req_idx = tl.load(
        req_pool_indices + req_row * req_pool_indices_stride,
        mask=has_rows,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=col_mask & has_rows,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1
        + out_rows[:, None] * page_table_stride_0
        + offs_n[None, :] * page_table_stride_1,
        vals[None, :],
        mask=mask,
    )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n[None, :] % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + out_rows[:, None] * real_page_table_stride_0
            + real_cols[None, :] * real_page_table_stride_1,
            (vals // real_page_size)[None, :],
            mask=real_mask,
        )


def fused_dsa_draft_extend_metadata(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: torch.Tensor,
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    total_len: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
    max_extend_len: int,
    max_total_len: int,
    static_extend_len: bool = False,
) -> None:
    assert seq_lens.is_cuda
    assert extend_seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert page_table_1.is_cuda
    assert seqlens_expanded.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return
    if total_len == 0:
        cache = seq_lens.to(torch.int32)
        cache_seqlens.copy_(cache)
        cu_seqlens_k[:1].zero_()
        cu_seqlens_k[1 : bs + 1].copy_(torch.cumsum(cache, dim=0, dtype=torch.int32))
        dsa_cu_seqlens_k[:1].zero_()
        return
    assert total_len <= max_total_len
    # Caller-owned graph metadata guarantees each request accepts at most
    # max_extend_len tokens. Avoid checking extend_seq_lens.max() here because
    # that would sync in the replay hot path.
    assert max_extend_len > 0
    assert total_len <= bs * max_extend_len

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        real_page_table = page_table_1

    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(max_total_len)
    block_rows = triton.next_power_of_2(max_extend_len)
    block_n = 128
    num_col_blocks = triton.cdiv(max_seqlen_k, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_draft_extend_metadata_kernel[grid](
        seq_lens,
        extend_seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        extend_seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        total_len,
        max_seqlen_k,
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        static_extend_len,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_ROWS=block_rows,
        BLOCK_N=block_n,
    )
