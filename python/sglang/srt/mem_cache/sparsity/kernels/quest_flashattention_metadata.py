import torch
import triton
import triton.language as tl

QUEST_DIRECT_METADATA_MAX_WIDTH = 1024


@triton.jit
def _quest_count_valid_selected_pages(
    topk_scores_ptr,
    k_per_req_ptr,
    recent_valid_ptr,
    batch_idx,
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    topk_mask = offsets < TOPK_WIDTH
    topk_scores = tl.load(
        topk_scores_ptr + batch_idx * TOPK_WIDTH + offsets,
        mask=topk_mask,
        other=-float("inf"),
    ).to(tl.float32)
    k_per_req = tl.load(k_per_req_ptr + batch_idx).to(tl.int32)
    topk_valid = (
        topk_mask
        & (offsets < k_per_req)
        & (topk_scores > -float("inf"))
        & (topk_scores < float("inf"))
    )

    recent_offsets = offsets - TOPK_WIDTH
    recent_mask = (recent_offsets >= 0) & (recent_offsets < RECENT_WIDTH)
    recent_valid = tl.load(
        recent_valid_ptr + batch_idx * RECENT_WIDTH + recent_offsets,
        mask=recent_mask,
        other=0,
    ).to(tl.int1)
    return tl.sum((topk_valid | (recent_mask & recent_valid)).to(tl.int32), axis=0)


@triton.jit
def _quest_finalize_to_flashattention_metadata_kernel(
    topk_scores_ptr,
    topk_indices_ptr,
    k_per_req_ptr,
    recent_indices_ptr,
    recent_valid_ptr,
    valid_lengths_ptr,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    page_table_ptr,
    page_table_stride_b,
    page_table_stride_p,
    cache_seqlens_ptr,
    cache_seqlens_stride_b,
    cu_seqlens_ptr,
    cu_seqlens_stride_b,
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    if batch_idx < BATCH_SIZE:
        topk_mask = offsets < TOPK_WIDTH
        topk_scores = tl.load(
            topk_scores_ptr + batch_idx * TOPK_WIDTH + offsets,
            mask=topk_mask,
            other=-float("inf"),
        ).to(tl.float32)
        topk_indices = tl.load(
            topk_indices_ptr + batch_idx * TOPK_WIDTH + offsets,
            mask=topk_mask,
            other=0,
        ).to(tl.int32)
        k_per_req = tl.load(k_per_req_ptr + batch_idx).to(tl.int32)
        topk_valid = (
            topk_mask
            & (offsets < k_per_req)
            & (topk_scores > -float("inf"))
            & (topk_scores < float("inf"))
        )

        recent_offsets = offsets - TOPK_WIDTH
        recent_mask = (recent_offsets >= 0) & (recent_offsets < RECENT_WIDTH)
        recent_indices = tl.load(
            recent_indices_ptr + batch_idx * RECENT_WIDTH + recent_offsets,
            mask=recent_mask,
            other=0,
        ).to(tl.int32)
        recent_valid = tl.load(
            recent_valid_ptr + batch_idx * RECENT_WIDTH + recent_offsets,
            mask=recent_mask,
            other=0,
        ).to(tl.int1)

        sentinel = 0x7FFFFFFF
        selected = tl.where(
            topk_valid,
            topk_indices,
            tl.where(recent_mask & recent_valid, recent_indices, sentinel),
        )
        sorted_indices = tl.sort(selected, descending=False)
        valid = sorted_indices != sentinel
        valid_length = tl.sum(valid.to(tl.int32), axis=0)

        use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b)
        active_valid_length = tl.where(use_sparse, valid_length, 0)
        tl.store(valid_lengths_ptr + batch_idx, active_valid_length)
        active = active_valid_length > 0
        write_mask = active & (offsets < OUTPUT_WIDTH) & valid
        req_idx = tl.load(
            req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b,
            mask=active,
            other=0,
        ).to(tl.int64)
        token_offsets = sorted_indices.to(tl.int64) * PAGE_SIZE
        first_tokens = tl.load(
            req_to_token_ptr
            + req_idx * req_to_token_stride_b
            + token_offsets * req_to_token_stride_t,
            mask=write_mask,
            other=0,
        ).to(tl.int64)
        physical_pages = first_tokens // PAGE_SIZE
        tl.store(
            page_table_ptr
            + batch_idx * page_table_stride_b
            + offsets * page_table_stride_p,
            physical_pages.to(tl.int32),
            mask=write_mask,
        )

    if UPDATE_LENGTHS and batch_idx == BATCH_SIZE:
        cumulative_length = 0
        for idx in range(BATCH_SIZE):
            seq_len = tl.load(seq_lens_ptr + idx * seq_lens_stride_b).to(tl.int64)
            row_valid_length = _quest_count_valid_selected_pages(
                topk_scores_ptr,
                k_per_req_ptr,
                recent_valid_ptr,
                idx,
                TOPK_WIDTH,
                RECENT_WIDTH,
                BLOCK_SIZE,
            ).to(tl.int64)
            row_sparse = tl.load(sparse_mask_ptr + idx * sparse_mask_stride_b)
            row_active = row_sparse & (row_valid_length > 0)
            last_page_length = tl.where(seq_len > 0, (seq_len - 1) % PAGE_SIZE + 1, 0)
            sparse_seq_len = (row_valid_length - 1) * PAGE_SIZE + last_page_length
            cache_seq_len = tl.where(row_active, sparse_seq_len, seq_len).to(tl.int32)

            tl.store(cache_seqlens_ptr + idx * cache_seqlens_stride_b, cache_seq_len)
            tl.store(cu_seqlens_ptr + idx * cu_seqlens_stride_b, cumulative_length)
            cumulative_length += cache_seq_len

        tl.store(cu_seqlens_ptr + BATCH_SIZE * cu_seqlens_stride_b, cumulative_length)


def quest_finalize_to_flashattention_metadata_(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    k_per_req: torch.Tensor,
    recent_indices: torch.Tensor,
    recent_valid: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    *,
    update_lengths: bool,
) -> None:
    """Finalize Quest top-k output directly into FlashAttention metadata."""
    if not topk_scores.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest direct FlashAttention metadata requires CUDA tensors")
    if topk_scores.ndim != 2 or topk_indices.shape != topk_scores.shape:
        raise ValueError("Quest top-k scores and indices must have matching 2D shapes")
    batch_size, topk_width = topk_scores.shape
    if recent_indices.ndim != 2 or recent_indices.shape[0] != batch_size:
        raise ValueError("Quest recent indices must have shape [batch, recent]")
    if recent_valid.shape != recent_indices.shape:
        raise ValueError("Quest recent validity must match recent indices")
    if k_per_req.shape != (batch_size,):
        raise ValueError("Quest per-request k must have shape [batch]")
    recent_width = recent_indices.shape[1]
    output_width = topk_width + recent_width
    if output_width <= 0 or output_width > QUEST_DIRECT_METADATA_MAX_WIDTH:
        raise ValueError(
            f"Quest direct metadata width must be in [1, "
            f"{QUEST_DIRECT_METADATA_MAX_WIDTH}], got {output_width}"
        )
    if valid_lengths.shape != (batch_size,):
        raise ValueError("valid_lengths must have shape [batch]")
    if sparse_mask.shape != (batch_size,):
        raise ValueError("sparse_mask must have shape [batch]")
    if seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have shape [batch]")
    if req_pool_indices.shape != (batch_size,):
        raise ValueError("req_pool_indices must have shape [batch]")
    if req_to_token.ndim != 2:
        raise ValueError("req_to_token must be a two-dimensional tensor")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, pages]")
    if page_table.shape[1] < output_width:
        raise ValueError(
            f"page_table width {page_table.shape[1]} is smaller than "
            f"selection width {output_width}"
        )
    if cache_seqlens_int32.shape != (batch_size,):
        raise ValueError("cache_seqlens_int32 must have shape [batch]")
    if cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch + 1]")
    if topk_scores.dtype != torch.float32:
        raise ValueError("Quest direct metadata top-k scores must use float32")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest direct metadata top-k indices must use int32 or int64")
    if k_per_req.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest direct metadata k-per-request must use int32 or int64")
    if valid_lengths.dtype != torch.int32:
        raise ValueError("Quest direct metadata valid lengths must use int32")
    if recent_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest recent indices must use int32 or int64")
    if recent_valid.dtype != torch.bool:
        raise ValueError("Quest recent validity must use bool")
    if sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must use bool")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must use int32 or int64")
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_pool_indices must use int32 or int64")
    if req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_to_token must use int32 or int64")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must use int32")
    if cache_seqlens_int32.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("FlashAttention sequence metadata must use int32")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    tensors = (
        topk_indices,
        k_per_req,
        recent_indices,
        recent_valid,
        valid_lengths,
        sparse_mask,
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        cache_seqlens_int32,
        cu_seqlens_k,
    )
    if any(tensor.device != topk_scores.device for tensor in tensors):
        raise ValueError("Quest direct metadata tensors must share one device")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            topk_scores,
            topk_indices,
            k_per_req,
            recent_indices,
            recent_valid,
            valid_lengths,
        )
    ):
        raise ValueError(
            "Quest direct metadata top-k/recent/output tensors must be contiguous"
        )
    if batch_size == 0:
        return

    block_size = triton.next_power_of_2(output_width)
    grid = (batch_size + (1 if update_lengths else 0),)
    _quest_finalize_to_flashattention_metadata_kernel[grid](
        topk_scores,
        topk_indices,
        k_per_req,
        recent_indices,
        recent_valid,
        valid_lengths,
        sparse_mask,
        sparse_mask.stride(0),
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        cache_seqlens_int32,
        cache_seqlens_int32.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        TOPK_WIDTH=topk_width,
        RECENT_WIDTH=recent_width,
        OUTPUT_WIDTH=output_width,
        BATCH_SIZE=batch_size,
        PAGE_SIZE=page_size,
        UPDATE_LENGTHS=update_lengths,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@triton.jit
def _quest_update_flashattention_metadata_kernel(
    selected_indices_ptr,
    selected_indices_stride_b,
    selected_indices_stride_p,
    valid_lengths_ptr,
    valid_lengths_stride_b,
    sparse_mask_ptr,
    sparse_mask_stride_b,
    seq_lens_ptr,
    seq_lens_stride_b,
    req_pool_indices_ptr,
    req_pool_indices_stride_b,
    req_to_token_ptr,
    req_to_token_stride_b,
    req_to_token_stride_t,
    page_table_ptr,
    page_table_stride_b,
    page_table_stride_p,
    cache_seqlens_ptr,
    cache_seqlens_stride_b,
    cu_seqlens_ptr,
    cu_seqlens_stride_b,
    max_selected,
    BATCH_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    UPDATE_LENGTHS: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_offsets = tl.program_id(1) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)

    valid_length = tl.load(valid_lengths_ptr + batch_idx * valid_lengths_stride_b).to(
        tl.int32
    )
    use_sparse = tl.load(sparse_mask_ptr + batch_idx * sparse_mask_stride_b)
    active = use_sparse & (valid_length > 0)
    write_mask = active & (page_offsets < max_selected) & (page_offsets < valid_length)

    selected_pages = tl.load(
        selected_indices_ptr
        + batch_idx * selected_indices_stride_b
        + page_offsets * selected_indices_stride_p,
        mask=write_mask,
        other=-1,
    ).to(tl.int64)
    nonnegative_page = selected_pages >= 0
    req_idx = tl.load(
        req_pool_indices_ptr + batch_idx * req_pool_indices_stride_b,
        mask=active,
        other=0,
    ).to(tl.int64)
    token_offsets = tl.maximum(selected_pages, 0) * PAGE_SIZE
    first_tokens = tl.load(
        req_to_token_ptr
        + req_idx * req_to_token_stride_b
        + token_offsets * req_to_token_stride_t,
        mask=write_mask & nonnegative_page,
        other=0,
    ).to(tl.int64)
    physical_pages = tl.where(nonnegative_page, first_tokens // PAGE_SIZE, 0)
    tl.store(
        page_table_ptr
        + batch_idx * page_table_stride_b
        + page_offsets * page_table_stride_p,
        physical_pages.to(tl.int32),
        mask=write_mask,
    )

    # One program computes the small batch prefix sum. Page-table programs are
    # independent, and the following attention launch provides stream ordering.
    if UPDATE_LENGTHS and batch_idx == 0 and tl.program_id(1) == 0:
        cumulative_length = 0
        for idx in range(BATCH_SIZE):
            seq_len = tl.load(seq_lens_ptr + idx * seq_lens_stride_b).to(tl.int64)
            row_valid_length = tl.load(
                valid_lengths_ptr + idx * valid_lengths_stride_b
            ).to(tl.int64)
            row_sparse = tl.load(sparse_mask_ptr + idx * sparse_mask_stride_b)
            row_active = row_sparse & (row_valid_length > 0)
            last_page_length = tl.where(seq_len > 0, (seq_len - 1) % PAGE_SIZE + 1, 0)
            sparse_seq_len = (row_valid_length - 1) * PAGE_SIZE + last_page_length
            cache_seq_len = tl.where(row_active, sparse_seq_len, seq_len).to(tl.int32)

            tl.store(
                cache_seqlens_ptr + idx * cache_seqlens_stride_b,
                cache_seq_len,
            )
            tl.store(
                cu_seqlens_ptr + idx * cu_seqlens_stride_b,
                cumulative_length,
            )
            cumulative_length += cache_seq_len

        tl.store(
            cu_seqlens_ptr + BATCH_SIZE * cu_seqlens_stride_b,
            cumulative_length,
        )


def quest_update_flashattention_metadata_(
    selected_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_size: int,
    *,
    update_lengths: bool,
) -> None:
    """Update fixed-address FA metadata from logical Quest pages."""
    if not selected_indices.is_cuda:
        raise ValueError("Quest FlashAttention metadata kernel requires CUDA tensors")
    if selected_indices.ndim != 2:
        raise ValueError("selected_indices must have shape [batch, pages]")

    batch_size, max_selected = selected_indices.shape
    if valid_lengths.shape != (batch_size,):
        raise ValueError("valid_lengths must have shape [batch]")
    if sparse_mask.shape != (batch_size,):
        raise ValueError("sparse_mask must have shape [batch]")
    if seq_lens.shape != (batch_size,):
        raise ValueError("seq_lens must have shape [batch]")
    if req_pool_indices.shape != (batch_size,):
        raise ValueError("req_pool_indices must have shape [batch]")
    if cache_seqlens_int32.shape != (batch_size,):
        raise ValueError("cache_seqlens_int32 must have shape [batch]")
    if cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch + 1]")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [batch, pages]")
    if page_table.shape[1] < max_selected:
        raise ValueError(
            f"page_table width {page_table.shape[1]} is smaller than "
            f"selection width {max_selected}"
        )
    if req_to_token.ndim != 2:
        raise ValueError("req_to_token must be a two-dimensional tensor")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_indices must use int32 or int64")
    if valid_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("valid_lengths must use int32 or int64")
    if sparse_mask.dtype != torch.bool:
        raise ValueError("sparse_mask must use bool")
    if req_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_to_token must use int32 or int64")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must use int32")
    if cache_seqlens_int32.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("FlashAttention sequence metadata must use int32")

    tensors = (
        valid_lengths,
        sparse_mask,
        seq_lens,
        req_pool_indices,
        req_to_token,
        page_table,
        cache_seqlens_int32,
        cu_seqlens_k,
    )
    if any(tensor.device != selected_indices.device for tensor in tensors):
        raise ValueError("Quest FlashAttention metadata tensors must share one device")
    if batch_size == 0:
        return

    block_pages = 128
    page_blocks = max(triton.cdiv(max_selected, block_pages), 1)
    _quest_update_flashattention_metadata_kernel[(batch_size, page_blocks)](
        selected_indices,
        selected_indices.stride(0),
        selected_indices.stride(1),
        valid_lengths,
        valid_lengths.stride(0),
        sparse_mask,
        sparse_mask.stride(0),
        seq_lens,
        seq_lens.stride(0),
        req_pool_indices,
        req_pool_indices.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        cache_seqlens_int32,
        cache_seqlens_int32.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        max_selected,
        BATCH_SIZE=batch_size,
        PAGE_SIZE=page_size,
        UPDATE_LENGTHS=update_lengths,
        BLOCK_PAGES=block_pages,
        num_warps=4,
    )
