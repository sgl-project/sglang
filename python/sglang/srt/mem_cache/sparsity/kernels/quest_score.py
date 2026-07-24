import torch
import triton
import triton.language as tl


@triton.jit
def _quest_page_score_kernel(
    queries_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    physical_pages_ptr,
    active_mask_ptr,
    history_page_counts_ptr,
    output_ptr,
    num_pages,
    num_pool_pages,
    APPLY_RETRIEVAL_MASK: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    physical_page_raw = tl.load(physical_pages_ptr + batch_idx * num_pages + page_idx)
    page_in_bounds = (physical_page_raw >= 0) & (physical_page_raw < num_pool_pages)
    page_is_selected = page_in_bounds
    if APPLY_RETRIEVAL_MASK:
        request_is_active = tl.load(active_mask_ptr + batch_idx).to(tl.int1)
        history_page_count = tl.load(history_page_counts_ptr + batch_idx)
        page_is_selected &= request_is_active & (page_idx < history_page_count)
    physical_page = tl.where(page_in_bounds, physical_page_raw, 0)
    page_is_valid = tl.load(
        page_valid_ptr + physical_page, mask=page_is_selected, other=0
    )

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = (dim_offsets < HEAD_DIM) & page_is_selected & page_is_valid
    page_bound = 0.0

    for kv_head in range(KV_HEADS):
        key_offsets = (
            physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dim_offsets
        )
        key_min = tl.load(page_k_min_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        key_max = tl.load(page_k_max_ptr + key_offsets, mask=dim_mask, other=0.0).to(
            tl.float32
        )

        query_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for group_idx in range(GROUP_SIZE):
            query_head = kv_head * GROUP_SIZE + group_idx
            query_offsets = (
                batch_idx * Q_HEADS * HEAD_DIM + query_head * HEAD_DIM + dim_offsets
            )
            query_sum += tl.load(
                queries_ptr + query_offsets, mask=dim_mask, other=0.0
            ).to(tl.float32)

        # Match the portable Quest fallback: average the query heads sharing a
        # KV head, round to the query dtype, then sum bounds over all KV heads.
        query = (query_sum / GROUP_SIZE).to(queries_ptr.dtype.element_ty).to(tl.float32)
        bound_keys = tl.where(query >= 0, key_max, key_min)
        page_bound += tl.sum(query * bound_keys, axis=0)

    score = tl.where(page_is_selected & page_is_valid, page_bound, -float("inf"))
    tl.store(output_ptr + batch_idx * num_pages + page_idx, score)


def quest_page_scores(
    queries: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    physical_pages: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    history_page_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Quest's per-page score without materializing intermediates.

    Out-of-range physical pages are masked to ``-inf``. When ``active_mask``
    and ``history_page_counts`` are provided, inactive requests and logical
    recent/padding pages are masked in the same kernel. The representation
    tensors are persistent contiguous pools in Quest; they are deliberately
    not copied here because doing so would dominate scoring.
    """
    if not queries.is_cuda:
        raise ValueError("quest_page_scores requires CUDA tensors")
    if page_k_min.dim() != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError(
            "Quest page min/max tensors must have matching [pages, heads, dim] shapes"
        )
    if page_k_min.dtype != torch.float32 or page_k_max.dtype != torch.float32:
        raise ValueError("Quest page min/max tensors must use float32")
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest page min/max tensors must be contiguous")
    if (
        page_valid.dim() != 1
        or page_valid.shape[0] != page_k_min.shape[0]
        or page_valid.dtype != torch.bool
    ):
        raise ValueError("Quest page validity must be a bool tensor of shape [pages]")
    if not page_valid.is_contiguous():
        raise ValueError("Quest page validity tensor must be contiguous")
    if physical_pages.dim() != 2 or physical_pages.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "Quest physical pages must be an int32/int64 [batch, pages] tensor"
        )
    if any(
        tensor.device != queries.device
        for tensor in (page_k_min, page_k_max, page_valid, physical_pages)
    ):
        raise ValueError("Quest score tensors must be on the same CUDA device")

    apply_retrieval_mask = active_mask is not None or history_page_counts is not None
    if apply_retrieval_mask:
        if active_mask is None or history_page_counts is None:
            raise ValueError(
                "Quest active mask and history page counts must be provided together"
            )
        if (
            active_mask.shape != (physical_pages.shape[0],)
            or active_mask.dtype != torch.bool
        ):
            raise ValueError("Quest active mask must be bool with shape [batch]")
        if history_page_counts.shape != (
            physical_pages.shape[0],
        ) or history_page_counts.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "Quest history page counts must be int32/int64 with shape [batch]"
            )
        if (
            active_mask.device != queries.device
            or history_page_counts.device != queries.device
        ):
            raise ValueError("Quest retrieval masks must share the score tensor device")

    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("Quest page representations require positive head dimensions")
    if queries.dim() == 2:
        batch_size, hidden_size = queries.shape
        if hidden_size % head_dim != 0:
            raise ValueError(
                f"Quest query hidden size {hidden_size} not divisible by "
                f"head_dim {head_dim}"
            )
        query_heads = hidden_size // head_dim
        queries = queries.reshape(batch_size, query_heads, head_dim)
    elif queries.dim() == 3:
        batch_size, query_heads, query_head_dim = queries.shape
        if query_head_dim != head_dim:
            raise ValueError(
                f"Quest query head_dim {query_head_dim} does not match {head_dim}"
            )
    else:
        raise ValueError(f"Unsupported query shape for Quest: {queries.shape}")

    if query_heads <= 0:
        raise ValueError("Quest queries require at least one attention head")
    if physical_pages.shape[0] != batch_size:
        raise ValueError(
            f"Quest physical page batch {physical_pages.shape[0]} does not match "
            f"query batch {batch_size}"
        )
    if query_heads % kv_heads != 0:
        raise ValueError(
            f"Query heads {query_heads} not divisible by KV heads {kv_heads}"
        )
    if head_dim > 256:
        raise ValueError(
            f"Quest Triton score kernel supports head_dim <= 256, got {head_dim}"
        )
    if not queries.is_contiguous():
        queries = queries.contiguous()
    if not physical_pages.is_contiguous():
        physical_pages = physical_pages.contiguous()
    if apply_retrieval_mask:
        if not active_mask.is_contiguous():
            active_mask = active_mask.contiguous()
        if not history_page_counts.is_contiguous():
            history_page_counts = history_page_counts.contiguous()

    num_pages = physical_pages.shape[1]
    output = torch.empty(
        (batch_size, num_pages), dtype=torch.float32, device=queries.device
    )
    if batch_size == 0 or num_pages == 0:
        return output
    if num_pool_pages == 0:
        raise ValueError("Quest page representation pool cannot be empty")

    block_d = triton.next_power_of_2(head_dim)
    active_mask_arg = active_mask if apply_retrieval_mask else physical_pages
    history_page_counts_arg = (
        history_page_counts if apply_retrieval_mask else physical_pages
    )
    _quest_page_score_kernel[(num_pages, batch_size)](
        queries,
        page_k_min,
        page_k_max,
        page_valid,
        physical_pages,
        active_mask_arg,
        history_page_counts_arg,
        output,
        num_pages,
        num_pool_pages,
        APPLY_RETRIEVAL_MASK=apply_retrieval_mask,
        Q_HEADS=query_heads,
        KV_HEADS=kv_heads,
        GROUP_SIZE=query_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output
