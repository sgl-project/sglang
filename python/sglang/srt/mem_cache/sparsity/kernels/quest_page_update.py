import torch
import triton
import triton.language as tl


@triton.jit
def _quest_update_page_representations_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    req_to_token_ptr,
    k_buffer_ptr,
    repr_constructed_ptr,
    last_constructed_page_ptr,
    page_k_min_ptr,
    page_k_max_ptr,
    page_valid_ptr,
    req_pool_indices_stride,
    seq_lens_stride,
    req_to_token_stride_r,
    req_to_token_stride_t,
    k_buffer_stride_t,
    k_buffer_stride_h,
    k_buffer_stride_d,
    num_k_tokens,
    num_pool_pages,
    PAGE_SIZE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head = tl.program_id(1)

    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    constructed = tl.load(repr_constructed_ptr + req_idx)
    tracked_page = tl.load(last_constructed_page_ptr + req_idx).to(tl.int64)
    end_page = seq_len // PAGE_SIZE
    logical_page = tl.where(constructed, tracked_page, 0)

    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, BLOCK_D)
    token_mask = token_offsets < PAGE_SIZE
    dim_mask = dim_offsets < HEAD_DIM

    while logical_page < end_page:
        logical_tokens = logical_page * PAGE_SIZE + token_offsets
        physical_tokens = tl.load(
            req_to_token_ptr
            + req_idx * req_to_token_stride_r
            + logical_tokens * req_to_token_stride_t,
            mask=token_mask,
            other=0,
        ).to(tl.int64)

        # Match the portable implementation for malformed table entries while
        # loading each token through req_to_token (physical tokens need not be
        # contiguous within a logical page).
        safe_physical_tokens = tl.minimum(
            tl.maximum(physical_tokens, 0), num_k_tokens - 1
        )
        key_offsets = (
            safe_physical_tokens[:, None] * k_buffer_stride_t
            + kv_head * k_buffer_stride_h
            + dim_offsets[None, :] * k_buffer_stride_d
        )
        load_mask = token_mask[:, None] & dim_mask[None, :]
        keys = tl.load(k_buffer_ptr + key_offsets, mask=load_mask, other=0.0).to(
            tl.float32
        )
        page_min = tl.min(tl.where(load_mask, keys, float("inf")), axis=0)
        page_max = tl.max(tl.where(load_mask, keys, -float("inf")), axis=0)

        first_physical_token = tl.load(
            req_to_token_ptr
            + req_idx * req_to_token_stride_r
            + logical_page * PAGE_SIZE * req_to_token_stride_t
        ).to(tl.int64)
        physical_page = tl.minimum(
            tl.maximum(first_physical_token // PAGE_SIZE, 0), num_pool_pages - 1
        )
        output_offsets = (
            physical_page * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dim_offsets
        )
        tl.store(
            page_k_min_ptr + output_offsets,
            page_min,
            mask=dim_mask,
        )
        tl.store(
            page_k_max_ptr + output_offsets,
            page_max,
            mask=dim_mask,
        )
        tl.store(page_valid_ptr + physical_page, 1, mask=kv_head == 0)

        logical_page += 1


@triton.jit
def _quest_advance_page_trackers_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    repr_constructed_ptr,
    last_constructed_page_ptr,
    req_pool_indices_stride,
    seq_lens_stride,
    PAGE_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(req_pool_indices_ptr + batch_idx * req_pool_indices_stride).to(
        tl.int64
    )
    seq_len = tl.load(seq_lens_ptr + batch_idx * seq_lens_stride).to(tl.int64)
    end_page = seq_len // PAGE_SIZE
    constructed = tl.load(repr_constructed_ptr + req_idx)
    tracked_page = tl.load(last_constructed_page_ptr + req_idx).to(tl.int64)
    start_page = tl.where(constructed, tracked_page, 0)
    update = start_page < end_page

    tl.store(repr_constructed_ptr + req_idx, 1, mask=update)
    tl.store(last_constructed_page_ptr + req_idx, end_page, mask=update)


def quest_update_page_representations_(
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    k_buffer: torch.Tensor,
    repr_constructed: torch.Tensor,
    last_constructed_page: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    page_valid: torch.Tensor,
    page_size: int,
    *,
    advance_trackers: bool,
) -> None:
    """Update complete Quest pages without reading device state on the host."""
    if not req_pool_indices.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest page update requires NVIDIA CUDA tensors")
    if req_pool_indices.ndim != 1 or seq_lens.shape != req_pool_indices.shape:
        raise ValueError("Quest request indices and sequence lengths must be 1D")
    if req_pool_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest request indices must use int32 or int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest sequence lengths must use int32 or int64")
    if req_to_token.ndim != 2 or req_to_token.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest req_to_token must be a 2D integer tensor")
    if k_buffer.ndim != 3 or k_buffer.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Quest key buffer must be a 3D floating-point tensor")
    if page_k_min.ndim != 3 or page_k_max.shape != page_k_min.shape:
        raise ValueError("Quest page min/max pools must have matching 3D shapes")
    if page_k_min.dtype != torch.float32 or page_k_max.dtype != torch.float32:
        raise ValueError("Quest page min/max pools must use float32")
    if not page_k_min.is_contiguous() or not page_k_max.is_contiguous():
        raise ValueError("Quest page min/max pools must be contiguous")

    num_pool_pages, kv_heads, head_dim = page_k_min.shape
    if k_buffer.shape[1:] != (kv_heads, head_dim):
        raise ValueError("Quest key buffer and representation dimensions must match")
    if (
        page_valid.shape != (num_pool_pages,)
        or page_valid.dtype != torch.bool
        or not page_valid.is_contiguous()
    ):
        raise ValueError("Quest page validity must be a contiguous bool vector")
    if (
        repr_constructed.ndim != 1
        or repr_constructed.dtype != torch.bool
        or not repr_constructed.is_contiguous()
    ):
        raise ValueError("Quest constructed tracker must be a contiguous bool vector")
    if (
        last_constructed_page.shape != repr_constructed.shape
        or last_constructed_page.dtype not in (torch.int32, torch.int64)
        or not last_constructed_page.is_contiguous()
    ):
        raise ValueError(
            "Quest last-page tracker must be a matching contiguous integer vector"
        )
    if page_size <= 0:
        raise ValueError(f"Quest page size must be positive, got {page_size}")
    if req_to_token.shape[1] < page_size:
        raise ValueError("Quest req_to_token width must hold at least one page")
    if k_buffer.shape[0] == 0 or num_pool_pages == 0:
        raise ValueError("Quest key and representation pools cannot be empty")
    if kv_heads <= 0 or head_dim <= 0:
        raise ValueError("Quest representation dimensions must be positive")

    tensors = (
        seq_lens,
        req_to_token,
        k_buffer,
        repr_constructed,
        last_constructed_page,
        page_k_min,
        page_k_max,
        page_valid,
    )
    if any(tensor.device != req_pool_indices.device for tensor in tensors):
        raise ValueError("Quest page update tensors must share one CUDA device")

    batch_size = req_pool_indices.numel()
    if batch_size == 0:
        return

    block_t = triton.next_power_of_2(page_size)
    block_d = triton.next_power_of_2(head_dim)
    _quest_update_page_representations_kernel[(batch_size, kv_heads)](
        req_pool_indices,
        seq_lens,
        req_to_token,
        k_buffer,
        repr_constructed,
        last_constructed_page,
        page_k_min,
        page_k_max,
        page_valid,
        req_pool_indices.stride(0),
        seq_lens.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        k_buffer.shape[0],
        num_pool_pages,
        PAGE_SIZE=page_size,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )

    # Keep tracker writes in a following launch: updating from one KV-head
    # program inside the representation kernel could race with another head
    # that has not read last_constructed_page yet.
    if advance_trackers:
        _quest_advance_page_trackers_kernel[(batch_size,)](
            req_pool_indices,
            seq_lens,
            repr_constructed,
            last_constructed_page,
            req_pool_indices.stride(0),
            seq_lens.stride(0),
            PAGE_SIZE=page_size,
            num_warps=1,
        )
