import triton
import triton.language as tl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    """
    Fill kv_indices with token page indices from `req_to_token_pool`, which
    will be used to index into `token_to_kv_pool`.
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    # Load all kv indices for a request
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    BLOCK_SIZE: tl.constexpr = 4096
    NUM_PAGE_PER_BLOCK: tl.constexpr = 64
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)

    for i in range(num_pages_loop):
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


@triton.jit
def create_causal_mask_from_page_triton(
    mask_ptr,  # [qo_len, kv_len]
    cu_qo_indptr,  # [bs + 1], cumulative ranges for each req
    cu_kv_lens_ptr,  # [bs + 1]
    prefix_lens_ptr,  # [bs + 1], indicates shared prefix by all qo in a request
    stride_mask_qo,
    max_kv_len_per_req: tl.constexpr,  # must be static
):
    """
    Creates causal mask from paged indices to filter out kv for each request.
    Each program handles a row of qo,
    with the prefix kv attended by all queries in a request.
    """
    pid_qo = tl.program_id(axis=0)
    req_id = tl.program_id(axis=1)

    # Find which req this row belongs to
    qo_start = tl.load(cu_qo_indptr + req_id).to(tl.int32)
    qo_end = tl.load(cu_qo_indptr + req_id + 1).to(tl.int32)
    if qo_start <= pid_qo and pid_qo < qo_end:
        kv_start = tl.load(cu_kv_lens_ptr + req_id).to(tl.int32)
        kv_end = tl.load(cu_kv_lens_ptr + req_id + 1).to(tl.int32)
        kv_len_local = kv_end - kv_start
        # mask_offset = pid_qo * stride_mask_qo + kv_start + tl.arange(0, max_kv_len)

        kv_indices_local = tl.arange(0, max_kv_len_per_req)
        is_valid_kv = kv_indices_local < kv_len_local
        kv_pos = kv_indices_local
        qo_idx_local = (pid_qo - qo_start) + tl.load(prefix_lens_ptr + req_id + 1).to(
            tl.int32
        )
        mask = (kv_pos <= qo_idx_local) & is_valid_kv
        mask_offsets = pid_qo * stride_mask_qo + kv_pos + kv_start
        tl.store(mask_ptr + mask_offsets, mask, mask=is_valid_kv)
