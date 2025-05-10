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
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
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
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset <= num_paged * PAGED_SIZE
        mask_out = paged_offset_out <= num_paged

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
def prepare_qk_for_deepseek_mha_triton(
    q,  # [bs, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]
    k,  # [bs, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]
    k_nope,  # [bs, num_local_heads, qk_nope_head_dim]
    q_pe,  # [bs, num_local_heads, qk_rope_head_dim]
    k_pe,  # [bs, 1, qk_rope_head_dim]
    qk_stride_0: tl.constexpr,
    qk_stride_1: tl.constexpr,
    k_nope_stride_0: tl.constexpr,
    k_nope_stride_1: tl.constexpr,
    q_pe_stride_0: tl.constexpr,
    q_pe_stride_1: tl.constexpr,
    k_pe_stride_0: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,  # 64 for DeepSeek-v3
    qk_nope_head_dim: tl.constexpr,  # 128 for DeepSeek-v3
    num_local_heads: tl.constexpr,  # 128 for DeepSeek-v3
):
    # Load k_nope to the first qk_nope_head_dim of k
    # Load q_pe and k_pe to the last qk_rope_head_dim of q and k
    # q_nope has already been in the first qk_nope_head_dim of q
    pid = tl.program_id(axis=0)
    nope_offset = tl.arange(0, qk_nope_head_dim)
    pe_offset = tl.arange(0, qk_rope_head_dim)
    q_ptr = q + pid * qk_stride_0
    k_ptr = k + pid * qk_stride_0

    # k_pe only needs to be loaded once
    k_pe_data = tl.load(k_pe + pid * k_pe_stride_0 + pe_offset)

    for head_idx in range(num_local_heads):
        q_pe_data = tl.load(
            q_pe + pid * q_pe_stride_0 + head_idx * q_pe_stride_1 + pe_offset
        )
        k_nope_data = tl.load(
            k_nope + pid * k_nope_stride_0 + head_idx * k_nope_stride_1 + nope_offset
        )
        tl.store(
            q_ptr + head_idx * qk_stride_1 + qk_nope_head_dim + pe_offset, q_pe_data
        )
        tl.store(
            k_ptr + head_idx * qk_stride_1 + qk_nope_head_dim + pe_offset, k_pe_data
        )
        tl.store(k_ptr + head_idx * qk_stride_1 + nope_offset, k_nope_data)
