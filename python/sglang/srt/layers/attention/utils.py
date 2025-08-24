import triton
import triton.language as tl

# Keep this in sync with the Triton kernel inside `create_flashmla_kv_indices_triton`.
# Number of pages that the kernel writes per iteration.
# Exposed here so other Python modules can import it instead of hard-coding 64.
TRITON_PAD_NUM_PAGE_PER_BLOCK = 64


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 1,
):
    """
    Create KV indices for FlashInfer attention backend.

    This Triton kernel builds a lookup table that maps from logical request/token
    coordinates to physical token locations in the global KV cache pool. It's used
    by FlashInfer attention backends to efficiently access scattered KV cache data.

    The kernel processes each request in parallel and converts the req_to_token
    lookup table into a flat list of token indices that can be used by attention kernels.

    general idea:
        blocktables/kv_indices_ptr = [batch_size * max_pages(for graph mode with
                                                            fixed number of pages)]
        max_pages = max_context_len / PAGED_SIZE
        kv_indices_ptr will store the flat list of the pages used by each request
    Args:
        Inputs Arguments (non mutable):

        req_to_token_ptr: Request to token location look up table
                         Shape: [max_batch, max_context_len]
        req_pool_indices_ptr: Request to pool index look up table. Each request uses
                             one pool.
                             Shape: [batch_size]
        page_kernel_lens_ptr: sequence lengths per request
                             Shape: [batch_size]
        kv_indptr: Should be computed based on number of pages used by each request.
                   It is used by flashinfer attention kernels to index into the kv_indices_ptr.
                   per request.
                  Shape: [batch_size + 1]
                  kv_indptr[i] = start index in kv_indices for request i
        kv_start_idx: Pointer to array containing start offsets for each request in SGL.
                     Can be None. If provided, adds offset to token positions.

        req_to_token_ptr_stride: Stride for the second dimension of req_to_token.
                                Equal to max_context_len.

        PAGED_SIZE: Number of tokens per page. Default is 1 for FlashInfer.

        Outputs:
        kv_indices_ptr: Pointer to output array where KV indices will be stored.
                    Shape:[total-num-pages],
                    where total_num_pages = sum(seq_lens // PAGED_SIZE)

    Example:
        If we have:
        - req_pool_indices = [0, 1] (request 0 uses pool 0, request 1 uses pool 1)
        - page_kernel_lens = [3, 2] (request 0 has 3 tokens, request 1 has 2 tokens)
        - req_to_token = [[10, 11, 12, -1], [20, 21, -1, -1]] (tokens are the elements
         in radix tree, use them as a pointer to the token location in the kv_indices_ptr)

        The kernel will output:
        If PAGE_SIZE = 1:
        packed
        - kv_indptr (passed in as input arg): [0,3,5]
        - kv_indices = [10, 11, 12, 20, 21]
        padded - max_pages is 10 tokens per req
        - kv_indptr (passed in as input arg): [0,10, 20]
        - kv_indices = [10, 11, 12, -1, -1, -1, -1, -1, -1, -1,
                        20, 21, -1, -1, -1, -1, -1, -1, -1, -1]

        If PAGE_SIZE = 2
        packed:
        - kv_indptr (passed in as input arg): [0,3,4]
        - kv_indices = [5,6,10]
        padded: max_pages is 4
        - kv_indptr (passed in as input arg): [0,4,8,..] (note that 4 is the max_pages)
        - kv_indices = [5, 6, -1, -1,
                        10, -1, -1, -1]
        This allows attention kernels to directly access the correct KV cache
        entries for each request's tokens.
    """
    BLOCK_SIZE: tl.constexpr = 512
    NUM_PAGES_PER_BLOCK: tl.constexpr = BLOCK_SIZE // PAGE_SIZE
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    kv_range = kv_end - kv_start
    num_pages = tl.cdiv(kv_range, PAGE_SIZE)
    num_loops = tl.cdiv(kv_range, BLOCK_SIZE)
    req_to_token_block_start = (
        req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + kv_start
    )
    for i in range(num_loops):
        token_offsets_in_block = (
            tl.arange(0, NUM_PAGES_PER_BLOCK).to(tl.int64) + i * NUM_PAGES_PER_BLOCK
        ) * PAGE_SIZE
        page_offsets_in_block = token_offsets_in_block // PAGE_SIZE
        valid_tokens = token_offsets_in_block < kv_range
        valid_pages = page_offsets_in_block < num_pages
        token_numbers = tl.load(
            req_to_token_block_start + token_offsets_in_block, mask=valid_tokens
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + page_offsets_in_block,
            token_numbers // PAGE_SIZE,  # write the page numbers to kv_indices_ptr
            mask=valid_pages,
        )


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    NUM_PAGE_PER_BLOCK: tl.constexpr = TRITON_PAD_NUM_PAGE_PER_BLOCK,
    PAGED_SIZE: tl.constexpr = 64,
):
    BLOCK_SIZE: tl.constexpr = 4096
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
