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

        # needs to be revised     
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
        If max_pages is constant(Eg: 4),
        - kv_indptr (passed in as input arg): [0,4,8,..] (note that 4 is the max_pages)
        - kv_indices = [5, 6, -1, -1,
                        10, -1, -1, -1]
        If there is no max_pages, it's going to be compressed -
        - kv_indptr (passed in as input arg): [0,3,4]
        - kv_indices = [5,6,10]
        This allows attention kernels to directly access the correct KV cache
        entries for each request's tokens.
    """
    BLOCK_SIZE: tl.constexpr = 512
    NUM_PAGES_PER_BLOCK: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)
    num_pages = tl.cdiv(kv_end - kv_start, PAGE_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    
    for i in range(num_pages_loop):
        page_offsets = (
            tl.arange(0, NUM_PAGES_PER_BLOCK).to(tl.int64) + i * NUM_PAGES_PER_BLOCK
            ) * PAGE_SIZE
        
        out_page_offsets = tl.arange(0, NUM_PAGES_PER_BLOCK) + i * NUM_PAGES_PER_BLOCK

        mask = page_offsets < (kv_end - kv_start)
        mask_out = out_page_offsets < num_pages 

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + page_offsets,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + out_page_offsets, data // PAGE_SIZE, mask=mask_out)


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
    """
    Create page-based KV indices for FlashMLA attention backend.
    
    This Triton kernel builds a page table that maps from logical request/token
    coordinates to page indices in the global KV cache pool. It's used by FlashMLA
    attention backends (like CutlassMLABackend) to efficiently access paged KV cache data.
    
    Unlike create_flashinfer_kv_indices_triton, this kernel converts token locations
    to page indices by dividing by PAGED_SIZE, creating a 2D page table structure.
    
    The kernel processes each request in parallel and converts the req_to_token
    lookup table into a page table where each entry represents a page index.
    
    Args:
        req_to_token_ptr: Pointer to the req_to_token lookup table.
                         Shape: [max_batch, max_context_len]
                         Maps (pool_index, token_position) -> global_token_location
        req_pool_indices_ptr: Pointer to array mapping batch requests to pool indices.
                             Shape: [batch_size]
                             req_pool_indices[i] = pool index for request i in batch
        page_kernel_lens_ptr: Pointer to array containing sequence lengths for each request.
                             Shape: [batch_size]
                             page_kernel_lens[i] = number of tokens for request i
        kv_start_idx: Pointer to array containing start offsets for each request.
                     Can be None. If provided, adds offset to token positions.
        kv_indices_ptr: Pointer to output array where page indices will be stored.
                       Shape: [batch_size, max_pages]
                       Output: 2D page table where each row contains page indices for a request
        req_to_token_ptr_stride: Stride for the second dimension of req_to_token.
                                Equal to max_context_len.
        kv_indices_ptr_stride: Stride for the second dimension of kv_indices.
                              Equal to max_pages.
        PAGED_SIZE: Number of tokens per page. Default is 64 for FlashMLA.
                   CutlassMLABackend uses 128, FlashMLABackend uses 64.
    
    Example:
        If we have:
        - req_pool_indices = [0, 1] (request 0 uses pool 0, request 1 uses pool 1)
        - page_kernel_lens = [5, 3] (request 0 has 5 tokens, request 1 has 3 tokens)
        - req_to_token = [[16, 17, 18, 19, 20, -1], [32, 33, 34, -1, -1, -1]]
        - PAGED_SIZE = 2
        
        The kernel will output:
        - kv_indices = [[8, 9, 10, -1, -1], [16, 17, -1, -1, -1]]
        - Page indices: 16//2=8, 17//2=8, 18//2=9, 19//2=9, 20//2=10, etc.
        
        This creates a page table where each entry indicates which page contains
        the corresponding tokens, allowing MLA attention kernels to efficiently
        access paged KV cache data.
    """
    # Define block size for efficient parallel processing (larger than FlashInfer version)
    BLOCK_SIZE: tl.constexpr = 4096
    # Number of pages processed per block
    NUM_PAGE_PER_BLOCK: tl.constexpr = 64
    # Get the program ID - each program processes one request in the batch
    pid = tl.program_id(axis=0)

    # Step 1: Find which pool this request uses (batch request -> pool index)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    # Step 2: Calculate the token range for this request
    kv_start = 0
    kv_end = 0
    # If kv_start_idx is provided, use it as the starting offset
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    # Add the sequence length to get the end position
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    # Step 3: Calculate page-related parameters
    # Number of pages needed for this request's tokens
    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    # Number of block iterations needed to process all pages
    num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)

    # Step 4: Process pages in blocks for efficiency
    for i in range(num_pages_loop):
        # Calculate page offsets for the current block
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        # Calculate output indices for storing page indices
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        # Step 5: Create masks to handle boundary conditions
        # Mask for valid token positions within the page range
        mask = paged_offset < num_paged * PAGED_SIZE
        # Mask for valid output positions in the page table
        mask_out = paged_offset_out < num_paged

        # Step 6: Load token locations from req_to_token table
        # This accesses: req_to_token[req_pool_index][kv_start + paged_offset]
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride  # Move to correct pool row
            + kv_start                                  # Move to start of this request's tokens
            + paged_offset,                             # Move to page-aligned token positions
            mask=mask,                                  # Only load valid positions
        )
        
        # Step 7: Convert token locations to page indices and store in 2D table
        # data // PAGED_SIZE converts token location to page index
        # This creates a 2D page table: [request_id][page_position] -> page_index
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,  # Convert token location to page index
            mask=mask_out,       # Only store valid page positions
        )
