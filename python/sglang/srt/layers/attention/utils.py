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


"""
[1,2,3,4,64,65,66]#当前所有的kv cache
req_pool_indices_ptr=[0,1]
page_kernel_lens_ptr=[4,3] #每个batch的长度
result_ptr = [bs,max_kvlen]
[
    [1]
    [2]
]
"""


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    result_ptr,
    max_len,
    num_batches,
):
    BATCH_BLOCK_SIZE: tl.constexpr = 16
    seq_block_size: tl.constexpr = 512

    batch_id = tl.program_id(axis=0)
    seq_block_id = tl.program_id(axis=1)

    batch_idx = batch_id * BATCH_BLOCK_SIZE + tl.arange(0, BATCH_BLOCK_SIZE)
    seq_idx = seq_block_id * seq_block_size + tl.arange(0, seq_block_size)
    batch_mask = batch_idx < num_batches
    req_pool_index = tl.load(req_pool_indices_ptr + batch_idx, mask=batch_mask)
    kv_indices_offset = tl.load(kv_indptr + batch_idx, mask=batch_mask)
    seq_len = tl.load(page_kernel_lens_ptr + batch_idx, mask=batch_mask)

    for b in range(BATCH_BLOCK_SIZE):
        # 报错
        valid_batch = batch_idx[b] < num_batches
        if not valid_batch:
            continue
        pos_mask = seq_idx < seq_len[b]
        if not tl.sum(pos_mask):
            continue
        for s in range(seq_block_size):
            if not pos_mask[s]:
                continue
            i = seq_idx[s]
            token_idx = tl.load(req_to_token_ptr + req_pool_index[b] + i)
            page_idx = (token_idx + 1) // 64
            skip = False
            if i > 0:
                prev_token_idx = tl.load(req_to_token_ptr + req_pool_index[b] + (i - 1))
                prev_page_idx = (prev_token_idx + 1) // 64
                skip = page_idx == prev_page_idx
            if not skip:
                result_offset = batch_idx[b] * max_len + i
                tl.store(result_ptr + result_offset, page_idx)
