import torch
import triton
import triton.language as tl


@triton.jit
def proxy_k_tensor_decode_kernel(
    key_cache_ptr,  # [size, num_head, num_dim]
    seq_lens_ptr,  # [bs]
    count_step_ptr,  # [bs]
    req_pool_indices_ptr,  # [bs]
    req_to_token_ptr,  # [max_bs, max_seq_len]
    proxy_k_tensor_ptr,  # [num_pages, 1, num_head, num_dim]
    page_size,
    num_head,
    num_dim,
    max_seq_len,
    accumlation_step,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + req_idx)

    count_step = tl.load(count_step_ptr + req_idx) + 1

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)

    if count_step < accumlation_step:
        if head_idx == 0 and dim_idx == 0:
            tl.store(count_step_ptr + req_idx, count_step)
        return

    update_len = seq_len - accumlation_step
    page_start = update_len // page_size
    page_end = (seq_len + page_size - 1) // page_size

    for page_idx in range(page_start, page_end):
        page_token_start = page_idx * page_size
        page_token_end = tl.minimum((page_idx + 1) * page_size, seq_len)

        sum_val = 0.0
        count_val = 0

        for block_start in range(page_token_start, page_token_end, BLOCK_SIZE):
            block_offsets = tl.arange(0, BLOCK_SIZE)
            token_offsets = block_start + block_offsets
            block_mask = token_offsets < page_token_end

            # 加载token indices
            token_indices = tl.load(
                req_to_token_ptr + req_pool_idx * max_seq_len + token_offsets,
                mask=block_mask,
                other=-1,
            )

            valid_mask = block_mask & (token_indices >= 0)

            key_cache_offsets = (
                token_indices * num_head * num_dim + head_idx * num_dim + dim_idx
            )
            key_values = tl.load(
                key_cache_ptr + key_cache_offsets, mask=valid_mask, other=0.0
            )

            masked_values = tl.where(valid_mask, key_values, 0.0)
            valid_count = tl.sum(tl.where(valid_mask, 1, 0))

            block_sum = tl.sum(masked_values)

            sum_val += block_sum
            count_val += valid_count

        first_token_in_page = tl.load(
            req_to_token_ptr + req_pool_idx * max_seq_len + page_token_start
        )
        physical_page_id = first_token_in_page // page_size

        # proxy_k_tensor shape: [num_pages, 1, num_head, num_dim]
        # 0: average
        avg_val = tl.where(count_val > 0, sum_val / count_val, 0.0)
        avg_offset = (
            physical_page_id * 1 * num_head * num_dim
            + 0 * num_head * num_dim
            + head_idx * num_dim
            + dim_idx
        )

        tl.store(proxy_k_tensor_ptr + avg_offset, avg_val)

    if head_idx == 0 and dim_idx == 0:
        tl.store(count_step_ptr + req_idx, 0)


def proxy_k_tensor_decode(
    key_cache,  # [size, num_head, num_dim]
    seq_lens,  # [bs]
    count_steps,  # [bs]
    req_pool_indices,  # [bs]
    req_to_token,  # [max_bs, max_seq_len]
    page_size,  # int
    accumlation_step,  # int
    proxy_k_tensor,  # [num_pages, 1, num_head, num_dim]
):
    bs = seq_lens.shape[0]
    _, _, num_head, num_dim = proxy_k_tensor.shape
    max_seq_len = req_to_token.shape[1]

    grid = (bs, num_head, num_dim)

    BLOCK_SIZE = 128
    proxy_k_tensor_decode_kernel[grid](
        key_cache,
        seq_lens,
        count_steps,
        req_pool_indices,
        req_to_token,
        proxy_k_tensor,
        page_size,
        num_head,
        num_dim,
        max_seq_len,
        accumlation_step,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def proxy_k_tensor_extend_kernel(
    key_cache_ptr,  # [size, num_head, num_dim]
    seq_lens_ptr,  # [bs]
    prefix_lens_ptr,  # [bs]
    req_pool_indices_ptr,  # [bs]
    req_to_token_ptr,  # [max_bs, max_seq_len]
    proxy_k_tensor_ptr,  # [num_pages, 1, num_head, num_dim]
    page_size,
    num_head,
    num_dim,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = tl.load(prefix_lens_ptr + req_idx)
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)

    extend_len = seq_len - prefix_len

    if extend_len <= 0:
        return

    page_start = prefix_len // page_size
    page_end = (seq_len + page_size - 1) // page_size

    for page_idx in range(page_start, page_end):
        page_token_start = page_idx * page_size
        page_token_end = tl.minimum((page_idx + 1) * page_size, seq_len)

        sum_val = 0.0
        count_val = 0

        for block_start in range(page_token_start, page_token_end, BLOCK_SIZE):
            block_offsets = tl.arange(0, BLOCK_SIZE)
            token_offsets = block_start + block_offsets
            block_mask = token_offsets < page_token_end

            token_indices = tl.load(
                req_to_token_ptr + req_pool_idx * max_seq_len + token_offsets,
                mask=block_mask,
                other=-1,
            )

            valid_mask = block_mask & (token_indices >= 0)

            key_cache_offsets = (
                token_indices * num_head * num_dim + head_idx * num_dim + dim_idx
            )
            key_values = tl.load(
                key_cache_ptr + key_cache_offsets, mask=valid_mask, other=0.0
            )

            masked_values = tl.where(valid_mask, key_values, 0.0)
            valid_count = tl.sum(tl.where(valid_mask, 1, 0))

            block_sum = tl.sum(masked_values)

            sum_val += block_sum
            count_val += valid_count

        first_token_in_page = tl.load(
            req_to_token_ptr + req_pool_idx * max_seq_len + page_token_start
        )
        physical_page_id = first_token_in_page // page_size

        # proxy_k_tensor shape: [num_pages, 1, num_head, num_dim]
        # 0: average
        avg_val = tl.where(count_val > 0, sum_val / count_val, 0.0)
        avg_offset = (
            physical_page_id * 1 * num_head * num_dim
            + 0 * num_head * num_dim
            + head_idx * num_dim
            + dim_idx
        )

        tl.store(proxy_k_tensor_ptr + avg_offset, avg_val)


def proxy_k_tensor_extend(
    key_cache,  # [size, num_head, num_dim]
    seq_lens,  # [bs]
    prefix_lens,  # [bs]
    req_pool_indices,  # [bs]
    req_to_token,  # [max_bs, max_seq_len]
    page_size,  # int
    proxy_k_tensor,  # [num_pages, 1, num_head, num_dim]
):
    bs = seq_lens.shape[0]
    _, _, num_head, num_dim = proxy_k_tensor.shape
    max_seq_len = req_to_token.shape[1]

    grid = (bs, num_head, num_dim)
    BLOCK_SIZE = 128
    proxy_k_tensor_extend_kernel[grid](
        key_cache,
        seq_lens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        proxy_k_tensor,
        page_size,
        num_head,
        num_dim,
        max_seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
