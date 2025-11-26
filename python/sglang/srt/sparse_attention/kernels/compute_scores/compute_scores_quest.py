import torch
import triton
import triton.language as tl


@triton.jit
def _compute_quest_score_kernel(
    Q,
    K,
    Out,
    # kv_pages_per_seq,
    req_to_token,
    req_pool_indices,
    kv_pages_num_per_seq,
    # Strides
    kv_pages_per_seq_stride_b,
    q_stride_b,
    q_stride_h,
    k_stride_p,
    k_stride_e,
    k_stride_h,
    scores_stride_b,
    scores_stride_h,
    # Dimensions
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    num_pages: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    # Sink and local pages
    num_sink_pages: tl.constexpr,
    num_local_pages: tl.constexpr,
    # Block sizes and padding
    BLOCK_SIZE_P: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    pid_block = tl.program_id(2)

    num_valid_pages = tl.load(kv_pages_num_per_seq + bid)

    q_head_start_idx = kv_hid * GROUP_SIZE
    q_head_offsets = tl.arange(0, GROUP_SIZE)
    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    q_ptrs = (
        Q
        + bid * q_stride_b
        + (q_head_start_idx + q_head_offsets[:, None]) * q_stride_h
        + dim_offsets[None, :]
    )
    q_load_mask = (q_head_offsets[:, None] < GROUP_SIZE) & (
        dim_offsets[None, :] < HEAD_DIM
    )
    q_matrix = tl.load(
        q_ptrs, mask=q_load_mask, other=0.0
    )  # [PADDED_GROUP_SIZE, HEAD_DIM]

    # page_ptr = kv_pages_per_seq + bid * kv_pages_per_seq_stride_b
    b_offset = tl.load(req_pool_indices + bid)
    if b_offset < 0:
        return
    page_ptr = req_to_token + b_offset * kv_pages_per_seq_stride_b

    out_ptr = Out + bid * scores_stride_b + kv_hid * scores_stride_h + tl.arange(0, 1)
    for i in tl.static_range(0, BLOCK_SIZE_P):
        page_offset = i + pid_block * BLOCK_SIZE_P
        out_ptrs = out_ptr + page_offset
        if (
            page_offset < num_sink_pages
            or page_offset >= num_valid_pages - num_local_pages
        ):
            neg_inf_scores = tl.full([1], -float("inf"), dtype=tl.float32)
            tl.store(out_ptrs, neg_inf_scores)
        else:
            page_indices = tl.load(page_ptr + page_offset * PAGE_SIZE) // PAGE_SIZE

            k_ptrs = K + page_indices * k_stride_p + kv_hid * k_stride_h + dim_offsets
            k_load_mask = dim_offsets < HEAD_DIM
            k_matrix_min = tl.load(k_ptrs, mask=k_load_mask, other=0.0)  # [HEAD_DIM]
            k_matrix_max = tl.load(
                k_ptrs + k_stride_e, mask=k_load_mask, other=0.0
            )  # [HEAD_DIM]

            score = tl.maximum(
                q_matrix * k_matrix_min[None, :], q_matrix * k_matrix_max[None, :]
            )
            final_scores = tl.sum(score)

            tl.store(out_ptrs, final_scores)


def compute_quest_score(
    q: torch.Tensor,
    k: torch.Tensor,
    out: torch.Tensor,
    # kv_pages_per_seq: torch.Tensor = None,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    kv_pages_num_per_seq: torch.Tensor,
    num_sink_pages: int = 0,
    num_local_pages: int = 0,
    page_size: int = 16,
):

    bs = q.shape[0]
    num_pages = out.shape[-1]
    NUM_KV_HEADS = k.shape[2]
    HEAD_DIM = k.shape[-1]

    q = q.view(bs, -1, HEAD_DIM)
    # k = k.squeeze(1)

    NUM_Q_HEADS = q.shape[1]
    GROUP_SIZE = triton.next_power_of_2(NUM_Q_HEADS // NUM_KV_HEADS)

    PADDED_HEAD_DIM = max(16, triton.next_power_of_2(HEAD_DIM))

    BLOCK_SIZE = 16
    grid = (bs, NUM_KV_HEADS, triton.cdiv(num_pages, BLOCK_SIZE))

    _compute_quest_score_kernel[grid](
        q,
        k,
        out,
        # kv_pages_per_seq,
        req_to_token,
        req_pool_indices,
        kv_pages_num_per_seq,
        # Strides
        # kv_pages_per_seq.stride(0),
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        out.stride(0),
        out.stride(1),
        # Dimensions
        NUM_KV_HEADS=NUM_KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        num_pages=num_pages,
        GROUP_SIZE=GROUP_SIZE,
        # Sink and local pages
        num_sink_pages=num_sink_pages,
        num_local_pages=num_local_pages,
        # Padding info
        PADDED_HEAD_DIM=PADDED_HEAD_DIM,
        BLOCK_SIZE_P=BLOCK_SIZE,
        PAGE_SIZE=page_size,
        num_warps=2,
        num_stages=4,
    )
