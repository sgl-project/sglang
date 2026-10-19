import torch
import triton
import triton.language as tl

PAGE_SIZE = 64


@triton.jit
def _gather_page64_kv_latent_kernel(
    k_cache_ptr,
    block_table_ptr,
    cache_seqlens_ptr,
    kv_out_ptr,
    valid_out_ptr,
    k_cache_stride_t,
    k_cache_stride_d,
    block_table_stride_b,
    block_table_num_pages: tl.constexpr,
    k_cache_num_tokens: tl.constexpr,
    kv_out_stride_b,
    kv_out_stride_t,
    kv_out_stride_d,
    valid_out_stride_b,
    valid_out_stride_t,
    GATHER_LEN: tl.constexpr,
    KV_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE_: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)
    did = tl.program_id(2)

    cache_seqlen = tl.load(cache_seqlens_ptr + bid).to(tl.int32)
    gather_start = tl.maximum(cache_seqlen - GATHER_LEN, 0)
    offs_t = tid * BLOCK_T + tl.arange(0, BLOCK_T)
    logical_token = gather_start + offs_t
    logical_valid = (offs_t < GATHER_LEN) & (logical_token < cache_seqlen)

    logical_page = logical_token // PAGE_SIZE_
    intra_page = logical_token - logical_page * PAGE_SIZE_
    page_table_valid = logical_valid & (logical_page < block_table_num_pages)
    physical_page = tl.load(
        block_table_ptr + bid * block_table_stride_b + logical_page,
        mask=page_table_valid,
        other=-1,
    ).to(tl.int32)
    physical_token = physical_page * PAGE_SIZE_ + intra_page
    physical_valid = (
        page_table_valid & (physical_page >= 0) & (physical_token < k_cache_num_tokens)
    )

    offs_d = did * BLOCK_D + tl.arange(0, BLOCK_D)
    values = tl.load(
        k_cache_ptr
        + physical_token[:, None] * k_cache_stride_t
        + offs_d[None, :] * k_cache_stride_d,
        mask=physical_valid[:, None] & (offs_d[None, :] < KV_DIM),
        other=0.0,
    )
    tl.store(
        kv_out_ptr
        + bid * kv_out_stride_b
        + offs_t[:, None] * kv_out_stride_t
        + offs_d[None, :] * kv_out_stride_d,
        values,
        mask=(offs_t[:, None] < GATHER_LEN) & (offs_d[None, :] < KV_DIM),
    )
    tl.store(
        valid_out_ptr + bid * valid_out_stride_b + offs_t * valid_out_stride_t,
        physical_valid,
        mask=(did == 0) & (offs_t < GATHER_LEN),
    )


def gather_page64_kv_latent(
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    window_size: int,
    s_q: int,
    kv_cache_dim: int,
):
    bs = cache_seqlens.shape[0]
    assert block_table.shape[0] == bs

    gather_len = ((window_size + s_q - 1 + 7) // 8) * 8
    kv_latent = torch.empty(
        (bs, gather_len, kv_cache_dim),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    kv_valid = torch.empty((bs, gather_len), dtype=torch.bool, device=k_cache.device)

    block_t = 8
    block_d = 128
    _gather_page64_kv_latent_kernel[
        (bs, triton.cdiv(gather_len, block_t), triton.cdiv(kv_cache_dim, block_d))
    ](
        k_cache,
        block_table,
        cache_seqlens,
        kv_latent,
        kv_valid,
        k_cache.stride(0),
        k_cache.stride(2),
        block_table.stride(0),
        block_table.shape[1],
        k_cache.shape[0],
        kv_latent.stride(0),
        kv_latent.stride(1),
        kv_latent.stride(2),
        kv_valid.stride(0),
        kv_valid.stride(1),
        GATHER_LEN=gather_len,
        KV_DIM=kv_cache_dim,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        PAGE_SIZE_=PAGE_SIZE,
        num_warps=4,
    )
    return kv_latent, kv_valid


@triton.jit
def _apply_swa_score_mask_kernel(
    scores_ptr,
    cache_seqlens_ptr,
    valid_ptr,
    scores_stride_b,
    scores_stride_h,
    scores_stride_q,
    scores_stride_t,
    valid_stride_b,
    valid_stride_t,
    GATHER_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    S_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    qid = tl.program_id(2)

    offs_t = tl.arange(0, BLOCK_T)
    cache_seqlen = tl.load(cache_seqlens_ptr + bid).to(tl.int32)
    gather_start = tl.maximum(cache_seqlen - GATHER_LEN, 0)
    kv_pos = gather_start + offs_t
    q_pos = cache_seqlen - S_Q + qid
    page_valid = tl.load(
        valid_ptr + bid * valid_stride_b + offs_t * valid_stride_t,
        mask=offs_t < GATHER_LEN,
        other=0,
    ).to(tl.int1)
    valid = (
        (offs_t < GATHER_LEN)
        & page_valid
        & (kv_pos <= q_pos)
        & (kv_pos >= q_pos - WINDOW_SIZE + 1)
        & (q_pos >= 0)
    )
    tl.store(
        scores_ptr
        + bid * scores_stride_b
        + hid * scores_stride_h
        + qid * scores_stride_q
        + offs_t * scores_stride_t,
        tl.full((BLOCK_T,), -3.4028234663852886e38, tl.float32),
        mask=(offs_t < GATHER_LEN) & ~valid,
    )


def apply_swa_score_mask(
    scores: torch.Tensor,
    cache_seqlens: torch.Tensor,
    kv_valid: torch.Tensor,
    num_heads: int,
    window_size: int,
    s_q: int,
):
    bs = cache_seqlens.shape[0]
    assert scores.shape[0] == bs
    assert scores.shape[1] == num_heads
    assert scores.shape[2] == s_q
    assert kv_valid.shape == (bs, scores.shape[3])
    gather_len = scores.shape[3]
    mask_block_t = triton.next_power_of_2(gather_len)
    _apply_swa_score_mask_kernel[(bs, num_heads, s_q)](
        scores,
        cache_seqlens,
        kv_valid,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        kv_valid.stride(0),
        kv_valid.stride(1),
        GATHER_LEN=gather_len,
        WINDOW_SIZE=window_size,
        S_Q=s_q,
        BLOCK_T=mask_block_t,
        num_warps=4,
    )
    return scores
