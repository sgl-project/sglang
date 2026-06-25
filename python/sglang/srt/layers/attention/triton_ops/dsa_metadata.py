import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dsa_decode_metadata_kernel(
    seq_lens,
    req_pool_indices,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table_1,
    dsa_cache_seqlens,
    dsa_cu_seqlens_k,
    real_page_table,
    seq_lens_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    req_to_token_stride_0: tl.constexpr,
    req_to_token_stride_1: tl.constexpr,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    real_page_table_stride_0: tl.constexpr,
    real_page_table_stride_1: tl.constexpr,
    bs: tl.constexpr,
    max_len: tl.constexpr,
    dsa_index_topk: tl.constexpr,
    real_page_size: tl.constexpr,
    HAS_REAL_PAGE_TABLE: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 0:
        offs_b = tl.arange(0, BLOCK_BS)
        mask_b = offs_b < bs
        seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
        seq_i32 = seq.to(tl.int32)
        dsa_seq = tl.minimum(seq_i32, dsa_index_topk)

        cu = tl.cumsum(seq_i32, 0)
        dsa_cu = tl.cumsum(dsa_seq, 0)

        tl.store(cache_seqlens + offs_b, seq_i32, mask=mask_b)
        tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)
        tl.store(dsa_cache_seqlens + offs_b, dsa_seq, mask=mask_b)
        tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
        tl.store(dsa_cu_seqlens_k + 1 + offs_b, dsa_cu, mask=mask_b)
        return

    num_col_blocks = tl.cdiv(max_len, BLOCK_N)
    page_pid = pid - 1
    row = page_pid // num_col_blocks
    col_block = page_pid - row * num_col_blocks
    offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row < bs) & (offs_n < max_len)

    req_idx = tl.load(
        req_pool_indices + row * req_pool_indices_stride,
        mask=row < bs,
        other=0,
    )
    vals = tl.load(
        req_to_token + req_idx * req_to_token_stride_0 + offs_n * req_to_token_stride_1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    tl.store(
        page_table_1 + row * page_table_stride_0 + offs_n * page_table_stride_1,
        vals,
        mask=mask,
    )

    if HAS_REAL_PAGE_TABLE:
        real_mask = mask & ((offs_n % real_page_size) == 0)
        real_cols = offs_n // real_page_size
        tl.store(
            real_page_table
            + row * real_page_table_stride_0
            + real_cols * real_page_table_stride_1,
            vals // real_page_size,
            mask=real_mask,
        )


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: torch.Tensor,
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    real_page_size: int,
) -> None:
    assert seq_lens.is_cuda
    assert req_pool_indices.is_cuda
    assert req_to_token.is_cuda
    assert cache_seqlens.is_cuda
    assert cu_seqlens_k.is_cuda
    assert page_table_1.is_cuda
    assert dsa_cache_seqlens.is_cuda
    assert dsa_cu_seqlens_k.is_cuda

    if bs == 0:
        return

    has_real_page_table = real_page_size > 1
    if has_real_page_table:
        assert real_page_table is not None
        assert real_page_table.is_cuda
    else:
        real_page_table = page_table_1

    block_bs = triton.next_power_of_2(bs)
    block_n = 128
    num_col_blocks = triton.cdiv(max_len, block_n)
    grid = (1 + bs * num_col_blocks,)

    _fused_dsa_decode_metadata_kernel[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        max_len,
        dsa_index_topk,
        real_page_size,
        has_real_page_table,
        BLOCK_BS=block_bs,
        BLOCK_N=block_n,
    )
