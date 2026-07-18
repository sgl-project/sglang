from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


@triton.jit
def _prepare_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    seq_start_a_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    idx_b = pid_m
    # seq_len_a / seq_start_a are per dst row: the copied slice of src_a is
    # [seq_start_a, seq_start_a + seq_len_a).
    seq_len_a = tl.load(seq_len_a_ptr + pid_m)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)
    seq_start_a = tl.load(seq_start_a_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + (seq_start_a + offs_n) * a_stride_n
        a_mask = mask & (seq_start_a + offs_n < LEN_A)
        val = tl.load(a_ptr, mask=a_mask, other=0)
        tl.store(dst, val, mask=mask)
    elif pid_n * BLOCK_N >= seq_len_a:
        offs_b = offs_n - seq_len_a
        b_ptr = src_b_ptr + idx_b * b_stride_m + offs_b * b_stride_n
        b_mask = mask & (offs_b < LEN_B)
        val = tl.load(b_ptr, mask=b_mask, other=0)
        tl.store(dst, val, mask=mask)
    else:
        # mixed part
        a_offs = offs_n
        a_mask = (a_offs < seq_len_a) & (seq_start_a + a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + (seq_start_a + a_offs) * a_stride_n
        a_val = tl.load(a_ptr, mask=a_mask, other=0)

        b_offs = offs_n - seq_len_a
        b_mask = (b_offs >= 0) & (b_offs < seq_len_b) & (b_offs < LEN_B)
        b_ptr = src_b_ptr + idx_b * b_stride_m + b_offs * b_stride_n
        b_val = tl.load(b_ptr, mask=b_mask, other=0)

        result = tl.where(offs_n < seq_len_a, a_val, b_val)
        tl.store(dst, result, mask=mask)


def prepare_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,  # expand page table
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,  # expand seq lens
    speculative_num_draft_tokens: int,
    seq_start_a: Optional[torch.Tensor] = None,
):
    # Concat a per-row slice of page_table_a with the expand page_table:
    # dst row m = src_a[m // ndt][seq_start_a[m] : seq_start_a[m] + seq_len_a[m]]
    #             ++ src_b[m][: seq_len_b[m]].
    # seq_len_a may be per request (expanded here) or already per dst row;
    # seq_start_a defaults to zeros (plain prefix concat).
    bs_expand = seq_len_b.numel()
    if seq_len_a.numel() != bs_expand:
        assert seq_len_a.numel() * speculative_num_draft_tokens == bs_expand
        seq_len_a = seq_len_a.repeat_interleave(speculative_num_draft_tokens)
    if seq_start_a is None:
        seq_start_a = torch.zeros_like(seq_len_a)

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    # Row writes are bounded by seq_len_a + seq_len_b, which callers guarantee
    # fits the dst width, so the grid only needs to cover dst columns.
    LEN_OUT = page_table_dst.shape[1]
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        seq_start_a,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


@triton.jit
def _fused_metadata_kernel_general(
    # Input tensors
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # Output buffers
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    swa_page_table,
    swa_page_table_stride_0,
    swa_page_table_stride_1,
    full_to_swa_mapping,
    full_to_swa_mapping_stride_0,
    # Scalar parameters
    B,
    max_seq_pages,
    page_size: tl.constexpr,
    seq_len_delta: tl.constexpr,
    use_swa: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # column chunk index

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # Load row index for this batch (all threads in block have same i)
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # Compute column indices in the source tensor (token offset)
    if page_size == 1:
        col_idx = col_offsets
    else:
        col_idx = col_offsets << SHIFT  # faster than multiplication for power-of-two

    # Load page indices from req_to_token
    rt_offsets = row_offset + col_idx * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # Compute page_table
    if page_size == 1:
        page_table_val = page_index
    else:
        page_table_val = page_index >> SHIFT

    # Store to page_table
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_table_val, mask=mask, cache_modifier=".cg")

    if use_swa:
        swa_slot = tl.load(
            full_to_swa_mapping + page_index * full_to_swa_mapping_stride_0,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        )
        if page_size == 1:
            swa_val = swa_slot
        else:
            swa_val = swa_slot >> SHIFT
        swa_offsets = (
            i * swa_page_table_stride_0 + col_offsets * swa_page_table_stride_1
        )
        tl.store(swa_page_table + swa_offsets, swa_val, mask=mask, cache_modifier=".cg")


@triton.jit
def _fused_metadata_kernel_ps1_no_swa(
    # Input tensors
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # Output buffers
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    # Scalar parameters
    B,
    max_seq_pages,
    seq_len_delta: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # column chunk index

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # Load row index for this batch (all threads in block have same i)
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # page_size = 1: col_idx = col_offsets
    rt_offsets = row_offset + col_offsets * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # page_table = page_index // 1 = page_index
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_index, mask=mask, cache_modifier=".cg")


def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    token_to_kv_pool: Optional["SWAKVPool"] = None,
):
    """
    Fused Triton implementation that replaces 4-5 sequential CUDA kernels with 1-2 kernels:
      1. cache_seqlens = seq_lens + seq_len_delta (int64->int32 cast)
      2. cu_seqlens_k = cumsum(cache_seqlens) (prefix-sum)
      3. page_indices = req_to_token[pool_idx, stride_idx] (2-D gather)
      4. page_table = page_indices // page_size (floor-divide)
      5. (optional) swa_page_table for sliding window attention

    Achieves ~5.2x speedup on H200 hardware for typical decode workloads.
    """
    assert (
        page_size > 0 and (page_size & (page_size - 1)) == 0
    ), f"page_size must be a power of two, got {page_size}"

    batch_size = cache_seqlens_int32.shape[0]
    device = seq_lens.device

    # Ensure contiguous memory layout for efficient Triton access
    seq_lens = seq_lens.contiguous()
    req_to_token = req_to_token.contiguous()
    req_pool_indices = req_pool_indices.contiguous()

    # Prepare tensor strides
    seq_lens_stride_0 = seq_lens.stride(0)
    req_to_token_stride_0 = req_to_token.stride(0)
    req_to_token_stride_1 = req_to_token.stride(1)
    req_pool_indices_stride_0 = req_pool_indices.stride(0)
    cache_seqlens_int32_stride_0 = cache_seqlens_int32.stride(0)
    cu_seqlens_k_stride_0 = cu_seqlens_k.stride(0)
    page_table_stride_0 = page_table.stride(0)
    page_table_stride_1 = page_table.stride(1)

    # Check if we should use the specialized fast path for page_size=1, no SWA
    use_swa = swa_page_table is not None and token_to_kv_pool is not None

    if page_size == 1 and not use_swa:
        # Specialized kernel for the common case (page_size=1, no SWA)
        BLOCK_COLS = 256
        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_ps1_no_swa[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            batch_size,
            max_seq_pages,
            seq_len_delta,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=8,
            num_stages=3,
        )
    else:
        # General kernel for page_size > 1 or SWA cases
        # SWA parameters
        if use_swa:
            from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

            assert isinstance(token_to_kv_pool, SWAKVPool)
            swa_page_table = swa_page_table.contiguous()
            swa_page_table_stride_0 = swa_page_table.stride(0)
            swa_page_table_stride_1 = swa_page_table.stride(1)
            # Extract the full_to_swa_index_mapping from token_to_kv_pool
            full_to_swa_mapping = (
                token_to_kv_pool.full_to_swa_index_mapping.contiguous()
            )
            full_to_swa_mapping_stride_0 = full_to_swa_mapping.stride(0)
        else:
            # Dummy tensors (not used)
            swa_page_table = torch.empty(0, dtype=torch.int32, device=device)
            swa_page_table_stride_0 = 0
            swa_page_table_stride_1 = 0
            full_to_swa_mapping = torch.empty(0, dtype=torch.int32, device=device)
            full_to_swa_mapping_stride_0 = 0

        # Kernel configuration
        BLOCK_COLS = 128
        shift = (page_size).bit_length() - 1 if page_size > 1 else 0

        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_general[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            swa_page_table,
            swa_page_table_stride_0,
            swa_page_table_stride_1,
            full_to_swa_mapping,
            full_to_swa_mapping_stride_0,
            batch_size,
            max_seq_pages,
            page_size,
            seq_len_delta,
            use_swa,
            shift,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=4,
            num_stages=3,
        )
