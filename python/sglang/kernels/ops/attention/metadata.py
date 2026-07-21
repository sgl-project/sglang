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
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + offs_n * a_stride_n
        a_mask = mask & (offs_n < LEN_A)
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
        a_mask = (a_offs < seq_len_a) & (a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
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
):
    # concat page_table and expand page_table by kv seq length
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    LEN_OUT = LEN_A + LEN_B
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
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


@triton.jit
def _draft_extend_metadata_kernel(
    # Input tensors
    seq_lens,
    seq_lens_stride_0,
    extend_seq_lens,
    extend_seq_lens_stride_0,
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
    cu_seqlens_q,
    cu_seqlens_q_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    full_to_swa_index_mapping,
    swa_page_table,
    out_cache_loc,
    swa_out_cache_loc,
    # Scalar parameters
    B,
    max_seq_pages,
    tokens_per_req,
    PAGE_SIZE_ONE: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    HAS_SWA: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # column chunk index

    # 1. Prefix sums (only one block does them): cache_seqlens + cu_seqlens_k
    #    from seq_lens, cu_seqlens_q from extend_seq_lens.
    if pid_b == 0 and pid_c == 0:
        acc_k = 0
        acc_q = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, seq)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc_k)
            acc_k += seq
            ext = tl.load(extend_seq_lens + idx * extend_seq_lens_stride_0).to(tl.int32)
            tl.store(cu_seqlens_q + idx * cu_seqlens_q_stride_0, acc_q)
            acc_q += ext
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc_k)
        tl.store(cu_seqlens_q + B * cu_seqlens_q_stride_0, acc_q)

    # 2. SWA write-loc translation for this request's extend tokens. Runs
    #    before the seq_len early-return so padded rows (seq_len 0) keep
    #    swa_out_cache_loc consistent with out_cache_loc.
    if HAS_SWA:
        if pid_c == 0:
            tok_idx = tl.arange(0, OUT_BLOCK)
            tok_mask = tok_idx < tokens_per_req
            tok_offsets = pid_b * tokens_per_req + tok_idx
            full_locs = tl.load(out_cache_loc + tok_offsets, mask=tok_mask, other=0)
            swa_locs = tl.load(
                full_to_swa_index_mapping + full_locs, mask=tok_mask, other=0
            )
            tl.store(swa_out_cache_loc + tok_offsets, swa_locs, mask=tok_mask)

    # 3. Page-table gather for this batch row and column chunk, self-guarded
    #    on the device-side seq_len (no host max; tails keep stale values the
    #    attention kernels never read past cache_seqlens).
    if max_seq_pages == 0:
        return

    seq_len = tl.load(seq_lens + pid_b * seq_lens_stride_0).to(tl.int32)
    if PAGE_SIZE_ONE:
        num_live_pages = seq_len
    else:
        num_live_pages = (seq_len + (1 << SHIFT) - 1) >> SHIFT
    num_live_pages = tl.minimum(num_live_pages, max_seq_pages)
    if pid_c * BLOCK_COLS >= num_live_pages:
        return

    row_idx = tl.load(req_pool_indices + pid_b * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_offsets = pid_c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < num_live_pages

    if PAGE_SIZE_ONE:
        col_idx = col_offsets
    else:
        col_idx = col_offsets << SHIFT

    rt_offsets = row_offset + col_idx * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    if PAGE_SIZE_ONE:
        page_table_val = page_index
    else:
        page_table_val = page_index >> SHIFT

    pt_offsets = pid_b * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_table_val, mask=mask, cache_modifier=".cg")

    if HAS_SWA:
        swa_loc = tl.load(full_to_swa_index_mapping + page_index, mask=mask, other=0)
        if PAGE_SIZE_ONE:
            swa_page_table_val = swa_loc
        else:
            swa_page_table_val = swa_loc >> SHIFT
        tl.store(
            swa_page_table + pt_offsets,
            swa_page_table_val.to(tl.int32),
            mask=mask,
            cache_modifier=".cg",
        )


def draft_extend_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    page_size: int,
    full_to_swa_index_mapping: Optional[torch.Tensor] = None,
    swa_page_table: Optional[torch.Tensor] = None,
    out_cache_loc: Optional[torch.Tensor] = None,
    swa_out_cache_loc: Optional[torch.Tensor] = None,
):
    """Fused, graph-recordable DRAFT_EXTEND_V2 metadata update (one launch):
      1. cache_seqlens = seq_lens (int32 cast)
      2. cu_seqlens_k = pad(cumsum(cache_seqlens))
      3. cu_seqlens_q = pad(cumsum(extend_seq_lens))
      4. page_table[:, :pages(seq_len)] = req_to_token[pool_idx, ::page_size] // page_size
      5. (SWA pools) swa_page_table likewise via the full->swa lookup, and
         swa_out_cache_loc = full_to_swa_index_mapping[out_cache_loc]

    The page gathers self-guard on the device-side seq_lens (no host max);
    row tails keep stale values that attention kernels never read past
    cache_seqlens, matching the eager replay path's bounded writes.
    """
    assert (
        page_size > 0 and (page_size & (page_size - 1)) == 0
    ), f"page_size must be a power of two, got {page_size}"

    batch_size = cache_seqlens_int32.shape[0]
    max_seq_pages = page_table.shape[1]

    has_swa = full_to_swa_index_mapping is not None
    if has_swa:
        assert swa_page_table is not None
        assert swa_page_table.shape == page_table.shape
        assert swa_page_table.stride() == page_table.stride()
        assert out_cache_loc is not None and swa_out_cache_loc is not None
        num_out_tokens = out_cache_loc.shape[0]
        assert swa_out_cache_loc.shape[0] == num_out_tokens
        assert num_out_tokens > 0 and num_out_tokens % batch_size == 0
        tokens_per_req = num_out_tokens // batch_size
        out_block = triton.next_power_of_2(tokens_per_req)
    else:
        tokens_per_req = 0
        out_block = 1

    BLOCK_COLS = 256
    grid = (batch_size, max(1, triton.cdiv(max_seq_pages, BLOCK_COLS)))

    _draft_extend_metadata_kernel[grid](
        seq_lens,
        seq_lens.stride(0),
        extend_seq_lens,
        extend_seq_lens.stride(0),
        req_to_token,
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_pool_indices,
        req_pool_indices.stride(0),
        cache_seqlens_int32,
        cache_seqlens_int32.stride(0),
        cu_seqlens_k,
        cu_seqlens_k.stride(0),
        cu_seqlens_q,
        cu_seqlens_q.stride(0),
        page_table,
        page_table.stride(0),
        page_table.stride(1),
        full_to_swa_index_mapping,
        swa_page_table,
        out_cache_loc,
        swa_out_cache_loc,
        batch_size,
        max_seq_pages,
        tokens_per_req,
        PAGE_SIZE_ONE=page_size == 1,
        SHIFT=(page_size).bit_length() - 1 if page_size > 1 else 0,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=8,
        num_stages=3,
        HAS_SWA=has_swa,
        OUT_BLOCK=out_block,
    )


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
