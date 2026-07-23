"""Fused CUDA-graph metadata update for the TRTLLM MHA backend.

`TRTLLMHAAttnBackend._apply_cuda_graph_metadata` used to rebuild the
page table(s) and seqlen buffers with ~25 small aten ops per graph
replay (index gathers, floor_divide, cumsum, dtype casts, copies).
On some CPUs that is ~0.7-1.0 ms of pure host dispatch, repeated 4x
per decode step (2 draft-decode steps + target-verify + draft-extend)
on every TP rank. The resulting per-rank CPU jitter skews the
cudaGraphLaunch across ranks and is paid as spin time inside the first
custom all-reduce of every replayed graph.

This kernel performs the whole update in ONE launch:
  - cache_seqlens[i]  = seq_lens[i] + seqlen_offset            (int32)
  - cu_seqlens_k[1:]  = cumsum(cache_seqlens)                  (int32)
  - cu_seqlens_q[1:]  = cumsum(qlens) or arange*q_stride       (optional)
  - page_table[i, p]  = req_to_token[req_pool_indices[i],
                                     p * page_size] // page_size
  - swa_page_table    = full_to_swa_mapping[token] // page_size (optional)
  - swa_out_cache_loc = full_to_swa_mapping[out_cache_loc], zero padded
                                                               (optional)
"""

import triton
import triton.language as tl

# cu_seqlens_q handling inside the fused kernel
Q_MODE_NONE = 0  # cu_seqlens_q is preset (decode / target-verify)
Q_MODE_CUMSUM = 1  # cu_seqlens_q[1:] = cumsum(qlens)   (draft-extend)
Q_MODE_STRIDED = 2  # cu_seqlens_q[1:] = arange*q_stride (draft-extend v2)


@triton.jit
def update_trtllm_mha_graph_metadata_kernel(
    # inputs
    req_pool_indices_ptr,  # [bs] int
    seq_lens_ptr,  # [bs] int
    req_to_token_ptr,  # [pool_size, req_to_token_stride] int32
    swa_mapping_ptr,  # [full_size + page_size + 1] int64, or None
    out_cache_loc_ptr,  # [num_out_tokens] int64, or None
    qlens_ptr,  # [bs] int, or None (Q_MODE_CUMSUM only)
    # outputs
    cache_seqlens_ptr,  # [bs] int32
    cu_seqlens_k_ptr,  # [bs + 1] int32
    cu_seqlens_q_ptr,  # [bs + 1] int32, or None
    page_table_ptr,  # [bs, page_table_stride] int32
    swa_page_table_ptr,  # [bs, swa_page_table_stride] int32, or None
    swa_out_cache_loc_ptr,  # [swa_out_len] int64, or None
    # scalars
    bs,
    seqlen_offset,  # added to seq_lens for cache_seqlens / cu_seqlens_k
    max_seq_pages,  # page-table columns to (re)write per row
    q_stride,  # Q_MODE_STRIDED stride
    num_out_tokens,  # valid prefix of out_cache_loc
    swa_out_len,  # full swa_out_cache_loc length (zero-padded tail)
    req_to_token_stride,
    page_table_stride,
    swa_page_table_stride,
    # constexpr
    PAGE_SIZE: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_SWA_OUT: tl.constexpr,
    Q_MODE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    BS_BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    if pid < bs:
        # One program per batch row: cache_seqlens + page table row(s).
        req_pool_index = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
        seqlen = (tl.load(seq_lens_ptr + pid) + seqlen_offset).to(tl.int32)
        tl.store(cache_seqlens_ptr + pid, seqlen)

        row_in = req_to_token_ptr + req_pool_index * req_to_token_stride
        row_out = page_table_ptr + pid.to(tl.int64) * page_table_stride
        if HAS_SWA:
            swa_row_out = swa_page_table_ptr + pid.to(tl.int64) * swa_page_table_stride
        for i in range(tl.cdiv(max_seq_pages, PAGE_BLOCK)):
            page_idx = i * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
            mask = page_idx < max_seq_pages
            token = tl.load(
                row_in + page_idx.to(tl.int64) * PAGE_SIZE, mask=mask, other=0
            )
            tl.store(row_out + page_idx, token // PAGE_SIZE, mask=mask)
            if HAS_SWA:
                token64 = token.to(tl.int64)
                # Real req_to_token slots are >=0; the token>=0 guard + other=-1 mirror
                # the swa_out_cache_loc -1 sentinel (uniform handling, no wrap).
                swa_token = tl.load(
                    swa_mapping_ptr + token64, mask=mask & (token64 >= 0), other=-1
                )
                swa_page = tl.where(swa_token < 0, -1, swa_token // PAGE_SIZE)
                tl.store(swa_row_out + page_idx, swa_page.to(tl.int32), mask=mask)
    elif pid == bs:
        # Single program: cu_seqlens_k (+ optional cu_seqlens_q) cumsum.
        offs = tl.arange(0, BS_BLOCK)
        mask = offs < bs
        seqlens = (tl.load(seq_lens_ptr + offs, mask=mask, other=0)).to(tl.int32)
        seqlens = tl.where(mask, seqlens + seqlen_offset, 0)
        tl.store(cu_seqlens_k_ptr + 1 + offs, tl.cumsum(seqlens, axis=0), mask=mask)
        if Q_MODE == 1:  # Q_MODE_CUMSUM
            qlens = tl.load(qlens_ptr + offs, mask=mask, other=0).to(tl.int32)
            qlens = tl.where(mask, qlens, 0)
            tl.store(cu_seqlens_q_ptr + 1 + offs, tl.cumsum(qlens, axis=0), mask=mask)
        if Q_MODE == 2:  # Q_MODE_STRIDED
            tl.store(
                cu_seqlens_q_ptr + 1 + offs,
                ((offs + 1) * q_stride).to(tl.int32),
                mask=mask,
            )
    else:
        # Remaining programs: swa_out_cache_loc translate + zero padding.
        if HAS_SWA_OUT:
            out_idx = (pid - bs - 1) * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
            in_range = out_idx < swa_out_len
            is_real = in_range & (out_idx < num_out_tokens)
            loc = tl.load(out_cache_loc_ptr + out_idx, mask=is_real, other=0)
            translated = tl.load(
                swa_mapping_ptr + loc, mask=is_real & (loc >= 0), other=0
            )
            translated = tl.where(is_real & (loc < 0), -1, translated)
            tl.store(swa_out_cache_loc_ptr + out_idx, translated, mask=in_range)


def update_trtllm_mha_graph_metadata(
    *,
    req_pool_indices,
    seq_lens,
    req_to_token,
    cache_seqlens,
    cu_seqlens_k,
    page_table,
    bs: int,
    seqlen_offset: int,
    max_seq_pages: int,
    page_size: int,
    swa_mapping=None,
    swa_page_table=None,
    out_cache_loc=None,
    swa_out_cache_loc=None,
    cu_seqlens_q=None,
    qlens=None,
    q_stride: int = 0,
    q_mode: int = Q_MODE_NONE,
):
    """Launch the fused metadata update (one kernel for the whole replay init)."""
    if bs == 0:
        return

    # Launch-block width: page-table columns each program writes per iteration
    # (also the swa_out_cache_loc tile width). 512 keeps the per-program working
    # set small enough to stay off the register-pressure / occupancy cliff while
    # being wide enough to cover the static page-table width in few iterations.
    PAGE_BLOCK = 512
    has_swa = swa_page_table is not None
    has_swa_out = swa_out_cache_loc is not None

    swa_out_len = swa_out_cache_loc.shape[0] if has_swa_out else 0
    if has_swa_out and out_cache_loc is not None:
        num_out_tokens = min(swa_out_len, out_cache_loc.shape[0])
    else:
        num_out_tokens = 0
    if num_out_tokens == 0:
        # All loads are masked out; pass a valid dummy pointer for codegen.
        out_cache_loc = swa_out_cache_loc

    grid_extra = triton.cdiv(swa_out_len, PAGE_BLOCK) if has_swa_out else 0
    grid = (bs + 1 + grid_extra,)

    update_trtllm_mha_graph_metadata_kernel[grid](
        req_pool_indices,
        seq_lens,
        req_to_token,
        swa_mapping,
        out_cache_loc,
        qlens,
        cache_seqlens,
        cu_seqlens_k,
        cu_seqlens_q,
        page_table,
        swa_page_table,
        swa_out_cache_loc,
        bs,
        seqlen_offset,
        max_seq_pages,
        q_stride,
        num_out_tokens,
        swa_out_len,
        req_to_token.stride(0),
        page_table.stride(0),
        swa_page_table.stride(0) if has_swa else 0,
        PAGE_SIZE=page_size,
        HAS_SWA=has_swa,
        HAS_SWA_OUT=has_swa_out,
        Q_MODE=q_mode,
        PAGE_BLOCK=PAGE_BLOCK,
        BS_BLOCK=triton.next_power_of_2(bs),
    )
