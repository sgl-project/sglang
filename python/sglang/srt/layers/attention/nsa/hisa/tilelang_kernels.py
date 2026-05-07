"""HISA tilelang kernels — production-active set.

These three interface functions plus their prim-func bodies are the only
tilelang kernels still on the production hot path:

  - ``fp8_native_block_mean_pooling_interface`` (orchestrator stage 1, K>=64)
  - ``fp8_native_block_mean_pooling_grouped_interface`` (orchestrator stage 1, K<64)
  - ``fp8_native_paged_mean_pooling_completed_blocks_interface`` (pool-K cache writeback)

All other tilelang code (legacy orchestrators, ablation variants, torch-ref
impls) lives in ``tilelang_legacy.py`` and is slated for removal once the
SGLANG_HISA_DISABLE_TRITON fallback is dropped.
"""

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling(
    max_num_pooling_blocks: int,
    pooling_block_size: int,
    dim: int,
    block_N: int = 64,
    num_stages=1,
    threads=256,
):
    """Mean-pool with fp8 re-quantization: outputs fp8 BlockedK + f32 BlockedKScale.

    Branches each block_N inner iter on tile fullness:
      - Full tile (cur_tl_block_size == block_N): bulk TMA load (fast path).
      - Partial tile: per-row T.Parallel + ``if bn_i < cur_size`` guard.

    Triggered only on the last CTA's last inner iter when ``seq_len_k % block_N
    != 0``. An unguarded bulk T.copy in that case over-reads up to block_N-1
    rows past seq_len_k → under sustained load eventually hits an unmapped
    page (Xid 13 / illegal memory access). See the grouped variant's
    docstring for the original incident.
    """
    dtype = T.float8_e4m3fn
    accum_dtype = T.float32

    seq_len_k = T.dynamic("seq_len_k")
    k_size = [seq_len_k, dim]
    scale_size = [seq_len_k]
    blocked_k_size = [max_num_pooling_blocks, dim]
    blocked_k_scale_size = [max_num_pooling_blocks]
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def fp8_native_block_mean_pooling_kernel(
        K: T.Tensor(k_size, dtype=dtype),  # type: ignore
        KScale: T.Tensor(scale_size, dtype=accum_dtype),  # type: ignore
        BlockedK: T.Tensor(blocked_k_size, dtype=dtype),  # type: ignore
        BlockedKScale: T.Tensor(blocked_k_scale_size, dtype=accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len_k, pooling_block_size), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)
            T.fill(acc, 0.0)

            k_start = bx * pooling_block_size
            k_end = T.min(k_start + pooling_block_size, seq_len_k)
            cur_pooling_block_size = k_end - k_start

            for b_i in T.serial(T.ceildiv(cur_pooling_block_size, block_N)):
                T.fill(index_k, 0.0)
                T.fill(scale, 0.0)

                tl_block_s = k_start + b_i * block_N
                tl_block_e = T.min(k_start + (b_i + 1) * block_N, k_end)
                cur_tl_block_size = tl_block_e - tl_block_s

                if cur_tl_block_size == block_N:
                    # Fast path: full tile, bulk TMA load.
                    T.copy(K[tl_block_s : tl_block_s + block_N, :], index_k)
                    for bn_i in T.Parallel(block_N):
                        scale[bn_i] = KScale[tl_block_s + bn_i]
                else:
                    # Partial tile (last CTA's tail): row-guard, no bulk T.copy.
                    for bn_i in T.Parallel(block_N):
                        if bn_i < cur_tl_block_size:
                            scale[bn_i] = KScale[tl_block_s + bn_i]
                    for bn_i, d_i in T.Parallel(block_N, dim):
                        if bn_i < cur_tl_block_size:
                            index_k[bn_i, d_i] = K[tl_block_s + bn_i, d_i]

                for bn_i, d_i in T.Parallel(block_N, dim):
                    index_k[bn_i, d_i] = index_k[bn_i, d_i] * scale[bn_i]

                T.reduce_sum(index_k, acc, dim=0, clear=False)

            inv_count = T.cast(1.0, accum_dtype) / T.cast(
                cur_pooling_block_size, accum_dtype
            )
            for d_i in T.Parallel(dim):
                acc[d_i] = acc[d_i] * inv_count

            # Re-quantize the f32 mean to fp8 with a per-block scale.
            T.reduce_absmax(acc, max_abs, dim=0, clear=True)
            block_scale = T.max(
                max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                T.cast(1e-10, accum_dtype),
            )
            inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

            for d_i in T.Parallel(dim):
                BlockedK[bx, d_i] = T.cast(acc[d_i] * inv_block_scale, dtype)
            BlockedKScale[bx] = block_scale

    return fp8_native_block_mean_pooling_kernel


def fp8_native_block_mean_pooling_interface(k, k_scale, k_block_size):
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty(
        (max_num_pooling_blocks, d), device=k.device, dtype=torch.float8_e4m3fn
    )
    blocked_k_scale = torch.empty(
        (max_num_pooling_blocks,), device=k.device, dtype=torch.float32
    )
    kernel = fp8_native_block_mean_pooling(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
    )
    kernel(
        k,
        k_scale,
        blocked_k,
        blocked_k_scale,
    )
    return blocked_k, blocked_k_scale


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_block_mean_pooling_grouped(
    max_num_pooling_blocks: int,
    pooling_block_size: int,  # K, must divide block_N
    dim: int,
    block_N: int = 64,
    threads: int = 256,
):
    """Grouped variant of ``fp8_native_block_mean_pooling`` for K < block_N.

    The non-grouped kernel does ``T.copy(K[s:s+block_N=64], ...)`` per pool
    block, which over-reads ``block_N - K`` rows past each pool block end —
    fine inside the K tensor body, but at the last few pool blocks it walks
    off the end of ``seq_len_k`` and (under sustained load) eventually hits
    an unmapped page → Xid 13 → silent crash.

    This grouped variant flips the parallelism: one CTA per ``block_N``
    tokens, producing ``G = block_N // K`` pool blocks. The ``T.copy``
    reads exactly ``block_N`` rows that all live within seq_len_k except
    possibly at the very last CTA, which we handle with a row-guard
    (``T.Parallel`` + Python ``if``) instead of bulk T.copy.

    Constraint: ``block_N % pooling_block_size == 0``. Use the non-grouped
    variant when ``K >= block_N``.
    """
    assert block_N % pooling_block_size == 0, (
        f"block_N ({block_N}) must be divisible by "
        f"pooling_block_size ({pooling_block_size})"
    )
    G = block_N // pooling_block_size

    dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    seq_len_k = T.dynamic("seq_len_k")
    FP8_MAX_INV = 1.0 / 448.0

    @T.prim_func
    def fp8_native_block_mean_pooling_grouped_kernel(
        K: T.Tensor([seq_len_k, dim], dtype=dtype),  # type: ignore
        KScale: T.Tensor([seq_len_k], dtype=accum_dtype),  # type: ignore
        BlockedK: T.Tensor([max_num_pooling_blocks, dim], dtype=dtype),  # type: ignore
        BlockedKScale: T.Tensor([max_num_pooling_blocks], dtype=accum_dtype),  # type: ignore
    ):
        # Grid: one CTA per block_N tokens (= G pool blocks).
        with T.Kernel(T.ceildiv(seq_len_k, block_N), threads=threads) as bx:
            index_k = T.alloc_fragment([block_N, dim], accum_dtype)
            scale = T.alloc_fragment([block_N], accum_dtype)
            acc_per_pool = T.alloc_fragment([G, dim], accum_dtype)
            max_abs_per_pool = T.alloc_fragment([G], accum_dtype)

            tl_block_s = bx * block_N
            cur_block_size = T.min(tl_block_s + block_N, seq_len_k) - tl_block_s

            # Bounds-safe row-by-row load. Avoids the OOB pattern of the
            # non-grouped kernel's bulk ``T.copy``. Slightly slower per-CTA
            # (no TMA), but only one CTA per block_N tokens vs one per pool
            # block, so total launches drop by G.
            T.fill(scale, 0.0)
            T.fill(index_k, 0.0)
            for bn_i in T.Parallel(block_N):
                if bn_i < cur_block_size:
                    scale[bn_i] = KScale[tl_block_s + bn_i]
            for bn_i, d_i in T.Parallel(block_N, dim):
                if bn_i < cur_block_size:
                    index_k[bn_i, d_i] = (
                        T.cast(K[tl_block_s + bn_i, d_i], accum_dtype) * scale[bn_i]
                    )

            # Per-pool sum: build a [G, K, dim] view of index_k and reduce
            # along axis 1. The intermediate fragment costs an extra register
            # tile but lets tilelang's IR pattern-match a clean reduction
            # (avoiding the non-unit-stride / "k_i used before def" pitfalls
            # of a manual nested-loop accumulator).
            gk_view = T.alloc_fragment([G, pooling_block_size, dim], accum_dtype)
            for g_i, k_inner, d_i in T.Parallel(G, pooling_block_size, dim):
                gk_view[g_i, k_inner, d_i] = index_k[
                    g_i * pooling_block_size + k_inner, d_i
                ]
            T.reduce_sum(gk_view, acc_per_pool, dim=1, clear=True)

            # Per-pool mean: divide by actual valid token count (handles the
            # partial trailing pool block when seq_len_k isn't a multiple of
            # K). ``T.if_then_else`` guards the divide-by-zero case for pool
            # blocks that fall entirely past seq_len_k (they will be masked
            # out at store time anyway).
            for g_i, d_i in T.Parallel(G, dim):
                pool_start = tl_block_s + g_i * pooling_block_size
                pool_end_c = T.min(pool_start + pooling_block_size, seq_len_k)
                pc = pool_end_c - pool_start
                inv_count = T.if_then_else(
                    pc > 0,
                    T.cast(1.0, accum_dtype) / T.cast(pc, accum_dtype),
                    T.cast(0.0, accum_dtype),
                )
                acc_per_pool[g_i, d_i] = acc_per_pool[g_i, d_i] * inv_count

            # Per-pool fp8 max-abs scale.
            T.reduce_absmax(acc_per_pool, max_abs_per_pool, dim=1, clear=True)

            # Quantize + masked store: skip pool blocks past num_pool_total
            # (last CTA's trailing slots).
            for g_i, d_i in T.Parallel(G, dim):
                out_idx = bx * G + g_i
                if out_idx < max_num_pooling_blocks:
                    bs = T.max(
                        max_abs_per_pool[g_i] * T.cast(FP8_MAX_INV, accum_dtype),
                        T.cast(1e-10, accum_dtype),
                    )
                    BlockedK[out_idx, d_i] = T.cast(
                        acc_per_pool[g_i, d_i] / bs,
                        dtype,
                    )
            for g_i in T.Parallel(G):
                out_idx = bx * G + g_i
                if out_idx < max_num_pooling_blocks:
                    bs = T.max(
                        max_abs_per_pool[g_i] * T.cast(FP8_MAX_INV, accum_dtype),
                        T.cast(1e-10, accum_dtype),
                    )
                    BlockedKScale[out_idx] = bs

    return fp8_native_block_mean_pooling_grouped_kernel


def fp8_native_block_mean_pooling_grouped_interface(
    k, k_scale, k_block_size, block_N=64
):
    """Tilelang grouped mean-pool for K < block_N. Same I/O contract as
    ``fp8_native_block_mean_pooling_interface``."""
    seq_len_k, d = k.shape
    max_num_pooling_blocks = (seq_len_k + k_block_size - 1) // k_block_size

    blocked_k = torch.empty(
        (max_num_pooling_blocks, d), device=k.device, dtype=torch.float8_e4m3fn
    )
    blocked_k_scale = torch.empty(
        (max_num_pooling_blocks,), device=k.device, dtype=torch.float32
    )
    kernel = fp8_native_block_mean_pooling_grouped(
        max_num_pooling_blocks=max_num_pooling_blocks,
        pooling_block_size=k_block_size,
        dim=d,
        block_N=block_N,
    )
    kernel(k, k_scale, blocked_k, blocked_k_scale)
    return blocked_k, blocked_k_scale


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def fp8_native_paged_mean_pooling_completed_blocks(
    paged_block_size: int,
    pooling_block_size: int,
    pool_page_size: int,
    dim: int,
    max_pool_per_req_grid: int,
    num_stages=1,
    threads=128,
):
    """Paged-output variant of completed_blocks — writes into ``pool_k_pages``.

    Grid: ``(batch, max_pool_per_req_grid)``. Cell ``(b, pblk_rel)`` is a
    potential new-completion at absolute pool-block index
    ``pblk_abs = prev_complete + pblk_rel``. If ``pblk_rel < n_new``, the
    kernel mean-pools the covering K=128 tokens and writes:
      * ``logical_pool_page = pblk_abs // pool_page_size``
      * ``slot = pblk_abs %  pool_page_size``
      * ``phys = pool_page_tables[req_idx, logical_pool_page]``
      * ``pool_k_pages[phys, slot, :D]``  (fp8)  and scale slot.
    """
    fp8_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32
    req_pool_idx_dtype = T.int64

    num_kv_blocks = T.dynamic("num_kv_blocks")
    batch = T.dynamic("batch")
    max_ctx = T.dynamic("max_ctx")
    max_pool_pages = T.dynamic("max_pool_pages")
    num_pool_pages_global = T.dynamic("num_pool_pages_global")
    max_running_req = T.dynamic("max_running_req")

    kv_cache_fp8_shape = [num_kv_blocks, paged_block_size * (dim + 4)]
    kv_cache_fp32_shape = [num_kv_blocks, paged_block_size * (dim + 4) // 4]
    req_to_token_shape = [max_running_req, max_ctx]
    pool_page_tables_shape = [max_running_req, max_pool_pages]
    seq_len_shape = [batch]
    pool_k_pages_fp8_shape = [num_pool_pages_global, pool_page_size * (dim + 4)]
    pool_k_pages_fp32_shape = [num_pool_pages_global, pool_page_size * (dim + 4) // 4]

    fp8_end = paged_block_size * dim
    scale_offset = paged_block_size * dim // 4
    FP8_MAX_INV = 1.0 / 448.0
    K = pooling_block_size

    block_N = paged_block_size  # 64
    assert K % block_N == 0

    @T.prim_func
    def kernel(
        KvCacheFP8View: T.Tensor(kv_cache_fp8_shape, fp8_dtype),  # type: ignore
        KvCacheFP32View: T.Tensor(kv_cache_fp32_shape, accum_dtype),  # type: ignore
        ReqToToken: T.Tensor(req_to_token_shape, index_dtype),  # type: ignore
        PoolPageTables: T.Tensor(pool_page_tables_shape, index_dtype),  # type: ignore
        ReqPoolIndices: T.Tensor(seq_len_shape, req_pool_idx_dtype),  # type: ignore
        PrevSeqLens: T.Tensor(seq_len_shape, index_dtype),  # type: ignore
        NewSeqLens: T.Tensor(seq_len_shape, index_dtype),  # type: ignore
        PoolKPagesFP8View: T.Tensor(pool_k_pages_fp8_shape, fp8_dtype),  # type: ignore
        PoolKPagesFP32View: T.Tensor(pool_k_pages_fp32_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, max_pool_per_req_grid, threads=threads) as (bx, by):
            b = bx
            pblk_rel = by

            index_k_shared = T.alloc_fragment([block_N * dim], accum_dtype)
            index_k_reshaped = T.reshape(index_k_shared, [block_N, dim])
            scale_shared = T.alloc_fragment([block_N], accum_dtype)
            acc = T.alloc_fragment([dim], accum_dtype)
            max_abs = T.alloc_fragment([1], accum_dtype)

            prev_len = PrevSeqLens[b]
            new_len = NewSeqLens[b]
            prev_complete = prev_len // K
            new_complete = new_len // K
            n_new = new_complete - prev_complete

            if pblk_rel < n_new:
                pblk_abs = prev_complete + pblk_rel
                req_idx = T.cast(ReqPoolIndices[b], index_dtype)
                logical_page = pblk_abs // pool_page_size
                slot = pblk_abs - logical_page * pool_page_size
                phys = PoolPageTables[req_idx, logical_page]
                logical_start = pblk_abs * K

                T.fill(acc, 0.0)
                for b_i in T.serial(K // block_N):
                    chunk_logical_start = logical_start + b_i * block_N
                    T.fill(index_k_shared, 0.0)

                    buf_pos = ReqToToken[req_idx, chunk_logical_start]
                    phys_page = buf_pos // paged_block_size

                    T.copy(KvCacheFP8View[phys_page, :fp8_end], index_k_shared)
                    T.copy(KvCacheFP32View[phys_page, scale_offset:], scale_shared)

                    for n_i, d_i in T.Parallel(block_N, dim):
                        index_k_reshaped[n_i, d_i] = (
                            T.cast(index_k_reshaped[n_i, d_i], accum_dtype)
                            * scale_shared[n_i]
                        )

                    T.reduce_sum(index_k_reshaped, acc, dim=0, clear=False)

                inv_count = T.cast(1.0, accum_dtype) / T.cast(K, accum_dtype)
                for d_i in T.Parallel(dim):
                    acc[d_i] = acc[d_i] * inv_count

                T.reduce_absmax(acc, max_abs, dim=0, clear=True)
                block_scale = T.max(
                    max_abs[0] * T.cast(FP8_MAX_INV, accum_dtype),
                    T.cast(1e-10, accum_dtype),
                )
                inv_block_scale = T.cast(1.0, accum_dtype) / block_scale

                fp8_row_off = slot * dim
                scale_f32_idx = pool_page_size * dim // 4 + slot
                for d_i in T.Parallel(dim):
                    PoolKPagesFP8View[phys, fp8_row_off + d_i] = T.cast(
                        acc[d_i] * inv_block_scale, fp8_dtype
                    )
                PoolKPagesFP32View[phys, scale_f32_idx] = block_scale

    return kernel


def fp8_native_paged_mean_pooling_completed_blocks_interface(
    kv_cache_flat: torch.Tensor,  # [num_pages, page_size * (D + 4)] uint8
    req_to_token: torch.Tensor,  # [R, T] int32
    pool_page_tables: torch.Tensor,  # [R, max_pool_pages] int32
    req_pool_indices: torch.Tensor,  # [B] int64
    prev_seq_lens: torch.Tensor,  # [B] int32
    new_seq_lens: torch.Tensor,  # [B] int32
    pool_k_pages: torch.Tensor,  # [N_pool_pages, pool_page_size * (D+4)] uint8 IN-OUT
    k_block_size: int,
    paged_block_size: int,
    pool_page_size: int,
    max_pool_per_req_grid: int,
):
    _, DPlus4_times_P = kv_cache_flat.shape
    D = DPlus4_times_P // paged_block_size - 4
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4)

    kernel = fp8_native_paged_mean_pooling_completed_blocks(
        paged_block_size=paged_block_size,
        pooling_block_size=k_block_size,
        pool_page_size=pool_page_size,
        dim=D,
        max_pool_per_req_grid=max_pool_per_req_grid,
    )
    kernel(
        kv_cache_flat.view(torch.float8_e4m3fn),
        kv_cache_flat.view(torch.float32),
        req_to_token,
        pool_page_tables,
        req_pool_indices,
        prev_seq_lens,
        new_seq_lens,
        pool_k_pages.view(torch.float8_e4m3fn),
        pool_k_pages.view(torch.float32),
    )
