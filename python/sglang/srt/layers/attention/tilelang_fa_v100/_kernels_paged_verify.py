"""Grouped split-KV attention for linear speculative verification on SM70.

The ordinary paged-extend kernel assigns one CTA to each query head.  That is
fine for prefill tiles, but it is a poor fit for a 16-token DFlash/MTP verify
block at a long cached prefix: every CTA scans the entire prefix serially and
GQA heads reload identical K/V data.  This kernel instead assigns CTAs to
``(sequence, kv_head, GQA subgroup, context split)``.  Each CTA computes up to
four query heads for all 16 verify tokens, keeping separate SM70 MMA K/V
layouts within V100's 96 KiB shared-memory limit.  A small second kernel
combines the normalized split outputs.
"""

import math
import os

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs

VERIFY_Q_BLOCK = 16
# A TP4 Qwen3.8 verifier rank has one KV head and only two grouped-GQA CTAs.
# Keep the context slices small enough to expose useful SM-level parallelism
# before long contexts saturate the fixed 80-CTA launch.  The former 1024
# value left only eight CTAs active at a 4K prefix on an 80-SM V100.
VERIFY_MIN_TOKENS_PER_SPLIT = 128
VERIFY_SM_TARGET = 80  # V100 SXM2 SM count; one memory-streaming CTA per SM.
_LOG2_E = 1.4426950408889634


def _verify_min_tokens_per_split():
    value = os.environ.get("SGLANG_V100_VERIFY_TOKENS_PER_SPLIT")
    if value is None:
        return VERIFY_MIN_TOKENS_PER_SPLIT
    try:
        value = int(value)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_V100_VERIFY_TOKENS_PER_SPLIT must be a positive integer, "
            f"got {value!r}."
        ) from exc
    if value <= 0:
        raise ValueError(
            "SGLANG_V100_VERIFY_TOKENS_PER_SPLIT must be a positive integer, "
            f"got {value}."
        )
    return value


@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
def _paged_verify_partial_kernel(
    batch,
    heads,
    heads_kv,
    dim,
    page_block_size,
    max_blocks_per_seq,
    num_pages,
    is_causal,
    max_splits,
    block_N,
    threads,
    fp8_kv,
    min_tokens_per_split,
):
    nt = T.dynamic("nt")
    group_size = heads // heads_kv
    heads_per_cta = min(4, group_size)
    gqa_ctas = math.ceil(group_size / heads_per_cta)
    block_M = max(64, VERIFY_Q_BLOCK * heads_per_cta)

    q_shape = [nt, heads, dim]
    kv_shape = [num_pages, page_block_size, heads_kv, dim]
    partial_o_shape = [batch, max_splits, VERIFY_Q_BLOCK, heads, dim]
    partial_lse_shape = [batch, max_splits, VERIFY_Q_BLOCK, heads]
    kv_dtype = T.uint8 if fp8_kv else T.float16

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.float16),
        K_cache: T.Tensor(kv_shape, kv_dtype),
        V_cache: T.Tensor(kv_shape, kv_dtype),
        FP8_LUT: T.Tensor([256], T.float16),
        block_table: T.Tensor([batch, max_blocks_per_seq], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        query_start_loc: T.Tensor([batch + 1], T.int32),
        prefix_kv_lens: T.Tensor([batch], T.int32),
        sm_scale: T.float32,
        Partial_O: T.Tensor(partial_o_shape, T.float16),
        Partial_LSE: T.Tensor(partial_lse_shape, T.float32),
    ):
        with T.Kernel(heads_kv * gqa_ctas, max_splits, batch, threads=threads) as (
            head_group,
            split_id,
            batch_id,
        ):
            Q_shared = T.alloc_shared([block_M, dim], T.float16)
            K_shared = T.alloc_shared([block_N, dim], T.float16)
            V_shared = T.alloc_shared([block_N, dim], T.float16)
            P_shared = T.alloc_shared([block_M, block_N], T.float16)

            acc_s = T.alloc_fragment([block_M, block_N], T.float32)
            acc_s_cast = T.alloc_fragment([block_M, block_N], T.float16)
            acc_o = T.alloc_fragment([block_M, dim], T.float32)
            m_i = T.alloc_fragment([block_M], T.float32)
            m_prev = T.alloc_fragment([block_M], T.float32)
            l_i = T.alloc_fragment([block_M], T.float32)
            scale = T.alloc_fragment([block_M], T.float32)
            row_sum = T.alloc_fragment([block_M], T.float32)

            kv_head = T.floordiv(head_group, gqa_ctas)
            gqa_cta = head_group - kv_head * gqa_ctas
            q_head_offset = gqa_cta * heads_per_cta

            total_kv = cache_seqlens[batch_id]
            active_splits = T.min(
                max_splits,
                T.max(1, T.ceildiv(total_kv, min_tokens_per_split)),
            )
            split_len = T.ceildiv(total_kv, active_splits)
            split_start = split_id * split_len
            split_end = T.min(split_start + split_len, total_kv)
            q_start = query_start_loc[batch_id]
            q_len = query_start_loc[batch_id + 1] - q_start
            scale_log2 = sm_scale * _LOG2_E

            with T.If(split_id < active_splits), T.Then():
                T.clear(Q_shared)
                for row, d in T.Parallel(block_M, dim):
                    q_i = T.floordiv(row, heads_per_cta)
                    local_head = row - q_i * heads_per_cta
                    q_head_i = q_head_offset + local_head
                    if (q_i < q_len) & (q_head_i < group_size):
                        Q_shared[row, d] = Q[
                            q_start + q_i,
                            kv_head * group_size + q_head_i,
                            d,
                        ]

                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(T.float32))
                T.fill(l_i, 0)

                for tile_i in T.Pipelined(
                    T.ceildiv(split_end - split_start, block_N), num_stages=0
                ):
                    tile_start = split_start + tile_i * block_N
                    T.clear(K_shared)
                    for n, d in T.Parallel(block_N, dim):
                        kv_i = tile_start + n
                        logical_page = T.floordiv(kv_i, page_block_size)
                        page_offset = kv_i - logical_page * page_block_size
                        if kv_i < split_end:
                            physical_page = block_table[batch_id, logical_page]
                            if fp8_kv:
                                raw = K_cache[physical_page, page_offset, kv_head, d]
                                K_shared[n, d] = FP8_LUT[T.cast(raw, T.int32)]
                            else:
                                K_shared[n, d] = K_cache[
                                    physical_page, page_offset, kv_head, d
                                ]

                    for row, n in T.Parallel(block_M, block_N):
                        q_i = T.floordiv(row, heads_per_cta)
                        kv_i = tile_start + n
                        if is_causal:
                            acc_s[row, n] = T.if_then_else(
                                (q_i < q_len)
                                & (kv_i < split_end)
                                & (kv_i <= prefix_kv_lens[batch_id] + q_i),
                                0,
                                -T.infinity(T.float32),
                            )
                        else:
                            acc_s[row, n] = T.if_then_else(
                                (q_i < q_len) & (kv_i < split_end),
                                0,
                                -T.infinity(T.float32),
                            )

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for row in T.Parallel(block_M):
                        m_i[row] = T.if_then_else(
                            m_i[row] == -T.infinity(T.float32), 0, m_i[row]
                        )
                        m_i[row] = T.max(m_i[row], m_prev[row])
                        scale[row] = T.exp2((m_prev[row] - m_i[row]) * scale_log2)
                        l_i[row] *= scale[row]
                    for row, d in T.Parallel(block_M, dim):
                        acc_o[row, d] *= scale[row]
                    for row, n in T.Parallel(block_M, block_N):
                        acc_s[row, n] = T.exp2((acc_s[row, n] - m_i[row]) * scale_log2)
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for row in T.Parallel(block_M):
                        l_i[row] += row_sum[row]

                    T.clear(V_shared)
                    for n, d in T.Parallel(block_N, dim):
                        kv_i = tile_start + n
                        logical_page = T.floordiv(kv_i, page_block_size)
                        page_offset = kv_i - logical_page * page_block_size
                        if kv_i < split_end:
                            physical_page = block_table[batch_id, logical_page]
                            if fp8_kv:
                                raw = V_cache[physical_page, page_offset, kv_head, d]
                                V_shared[n, d] = FP8_LUT[T.cast(raw, T.int32)]
                            else:
                                V_shared[n, d] = V_cache[
                                    physical_page, page_offset, kv_head, d
                                ]

                    for row, n in T.Parallel(block_M, block_N):
                        P_shared[row, n] = T.cast(acc_s[row, n], T.float16)
                    T.copy(P_shared, acc_s_cast)
                    T.gemm(
                        acc_s_cast,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.Square,
                    )

                for row, d in T.Parallel(block_M, dim):
                    q_i = T.floordiv(row, heads_per_cta)
                    local_head = row - q_i * heads_per_cta
                    q_head_i = q_head_offset + local_head
                    if (q_i < q_len) & (q_head_i < group_size):
                        Partial_O[
                            batch_id,
                            split_id,
                            q_i,
                            kv_head * group_size + q_head_i,
                            d,
                        ] = T.cast(
                            acc_o[row, d] / T.if_then_else(l_i[row] == 0, 1, l_i[row]),
                            T.float16,
                        )
                for row in T.Parallel(block_M):
                    q_i = T.floordiv(row, heads_per_cta)
                    local_head = row - q_i * heads_per_cta
                    q_head_i = q_head_offset + local_head
                    if (q_i < q_len) & (q_head_i < group_size):
                        Partial_LSE[
                            batch_id,
                            split_id,
                            q_i,
                            kv_head * group_size + q_head_i,
                        ] = T.if_then_else(
                            l_i[row] == 0,
                            -(2**30),
                            T.log2(l_i[row]) + m_i[row] * scale_log2,
                        )

    return main


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def _paged_verify_combine_kernel(
    batch,
    heads,
    dim,
    max_splits,
    threads,
    min_tokens_per_split,
):
    partial_o_shape = [batch, max_splits, VERIFY_Q_BLOCK, heads, dim]
    partial_lse_shape = [batch, max_splits, VERIFY_Q_BLOCK, heads]

    @T.prim_func
    def main(
        Partial_O: T.Tensor(partial_o_shape, T.float16),
        Partial_LSE: T.Tensor(partial_lse_shape, T.float32),
        cache_seqlens: T.Tensor([batch], T.int32),
        query_start_loc: T.Tensor([batch + 1], T.int32),
        Output: T.Tensor([batch * VERIFY_Q_BLOCK, heads, dim], T.float16),
    ):
        with T.Kernel(VERIFY_Q_BLOCK, heads, batch, threads=threads) as (
            q_i,
            head_i,
            batch_id,
        ):
            lse = T.alloc_shared([max_splits], T.float32)
            lse_max = T.alloc_fragment([1], T.float32)
            lse_sum = T.alloc_fragment([1], T.float32)
            acc_o = T.alloc_fragment([dim], T.float32)

            active_splits = T.min(
                max_splits,
                T.max(
                    1,
                    T.ceildiv(cache_seqlens[batch_id], min_tokens_per_split),
                ),
            )
            q_start = query_start_loc[batch_id]
            q_len = query_start_loc[batch_id + 1] - q_start

            with T.If(q_i < q_len), T.Then():
                for split_i in T.Parallel(max_splits):
                    lse[split_i] = T.if_then_else(
                        split_i < active_splits,
                        Partial_LSE[batch_id, split_i, q_i, head_i],
                        -(2**30),
                    )

                T.fill(lse_max, -(2**30))
                for split_i in T.serial(max_splits):
                    lse_max[0] = T.max(lse_max[0], lse[split_i])
                T.fill(lse_sum, 0)
                for split_i in T.serial(max_splits):
                    if split_i < active_splits:
                        lse_sum[0] += T.exp2(lse[split_i] - lse_max[0])

                T.fill(acc_o, 0)
                for split_i in T.serial(max_splits):
                    if split_i < active_splits:
                        weight = T.exp2(lse[split_i] - lse_max[0]) / lse_sum[0]
                        for d in T.Parallel(dim):
                            acc_o[d] += weight * Partial_O[
                                batch_id, split_i, q_i, head_i, d
                            ].astype(T.float32)
                for d in T.Parallel(dim):
                    Output[q_start + q_i, head_i, d] = T.cast(acc_o[d], T.float16)

    return main


_VERIFY_KERNEL_CACHE = {}


def get_paged_verify_kernels(
    batch,
    heads,
    heads_kv,
    dim,
    block_size,
    num_pages,
    max_blocks,
    causal,
    fp8_kv=False,
    min_tokens_per_split=None,
):
    """Compile the grouped partial/combine pair for one CUDA-graph shape."""
    assert heads % heads_kv == 0
    group_size = heads // heads_kv
    heads_per_cta = min(4, group_size)
    gqa_ctas = math.ceil(group_size / heads_per_cta)
    max_splits = max(1, math.ceil(VERIFY_SM_TARGET / (batch * heads_kv * gqa_ctas)))
    block_n = {128: 64, 256: 32}.get(dim, 64)
    threads = 256
    if min_tokens_per_split is None:
        min_tokens_per_split = _verify_min_tokens_per_split()
    key = (
        batch,
        heads,
        heads_kv,
        dim,
        block_size,
        num_pages,
        max_blocks,
        causal,
        max_splits,
        block_n,
        threads,
        fp8_kv,
        min_tokens_per_split,
    )
    if key not in _VERIFY_KERNEL_CACHE:
        partial = _paged_verify_partial_kernel(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            page_block_size=block_size,
            max_blocks_per_seq=max_blocks,
            num_pages=num_pages,
            is_causal=causal,
            max_splits=max_splits,
            block_N=block_n,
            threads=threads,
            fp8_kv=fp8_kv,
            min_tokens_per_split=min_tokens_per_split,
        )
        combine = _paged_verify_combine_kernel(
            batch=batch,
            heads=heads,
            dim=dim,
            max_splits=max_splits,
            threads=128,
            min_tokens_per_split=min_tokens_per_split,
        )
        _VERIFY_KERNEL_CACHE[key] = (partial, combine, max_splits)
    return _VERIFY_KERNEL_CACHE[key]
