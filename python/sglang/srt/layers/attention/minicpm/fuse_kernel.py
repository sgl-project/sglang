import math
from functools import partial

import tilelang
import tilelang.language as T
import tilelang.math

_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def _fused_attn_pooling_online_topk(
    batch_size: int,
    groups: int,
    heads: int,
    dim: int,
    topk: int,
    max_seqlen_q_grid: int,  # Static param for grid (use bucketing)
    pooled_k_len: int,  # Static param (use bucketing) = ceil(max_seqlen_k / block_size)
    is_causal: bool,
    dense_len: int = 0,
    m_block_dim: int = 16,
    block_M: int = 16,
    block_N: int = 64,
    # infllmv2 pooling parameters:
    # block_stride = block_size // kernel_stride = 64 // 16 = 4
    # pad_len = kernel_size // kernel_stride - 1 = 32 // 16 - 1 = 1
    # num_offs = kernel_size // kernel_stride + block_size // kernel_stride - 1 = 2 + 4 - 1 = 5
    block_stride: int = 4,  # pool output block stride
    pad_len: int = 1,  # padding for pool blocks
    num_offs: int = 5,  # number of k positions each pool block reads
    kernel_stride: int = 16,
    block_size: int = 64,  # block size for q/k block computation
    init_blocks: int = 0,
    local_blocks: int = 0,
    num_stages: int = 0,
    threads: int = 128,
    dtype_str: str = "bfloat16",
):
    """
    Fused Attention + Max Pooling + Online TopK for prefill and decode.

    Chunk prefill support:
    - cache_lens: tensor of shape [batch_size], cache length for each batch
    - When cache_lens[i] = 0, it's standard prefill
    - When cache_lens[i] > 0, it's chunk prefill (continuing from cached state)

    Pooling logic aligned with infllmv2_cuda_impl:
    - For each pool block b, it aggregates k scores in range [b * block_stride - pad_len, b * block_stride - pad_len + num_offs)
    - block_stride = 4, pad_len = 1, num_offs = 5
    - pool block 0: k in [0-1, 0-1+5) = [-1, 4) -> [0, 4)
    - pool block 1: k in [3, 8)
    - pool block 2: k in [7, 12)
    - etc.
    """
    assert topk == tilelang.math.next_power_of_2(topk), "topk must be power of 2"

    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups

    # Dynamic dimensions - inferred from tensor shapes at runtime
    UQ = T.dynamic("UQ")
    UKV = T.dynamic("UKV")

    q_shape = [UQ * groups, head_kv, dim]
    kv_shape = [UKV, head_kv, dim]
    topk_indices_shape = [head_kv, UQ, topk]
    topk_values_shape = [head_kv, UQ, topk]

    dtype = dtype_str
    accum_dtype = "float"

    N = 2 * topk
    num_sort_iters = int(round(math.log2(N)))
    block_P = topk

    @T.macro
    def bitonic_sort(
        topk_index_shared: T.Buffer([N], "int32"),
        topk_value_shared: T.Buffer([N], "float32"),
    ):
        T.sync_threads()
        for i1 in T.serial(num_sort_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and (
                        (ascending and topk_value_shared[i] > topk_value_shared[j])
                        or (
                            not ascending
                            and topk_value_shared[i] < topk_value_shared[j]
                        )
                    ):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(kv_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], "int32"),
        cu_seqlens_k: T.Tensor([batch_size + 1], "int32"),
        cache_lens: T.Tensor(
            [batch_size], "int32"
        ),  # Per-batch cache length for chunk prefill
        TopkIndices: T.Tensor(topk_indices_shape, "int32"),
        TopkValues: T.Tensor(topk_values_shape, "float32"),
    ):
        with T.Kernel(max_seqlen_q_grid, head_kv, batch_size, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            topk_index_shared = T.alloc_shared([N], "int32")
            topk_value_shared = T.alloc_shared([N], "float32")
            pool_max_shared = T.alloc_shared([block_P], "float32")

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            acc_output = T.alloc_fragment([block_N], accum_dtype)

            batch_idx = bz
            kv_head_idx = by
            original_q_idx = bx

            q_start_idx = cu_seqlens_q[batch_idx]
            k_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            k_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = T.alloc_var("int32", init=q_end_idx - q_start_idx)
            k_current_seqlen = T.alloc_var("int32", init=k_end_idx - k_start_idx)

            # Chunk prefill: cache_len from tensor (0 for standard prefill, >0 for chunk prefill)
            cache_len = cache_lens[batch_idx]
            if not is_causal:
                active = cache_len + 1 >= dense_len
                q_current_seqlen = T.if_then_else(active, q_current_seqlen, 0)
                k_current_seqlen = T.if_then_else(active, k_current_seqlen, 0)
            if is_causal:
                actual_pooled_k_len = (
                    k_current_seqlen - 1 + pad_len
                ) // block_stride + 1
            else:
                actual_pooled_k_len = (1 + cache_len + block_size - 1) // block_size
            effective_pooled_k_len = T.min(actual_pooled_k_len, pooled_k_len)

            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            # Use q_end_idx to avoid out-of-bounds access for Q
            q_copy_end = T.min(
                q_start_idx * groups + (bx + 1) * block_M, q_end_idx * groups
            )
            T.copy(
                Q_unpad[
                    q_start_idx * groups + bx * block_M : q_copy_end, kv_head_idx, :
                ],
                Q_shared,
            )
            for i, d in T.Parallel(block_M, dim):
                if original_q_idx >= q_current_seqlen:
                    Q_shared[i, d] = 0

            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range_k = T.ceildiv(k_current_seqlen, block_N)

            for k in T.Pipelined(loop_range_k, num_stages=num_stages):
                # Use k_end_idx to avoid out-of-bounds access for last block
                k_copy_end = T.min(k_start_idx + (k + 1) * block_N, k_end_idx)
                T.copy(
                    K_unpad[k_start_idx + k * block_N : k_copy_end, kv_head_idx, :],
                    K_shared,
                )
                for i, d in T.Parallel(block_N, dim):
                    if k * block_N + i >= k_current_seqlen:
                        K_shared[i, d] = 0

                for i, j in T.Parallel(block_M, block_N):
                    k_idx = k * block_N + j
                    boundary_mask = (original_q_idx >= q_current_seqlen) or (
                        k_idx >= k_current_seqlen
                    )
                    if is_causal:
                        row_idx = original_q_idx * block_M + i + cache_len * block_M
                        orig_row_idx = row_idx // m_block_dim
                        orig_seqlen_q = (
                            (q_current_seqlen + cache_len) * block_M
                        ) // m_block_dim
                        compressed_seqlen_q = (
                            orig_seqlen_q - kernel_stride + 1
                        ) // kernel_stride
                        offset_row_idx = T.max(
                            0,
                            (orig_row_idx + 1) // kernel_stride
                            - 1
                            + k_current_seqlen
                            - compressed_seqlen_q,
                        )
                        q_compress_clamped = T.min(k_current_seqlen, offset_row_idx)
                        causal_mask = k_idx > q_compress_clamped
                        acc_s[i, j] = T.if_then_else(
                            boundary_mask or causal_mask, -1e9, 0
                        )
                    else:
                        acc_s[i, j] = T.if_then_else(boundary_mask, -1e9, 0)

                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )

                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

            loop_range_pool = T.ceildiv(effective_pooled_k_len, block_P)

            for p_block in T.serial(loop_range_pool):
                T.fill(pool_max_shared, float("-inf"))
                T.sync_threads()

                for k in T.serial(loop_range_k):
                    # Use k_end_idx to avoid out-of-bounds access for last block
                    k_copy_end = T.min(k_start_idx + (k + 1) * block_N, k_end_idx)
                    T.copy(
                        K_unpad[k_start_idx + k * block_N : k_copy_end, kv_head_idx, :],
                        K_shared,
                    )
                    for i, d in T.Parallel(block_N, dim):
                        if k * block_N + i >= k_current_seqlen:
                            K_shared[i, d] = 0

                    for i, j in T.Parallel(block_M, block_N):
                        k_idx = k * block_N + j
                        boundary_mask = (original_q_idx >= q_current_seqlen) or (
                            k_idx >= k_current_seqlen
                        )
                        if is_causal:
                            row_idx = original_q_idx * block_M + i + cache_len * block_M
                            orig_row_idx = row_idx // m_block_dim
                            orig_seqlen_q = (
                                (q_current_seqlen + cache_len) * block_M
                            ) // m_block_dim
                            compressed_seqlen_q = (
                                orig_seqlen_q - kernel_stride + 1
                            ) // kernel_stride
                            offset_row_idx = T.max(
                                0,
                                (orig_row_idx + 1) // kernel_stride
                                - 1
                                + k_current_seqlen
                                - compressed_seqlen_q,
                            )
                            q_compress_clamped = T.min(k_current_seqlen, offset_row_idx)
                            causal_mask = k_idx > q_compress_clamped
                            acc_s[i, j] = T.if_then_else(
                                boundary_mask or causal_mask, -1e9, 0
                            )
                        else:
                            acc_s[i, j] = T.if_then_else(boundary_mask, -1e9, 0)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    # Normalize and handle NaN/Inf (when logsum is 0 or very small)
                    for i, j in T.Parallel(block_M, block_N):
                        normalized = (
                            T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            / logsum[i]
                        )
                        # Handle NaN/Inf: if logsum is very small or result is invalid, set to 0
                        acc_s[i, j] = T.if_then_else(
                            (logsum[i] > 1e-10)
                            and (normalized >= 0)
                            and (normalized <= 1e10),
                            normalized,
                            T.Cast(accum_dtype, 0.0),
                        )

                    T.fill(acc_output, 0)
                    T.reduce_sum(acc_s, acc_output, dim=0)

                    # infllmv2 block-based pooling:
                    # For pool block b, it aggregates k scores in range [b * block_stride - pad_len, b * block_stride - pad_len + num_offs)
                    # For k_idx, it contributes to pool block b if:
                    #   b * block_stride - pad_len <= k_idx < b * block_stride - pad_len + num_offs
                    # So:
                    #   start_b = max(0, ceil((k_idx - num_offs + 1 + pad_len) / block_stride))
                    #   end_b = floor((k_idx + pad_len) / block_stride)
                    for j in T.Parallel(block_N):
                        k_idx = k * block_N + j
                        if (
                            original_q_idx < q_current_seqlen
                            and k_idx < k_current_seqlen
                        ):
                            # Calculate which pool blocks this k_idx contributes to
                            # start_b = ceil((k_idx - num_offs + 1 + pad_len) / block_stride)
                            #         = ceil((k_idx - 5 + 1 + 1) / 4) = ceil((k_idx - 3) / 4)
                            start_pool = T.max(
                                0,
                                (k_idx - num_offs + 1 + pad_len + block_stride - 1)
                                // block_stride,
                            )
                            end_pool = T.min(
                                effective_pooled_k_len - 1,
                                (k_idx + pad_len) // block_stride,
                            )

                            pool_block_start = p_block * block_P
                            pool_block_end = T.min(
                                (p_block + 1) * block_P, effective_pooled_k_len
                            )

                            for p_off in T.serial(
                                num_offs
                            ):  # at most num_offs pool blocks per k
                                p_idx = start_pool + p_off
                                if (
                                    p_idx >= pool_block_start
                                    and p_idx < pool_block_end
                                    and p_idx <= end_pool
                                ):
                                    local_p_idx = p_idx - pool_block_start
                                    T.atomic_max(
                                        pool_max_shared[local_p_idx], acc_output[j]
                                    )
                    T.sync_threads()

                for p_off in T.Parallel(block_P):
                    p_idx = p_block * block_P + p_off
                    if (
                        p_idx < effective_pooled_k_len
                        and original_q_idx < q_current_seqlen
                    ):
                        off_bq = (original_q_idx + cache_len) // block_size
                        off_bk = p_idx

                        # Match Torch implementation exactly:
                        # if init_blocks > 0 and off_bk < init_blocks:
                        #     should_mask_inf = True
                        # elif local_blocks > 0:
                        #     if (off_bq >= off_bk) and (off_bq <= off_bk + local_blocks):
                        #         should_mask_inf = True
                        is_init_masked = (init_blocks > 0) and (off_bk < init_blocks)
                        is_local_masked = (
                            (local_blocks > 0)
                            and (off_bq >= off_bk)
                            and (off_bq <= off_bk + local_blocks)
                        )
                        # Use elif logic: local_blocks check only when not init_masked
                        is_masked = T.if_then_else(
                            is_init_masked, 1, T.if_then_else(is_local_masked, 1, 0)
                        )

                        topk_index_shared[topk + p_off] = p_idx
                        # Use inf for masked blocks to force selection
                        # Compare only index sets, not order
                        topk_value_shared[topk + p_off] = T.if_then_else(
                            is_masked == 1,
                            T.Cast("float32", float("inf")),
                            pool_max_shared[p_off],
                        )
                T.sync_threads()

                bitonic_sort(topk_index_shared, topk_value_shared)

            for i in T.Parallel(topk):
                if original_q_idx < q_current_seqlen:
                    global_q_idx = q_start_idx + original_q_idx
                    TopkIndices[kv_head_idx, global_q_idx, i] = topk_index_shared[i]
                    TopkValues[kv_head_idx, global_q_idx, i] = topk_value_shared[i]

    return main


fused_attn_pooling_online_topk_prefill = partial(
    _fused_attn_pooling_online_topk, is_causal=True
)
fused_attn_pooling_online_topk_decode = partial(
    _fused_attn_pooling_online_topk,
    max_seqlen_q_grid=1,
    is_causal=False,
)
