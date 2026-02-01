# -*- coding: utf-8 -*-

import math

from typing import Optional, List

import torch

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flash_mla_varlen_func_kernel(UQ, UKV, heads, dim_qk, dim_vo, softmax_scale, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    batch_size = T.dynamic("batch_size")
    scale = softmax_scale * 1.44269504  # log2(e)
    q_shape = [UQ, heads, dim_qk]
    k_shape = [UKV, heads, dim_qk]
    v_shape = [UKV, heads, dim_vo]
    o_shape = [UQ, heads, dim_vo]

    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_vo], dtype)
            O_shared = T.alloc_shared([block_M, dim_vo], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_vo], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            kv_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :], Q_shared
            )  # OOB positions will be handled below

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            offset = kv_current_seqlen - q_current_seqlen  # always align on the right
            loop_range = (
                T.min(T.ceildiv(offset + (bx + 1) * block_M, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Q * K
                T.copy(
                    K_unpad[kv_start_idx + k * block_N : kv_start_idx + k * block_N + block_N, head_idx, :], K_shared
                )  # OOB positions will be handled below
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i + offset < k * block_N + j)
                            or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen), -1e9, 0
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                # in the first ceil_div(kBlockM, kBlockN) steps.
                # for i in T.Parallel(block_M):
                #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    # max * log_2(e)) This allows the compiler to use the ffma
                    # instruction instead of fadd and fmul separately.
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                # Rescale
                for i, j in T.Parallel(block_M, dim_vo):
                    acc_o[i, j] *= scores_scale[i]

                # V * softmax(Q * K)
                T.copy(
                    V_unpad[kv_start_idx + k * block_N : kv_start_idx + k * block_N + block_N, head_idx, :], V_shared
                )  # OOB positions' weights are 0

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim_vo):
                # When sq > skv, some tokens can see nothing
                acc_o[i, j] = 0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, dim_vo):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flash_mla_with_kvcache_kernel(
    batch, 
    seqlen_q,
    h_q, 
    h_kv, 
    dv, 
    dpe, 
    block_N, 
    block_H, 
    num_split, 
    block_size, 
    num_pages, 
    max_num_blocks_per_seq, 
    softmax_scale=None
):
    if softmax_scale is None:
        softmax_scale = (dv + dpe) ** -0.5
    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32
    
    # Enforce constraints for this specific kernel version
    assert h_kv == 1, "h_kv must be 1"
    
    # kv_group_num equals h_q when h_kv is 1
    kv_group_num = h_q 
    
    VALID_BLOCK_H = min(block_H, kv_group_num)
    VALID_BLOCK_H = max(VALID_BLOCK_H, 16)
    block_H = VALID_BLOCK_H
    
    assert block_size >= block_N and block_size % block_N == 0, \
        "block_size must be larger than block_N and a multiple of block_N"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, seqlen_q, h_q, dv + dpe], dtype),
        # Paged KV: [num_pages, page_size, heads=1, dim]
        KV: T.Tensor([num_pages, block_size, 1, dv + dpe], dtype),
        block_table: T.Tensor([batch, max_num_blocks_per_seq], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, seqlen_q, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, seqlen_q, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, seqlen_q, h_q, dv], dtype),
    ):
        # Grid: batch, split_Q_heads, split_seq
        # Since h_kv=1, all Q heads within the tile attend to the same KV head (0)
        with T.Kernel(batch, seqlen_q * (h_q + VALID_BLOCK_H - 1) // VALID_BLOCK_H, num_split, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            seq_q_head_block_idx = by
            seq_q_idx = T.floordiv(seq_q_head_block_idx, (h_q + VALID_BLOCK_H - 1) // VALID_BLOCK_H)
            head_block_idx = T.floormod(seq_q_head_block_idx, (h_q + VALID_BLOCK_H - 1) // VALID_BLOCK_H)

            T.use_swizzle(10)

            T.copy(Q[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, : dv], Q_shared)
            T.copy(Q[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, dv:], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            total_blocks = T.ceildiv(cache_seqlens[bx] - seqlen_q + seq_q_idx + 1, block_N)
            blocks_per_split = T.floordiv(total_blocks, num_split)
            remaining_blocks = T.floormod(total_blocks, num_split)
            loop_range = blocks_per_split + T.if_then_else(bz < remaining_blocks, 1, 0)
            start = (blocks_per_split * bz + T.min(bz, remaining_blocks)) * block_N

            for k in T.Pipelined(loop_range, num_stages=2):
                global_token_idx = start + k * block_N
                
                logical_page_idx = global_token_idx // block_size
                page_offset = global_token_idx % block_size
                physical_page_id = block_table[bx, logical_page_idx]

                # KV Head is fixed to 0
                T.copy(KV[physical_page_id, page_offset : page_offset + block_N, 0, :dv], KV_shared)
                T.copy(KV[physical_page_id, page_offset : page_offset + block_N, 0, dv:], K_pe_shared)
                
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(start + k * block_N + j >= cache_seqlens[bx] - seqlen_q + seq_q_idx + 1, -T.infinity(accum_dtype), acc_s[i, j])
                
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                T.copy(S_shared, acc_s_cast)
                
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            
            T.copy(logsum, glse[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, bz, :])

        # Combine kernel
        with T.Kernel(seqlen_q * h_q, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dv], dtype)
            o_accum_local = T.alloc_fragment([dv], accum_dtype)
            # lse_local_split = T.alloc_var(accum_dtype)
            # lse_logsum_local = T.alloc_var(accum_dtype)
            # lse_max_local = T.alloc_var(accum_dtype)
            # scale_local = T.alloc_var(accum_dtype)
            lse_local_split = T.alloc_fragment([1], accum_dtype)
            lse_logsum_local = T.alloc_fragment([1], accum_dtype)
            lse_max_local = T.alloc_fragment([1], accum_dtype)
            scale_local = T.alloc_fragment([1], accum_dtype)

            seq_q_head_idx = by
            seq_q_idx = T.floordiv(seq_q_head_idx, h_q)
            head_idx = T.floormod(seq_q_head_idx, h_q)

            lse_logsum_local[0] = 0.0
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, seq_q_idx, head_idx, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, seq_q_idx, head_idx, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dv):
                    po_local[i] = Output_partial[bz, seq_q_idx, head_idx, k, i]
                lse_local_split[0] = glse[bz, seq_q_idx, head_idx, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dv):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dv):
                Output[bz, seq_q_idx, head_idx, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor([batch, seqlen_q, h_q, dv+ dpe], dtype),
        KV: T.Tensor([num_pages, block_size, 1, dv+ dpe], dtype),
        block_table: T.Tensor([batch, max_num_blocks_per_seq], T.int32),
        cache_seqlens: T.Tensor([batch], T.int32),
        glse: T.Tensor([batch, seqlen_q, h_q, num_split], dtype),
        Output_partial: T.Tensor([batch, seqlen_q, h_q, num_split, dv], dtype),
        Output: T.Tensor([batch, seqlen_q, h_q, dv], dtype),
    ):
        with T.Kernel(batch, seqlen_q * h_q // VALID_BLOCK_H, threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dv], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dpe], dtype)
            KV_shared = T.alloc_shared([block_N, dv], dtype)
            K_pe_shared = T.alloc_shared([block_N, dpe], dtype)
            O_shared = T.alloc_shared([block_H, dv], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dv], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            seq_q_head_block_idx = by
            seq_q_idx = T.floordiv(seq_q_head_block_idx, (h_q + VALID_BLOCK_H - 1) // VALID_BLOCK_H)
            head_block_idx = T.floormod(seq_q_head_block_idx, (h_q + VALID_BLOCK_H - 1) // VALID_BLOCK_H)

            T.use_swizzle(10)

            T.copy(Q[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, :dv], Q_shared)
            T.copy(Q[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, dv:], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(cache_seqlens[bx] - seqlen_q + seq_q_idx + 1, block_N)
            for kr in T.Pipelined(loop_range, num_stages=2):
                k = loop_range - 1 - kr
                global_token_idx = k * block_N
                
                logical_page_idx = global_token_idx // block_size
                page_offset = global_token_idx % block_size
                physical_page_id = block_table[bx, logical_page_idx]

                # KV Head is fixed to 0
                T.copy(KV[physical_page_id, page_offset : page_offset + block_N, 0, : dv], KV_shared)
                T.copy(KV[physical_page_id, page_offset : page_offset + block_N, 0, dv:], K_pe_shared)
                
                T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                if kr == 0:
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= cache_seqlens[bx] - seqlen_q + seq_q_idx + 1, -T.infinity(accum_dtype), acc_s[i, j])
                
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dv):
                    acc_o[i, j] *= scores_scale[i]
                
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            
            for i, j in T.Parallel(block_H, dv):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bx, seq_q_idx, head_block_idx * VALID_BLOCK_H : (head_block_idx + 1) * VALID_BLOCK_H, :])

    if num_split > 1:
        return main_split
    else:
        return main_no_split


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        # tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def zigzag_attn_varlen_func_kernel(num_heads, dim_qk, dim_vo, softmax_scale, block_M=64, block_N=128, num_stages=1, threads=128):
    num_seqs = T.dynamic("num_seqs")
    nnz_qo = T.dynamic("nnz_qo")
    nnz_kv = T.dynamic("nnz_kv")

    softmax_scale = softmax_scale * 1.44269504 # log2(e)

    q_shape = [nnz_qo, num_heads, dim_qk]
    k_shape = [nnz_kv, num_heads, dim_qk]
    v_shape = [nnz_kv, num_heads, dim_vo]
    o_shape = [nnz_qo, num_heads, dim_vo]
    streaming_info_shape = [num_heads, 2]
    head_mask_type_shape = [num_heads]

    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([num_seqs + 1], T.int32),
        cu_seqlens_k: T.Tensor([num_seqs + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
        streaming_info: T.Tensor(streaming_info_shape, T.int32),
        head_mask_type: T.Tensor(head_mask_type_shape, T.int32),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), num_heads, num_seqs, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_vo], dtype)
            O_shared = T.alloc_shared([block_M, dim_vo], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_vo], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
    
            head_idx = by
            batch_idx = bz

            q_start_idx = cu_seqlens_q[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            q_seqlen = q_end_idx - q_start_idx

            kv_start_idx = cu_seqlens_k[batch_idx]
            kv_end_idx = cu_seqlens_k[batch_idx + 1]
            kv_seqlen = kv_end_idx - kv_start_idx

            sink_blocks = streaming_info[head_idx, 0]
            recent_blocks = streaming_info[head_idx, 1]
            head_mask_type_ = head_mask_type[head_idx]

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :], Q_shared
            )  # OOB positions will be handled below

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            offset = kv_seqlen - q_seqlen  # always align on the right

            cache_seqlen = T.min(offset + (bx + 1) * block_M, kv_seqlen)
            total_blocks = T.ceildiv(cache_seqlen, block_N)

            sink_block_end = T.min(sink_blocks, total_blocks)
            recent_block_start = T.max(sink_block_end, total_blocks - recent_blocks - 1) # Add one more block for chunked case.
            # streaming_blocks = sink_block_end + T.if_then_else(recent_block_start < total_blocks, total_blocks - recent_block_start, 0)

            for k in T.Pipelined(total_blocks, num_stages=num_stages):
            # for k in T.Pipelined(streaming_blocks, num_stages=num_stages):
                if head_mask_type_ == -1 and (k < sink_block_end or k >= recent_block_start):
                # if head_mask_type_ == -1:
                    # Q * K
                    T.copy(
                        K_unpad[kv_start_idx + k * block_N: kv_start_idx + k * block_N + block_N , head_idx, :], K_shared
                    )  # OOB positions will be handled below

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i + offset < k * block_N + j)
                            or (bx * block_M + i >= q_seqlen or k * block_N + j >= kv_seqlen)
                            or (k >= recent_block_start and k < T.ceildiv(bx * block_M + i + offset + 1, block_N) - recent_blocks), # Filter one more block for chunked case.
                            -1e9,
                            0,
                        )

                    # block_start_idx = T.if_then_else(k < sink_block_end, k, recent_block_start + k - sink_block_end)
                    # T.copy(
                    #     K_unpad[kv_start_idx + block_start_idx * block_N: kv_start_idx + block_start_idx * block_N + block_N , head_idx, :], K_shared
                    # )  # OOB positions will be handled below

                    # for i, j in T.Parallel(block_M, block_N):
                    #     acc_s[i, j] = T.if_then_else(
                    #         (bx * block_M + i + offset < block_start_idx * block_N + j)
                    #         or (bx * block_M + i >= q_seqlen or block_start_idx * block_N + j >= kv_seqlen)
                    #         or (block_start_idx >= recent_block_start and block_start_idx < T.ceildiv(bx * block_M + i + offset + 1, block_N) - recent_blocks), # Filter one more block for chunked case.
                    #         -1e9,
                    #         0,
                    #     )

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                    # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                    # in the first ceil_div(kBlockM, kBlockN) steps.
                    # for i in T.Parallel(block_M):
                    #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * softmax_scale - scores_max[i] * softmax_scale)
                    for i, j in T.Parallel(block_M, block_N):
                        # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        # max * log_2(e)) This allows the compiler to use the ffma
                        # instruction instead of fadd and fmul separately.
                        acc_s[i, j] = T.exp2(acc_s[i, j] * softmax_scale - scores_max[i] * softmax_scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    # Rescale
                    for i, j in T.Parallel(block_M, dim_vo):
                        acc_o[i, j] *= scores_scale[i]

                    # V * softmax(Q * K)
                    T.copy(
                        V_unpad[kv_start_idx + k * block_N: kv_start_idx + k * block_N + block_N, head_idx, :], V_shared
                    )  # OOB positions' weights are 0
                    # T.copy(
                    #     V_unpad[kv_start_idx + block_start_idx * block_N: kv_start_idx + block_start_idx * block_N + block_N, head_idx, :], V_shared
                    # )  # OOB positions' weights are 0

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim_vo):
                # When sq > skv, some tokens can see nothing
                acc_o[i, j] = 0 if bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, dim_vo):
                if bx * block_M + i < q_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def zigzag_attn_with_kvcache_kernel(num_seqs, seqlen_q, num_heads, num_kv_heads, dim_nope, dim_rope, softmax_scale, num_pages, page_size, block_H=64, block_N=128, num_splits=1, num_stages=1, threads=128):
    # num_seqs = T.dynamic("num_seqs")
    max_num_pages = T.dynamic("max_num_pages")

    softmax_scale = softmax_scale * 1.44269504 # log2(e)

    num_query_groups = num_heads // num_kv_heads
    q_shape = [num_seqs, seqlen_q, num_heads, dim_nope + dim_rope]
    kv_shape = [num_pages, page_size, num_kv_heads, dim_nope + dim_rope]
    block_table_shape = [num_seqs, max_num_pages]
    lse_shape = [num_seqs, seqlen_q, num_heads, num_splits]
    o_partial_shape = [num_seqs, seqlen_q, num_heads, num_splits, dim_nope]
    o_shape = [num_seqs, seqlen_q, num_heads, dim_nope]
    streaming_info_shape = [num_heads, 2]
    head_mask_type_shape = [num_heads]

    assert num_kv_heads == 1
    assert page_size == 64
    assert block_N >= page_size and block_N % page_size == 0, "page_size must be smaller than block_N and divisible by block_N."
    block_S = page_size
    block_H = min(block_H, num_query_groups) # valid_block_H <=(by num_query_groups) block_H.
    if block_H % 16 != 0:
        block_H = (block_H + 16 - 1) // 16 * 16
    num_head_blocks = (num_query_groups + block_H - 1) // block_H

    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main_split(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        block_table: T.Tensor(block_table_shape, T.int32),
        cache_seqlens: T.Tensor([num_seqs], T.int32),
        Lse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(o_partial_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        streaming_info: T.Tensor(streaming_info_shape, T.int32),
        head_mask_type: T.Tensor(head_mask_type_shape, T.int32),
    ):
        with T.Kernel(num_splits, seqlen_q * num_head_blocks, num_seqs, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_H, dim_nope], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dim_rope], dtype)
            KV_shared = T.alloc_shared([block_S, dim_nope], dtype)
            K_pe_shared = T.alloc_shared([block_S, dim_rope], dtype)
            O_shared = T.alloc_shared([block_H, dim_nope], dtype)
            S_shared = T.alloc_shared([block_H, block_S], dtype)
            acc_s = T.alloc_fragment([block_H, block_S], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_S], dtype)
            acc_o = T.alloc_fragment([block_H, dim_nope], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            split_idx = bx
            seq_q_head_block_idx = by
            seq_q_idx = T.floordiv(seq_q_head_block_idx, num_head_blocks)
            head_block_idx = T.floormod(seq_q_head_block_idx, num_head_blocks)
            batch_idx = bz
            
            sink_blocks = streaming_info[head_block_idx * block_H, 0]
            recent_blocks = streaming_info[head_block_idx * block_H, 1]
            head_mask_type_ = head_mask_type[head_block_idx * block_H]

            T.use_swizzle(10)

            T.copy(Q[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, : dim_nope], Q_shared)
            T.copy(Q[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, dim_nope: ], Q_pe_shared)
            # OOB positions will be handled below.

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            cache_seqlen = cache_seqlens[batch_idx] - seqlen_q + seq_q_idx + 1 # Reduce seqlen for MTP case.
            total_blocks = T.ceildiv(cache_seqlen, block_N)
            pages_per_block = T.ceildiv(block_N, block_S)

            sink_block_end = T.min(sink_blocks, total_blocks)
            sink_page_end = sink_block_end * pages_per_block
            recent_block_start = T.max(sink_block_end, total_blocks - recent_blocks)
            recent_page_start = recent_block_start * pages_per_block

            effective_blocks = total_blocks - recent_block_start + sink_block_end
            blocks_per_split = T.floordiv(effective_blocks, num_splits)
            remaining_blocks = T.floormod(effective_blocks, num_splits)
            kv_start_page_idx = (blocks_per_split * split_idx + T.min(split_idx, remaining_blocks)) * pages_per_block
            streaming_loop_range = (blocks_per_split + T.if_then_else(split_idx < remaining_blocks, 1, 0)) * pages_per_block

            for p in T.Pipelined(streaming_loop_range, num_stages=num_stages):
                if head_mask_type_ == -1:
                    # Q * K
                    page_start_idx = T.if_then_else(kv_start_page_idx + p < sink_page_end, kv_start_page_idx + p, recent_page_start + kv_start_page_idx + p - sink_page_end) * block_S
                    physical_page_idx = block_table[batch_idx, T.floordiv(page_start_idx, block_S)]
                    T.copy(KV[physical_page_idx, :, 0, : dim_nope], KV_shared)
                    T.copy(KV[physical_page_idx, :, 0, dim_nope: ], K_pe_shared)
                    # OOB positions will be handled below.

                    T.clear(acc_s)
                    for i, j in T.Parallel(block_H, block_S):
                        acc_s[i, j] = T.if_then_else(
                            page_start_idx + j >= cache_seqlen,
                            -1e9,
                            0
                        )
                    
                    T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                    # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                    # in the first ceil_div(kBlockM, kBlockN) steps.
                    # for i in T.Parallel(block_H):
                    #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])    
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * softmax_scale - scores_max[i] * softmax_scale)
                    for i, j in T.Parallel(block_H, block_S):
                        # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        # max * log_2(e)) This allows the compiler to use the ffma
                        # instruction instead of fadd and fmul separately.
                        acc_s[i, j] = T.exp2(acc_s[i, j] * softmax_scale - scores_max[i] * softmax_scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    T.copy(S_shared, acc_s_cast)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    # Rescale
                    for i, j in T.Parallel(block_H, dim_nope):
                        acc_o[i, j] *= scores_scale[i]
                    
                    # V * softmax(Q * K)
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i, j in T.Parallel(block_H, dim_nope):
                acc_o[i, j] = acc_o[i, j] / logsum[i]
            
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * softmax_scale
            
            T.copy(logsum, Lse[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, split_idx])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, split_idx, :])

        with T.Kernel(seqlen_q * num_query_groups, num_seqs, threads=threads // 2) as (by, bz):
            po_local = T.alloc_fragment([dim_nope], dtype)
            o_accum_local = T.alloc_fragment([dim_nope], accum_dtype)
            # lse_local_split = T.alloc_var(accum_dtype)
            # lse_logsum_local = T.alloc_var(accum_dtype)
            # lse_max_local = T.alloc_var(accum_dtype)
            # scale_local = T.alloc_var(accum_dtype)
            lse_local_split = T.alloc_fragment([1], accum_dtype)
            lse_logsum_local = T.alloc_fragment([1], accum_dtype)
            lse_max_local = T.alloc_fragment([1], accum_dtype)
            scale_local = T.alloc_fragment([1], accum_dtype)

            seq_q_head_idx = by
            seq_q_idx = T.floordiv(seq_q_head_idx, num_query_groups)
            head_idx = T.floormod(seq_q_head_idx, num_query_groups)
            batch_idx = bz

            lse_logsum_local[0] = 0.0
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)

            # Pass 1: Find Global Max LSE
            for k in T.serial(num_splits):
                lse_max_local[0] = T.max(lse_max_local[0], Lse[batch_idx, seq_q_idx, head_idx, k])
            
            # Pass 2: Calculate LogSumExp
            for k in T.serial(num_splits):
                lse_local_split[0] = Lse[batch_idx, seq_q_idx, head_idx, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            
            # Pass 3: Accumulate Weighted Output
            for k in T.serial(num_splits):
                for i in T.Parallel(dim_nope):
                    po_local[i] = Output_partial[batch_idx, seq_q_idx, head_idx, k, i]
                
                lse_local_split[0] = Lse[batch_idx, seq_q_idx, head_idx, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                
                for i in T.Parallel(dim_nope):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            
            for i in T.Parallel(dim_nope):
                Output[batch_idx, seq_q_idx, head_idx, i] = o_accum_local[i]

    @T.prim_func
    def main_no_split(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        block_table: T.Tensor(block_table_shape, T.int32),
        cache_seqlens: T.Tensor([num_seqs], T.int32),
        Lse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(o_partial_shape, dtype),
        Output: T.Tensor(o_shape, dtype),
        streaming_info: T.Tensor(streaming_info_shape, T.int32),
        head_mask_type: T.Tensor(head_mask_type_shape, T.int32),
    ):
        with T.Kernel(seqlen_q * num_head_blocks, num_seqs, threads=threads) as (by, bz):
            Q_shared = T.alloc_shared([block_H, dim_nope], dtype)
            Q_pe_shared = T.alloc_shared([block_H, dim_rope], dtype)
            KV_shared = T.alloc_shared([block_S, dim_nope], dtype)
            K_pe_shared = T.alloc_shared([block_S, dim_rope], dtype)
            O_shared = T.alloc_shared([block_H, dim_nope], dtype)
            S_shared = T.alloc_shared([block_H, block_S], dtype)
            acc_s = T.alloc_fragment([block_H, block_S], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_S], dtype)
            acc_o = T.alloc_fragment([block_H, dim_nope], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            seq_q_head_block_idx = by
            seq_q_idx = T.floordiv(seq_q_head_block_idx, num_head_blocks)
            head_block_idx = T.floormod(seq_q_head_block_idx, num_head_blocks)
            batch_idx = bz

            sink_blocks = streaming_info[head_block_idx * block_H, 0]
            recent_blocks = streaming_info[head_block_idx * block_H, 1]
            head_mask_type_ = head_mask_type[head_block_idx * block_H]

            T.use_swizzle(10)

            T.copy(Q[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, : dim_nope], Q_shared)
            T.copy(Q[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, dim_nope: ], Q_pe_shared)
            # OOB positions will be handled below.

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            cache_seqlen = cache_seqlens[batch_idx] - seqlen_q + seq_q_idx + 1 # Reduce seqlen for MTP case.
            total_blocks = T.ceildiv(cache_seqlen, block_N)
            pages_per_block = T.ceildiv(block_N, block_S)

            sink_block_end = T.min(sink_blocks, total_blocks)
            sink_page_end = sink_block_end * pages_per_block
            recent_block_start = T.max(sink_block_end, total_blocks - recent_blocks)
            recent_page_start = recent_block_start * pages_per_block

            effective_blocks = total_blocks - recent_block_start + sink_block_end
            kv_start_page_idx = 0
            streaming_loop_range = effective_blocks * pages_per_block

            for p in T.Pipelined(streaming_loop_range, num_stages=num_stages):
                if head_mask_type_ == -1:
                    # Q * K
                    page_start_idx = T.if_then_else(kv_start_page_idx + p < sink_page_end, kv_start_page_idx + p, recent_page_start + kv_start_page_idx + p - sink_page_end) * block_S
                    physical_page_idx = block_table[batch_idx, T.floordiv(page_start_idx, block_S)] 
                    T.copy(KV[physical_page_idx, :, 0, : dim_nope], KV_shared)
                    T.copy(KV[physical_page_idx, :, 0, dim_nope: ], K_pe_shared)
                    # OOB positions will be handled below.

                    T.clear(acc_s)
                    for i, j in T.Parallel(block_H, block_S):
                        acc_s[i, j] = T.if_then_else(
                            page_start_idx + j >= cache_seqlen,
                            -1e9,
                            0
                        )
                    
                    T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                    # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                    # in the first ceil_div(kBlockM, kBlockN) steps.
                    # for i in T.Parallel(block_M):
                    #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])    
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * softmax_scale - scores_max[i] * softmax_scale)
                    for i, j in T.Parallel(block_H, block_S):
                        # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        # max * log_2(e)) This allows the compiler to use the ffma
                        # instruction instead of fadd and fmul separately.
                        acc_s[i, j] = T.exp2(acc_s[i, j] * softmax_scale - scores_max[i] * softmax_scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    T.copy(acc_s, S_shared)
                    T.copy(S_shared, acc_s_cast)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    # Rescale
                    for i, j in T.Parallel(block_H, dim_nope):
                        acc_o[i, j] *= scores_scale[i]
                    
                    # V * softmax(Q * K)
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for i, j in T.Parallel(block_H, dim_nope):
                acc_o[i, j] = acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[batch_idx, seq_q_idx, head_block_idx * block_H: head_block_idx * block_H + block_H, :])

    if num_splits > 1:
        return main_split
    else:
        return main_no_split


def get_splits(
    batch_size: int, 
    num_heads: int,
    seqlen_q: int, 
    avg_seqlen_k: int, 
    block_size_h: int = 64,
    block_size_n: int = 128,
    streaming_info: Optional[List[int]] = None,
):
    """
    Calculates the optimal static num_splits to saturate the GPU
    without incurring unnecessary reduction overhead.
    """
    # 1. Get Device Capabilities
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    num_sms = props.multi_processor_count
    
    # 2. Define Saturation Target
    # We usually want 1 to 2 waves of blocks per SM to hide latency.
    # For MLA (memory bound), 1 full wave is often enough, but 2 is safer.
    target_blocks_per_sm = 2 
    target_total_blocks = num_sms * target_blocks_per_sm

    # 3. Calculate Effective KV Length (Maybe The "Streaming" Adjustment)
    # The effective length is capped by the cache capacity 
    # (Sink + Local) when sequence length exceeds the window.
    effective_seqlen_k = avg_seqlen_k
    if streaming_info is not None and len(streaming_info) >= 2:
        sink_block_num = streaming_info[0]
        local_block_num = streaming_info[1]
        
        effective_seqlen_k = min(avg_seqlen_k, (sink_block_num + local_block_num) * block_size_n)
    
    # 4. Calculate "Natural" Parallelism
    # The number of independent tasks available without splitting KV.
    # Usually: Batch * Heads (since seqlen_q is typically 1 in decoding)
    # ceil_div(num_heads, block_size_h) handles cases where heads are grouped.
    num_head_blocks = (num_heads + block_size_h - 1) // block_size_h
    natural_blocks = batch_size * seqlen_q * num_head_blocks
    
    # 5. Determine Split Ratio needed to hit target
    if natural_blocks >= target_total_blocks:
        # We already have enough parallelism to saturate the GPU.
        # Splitting further just adds overhead.
        return 1
    
    # We need to split to create more blocks
    needed_splits = (target_total_blocks + natural_blocks - 1) // natural_blocks
    
    # 5. Clamp based on Sequence Length (The "Don't shred it" check)
    # We don't want a tile processing fewer than, say, 128 tokens.
    # Otherwise, loop overhead dominates the math.
    min_tokens_per_tile = block_size_n
    max_splits_possible = max(1, effective_seqlen_k // min_tokens_per_tile)
    
    optimal_split = min(needed_splits, max_splits_possible)
    
    # 6. (Optional) Power of 2 alignment often helps compiler optimizations
    # Rounds to nearest power of 2 (1, 2, 4, 8, 16...)
    if optimal_split > 1:
        optimal_split = 2 ** math.floor(math.log2(optimal_split))
        
    return int(max(1, optimal_split))


def zigzag_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool = True,
    streaming_info: torch.Tensor = None,
    head_mask_type: torch.Tensor = None,
):
    assert causal == True

    nnz_qo, num_heads_q, head_dim_qk = q.shape
    # nnz_kv, num_heads_kv, head_dim_qk = v.shape
    nnz_kv, num_heads_kv, head_dim_vo = v.shape
    assert num_heads_q == num_heads_kv

    # streaming_info = torch.tensor(
    #     [[1, 7]] * q.shape[1],
    #     device=q.device,
    #     dtype=torch.int32,
    # )
    # head_mask_type = torch.full(
    #     (q.shape[1],),
    #     -1,
    #     device=q.device,
    #     dtype=torch.int32,
    # )

    if streaming_info is not None:
        block_M = 128
        block_N = 128
        num_stages = 1
        threads = 256

        kernel = zigzag_attn_varlen_func_kernel(
            num_heads_q, head_dim_qk, head_dim_vo, softmax_scale, block_M, block_N, num_stages, threads
        )
        return kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, streaming_info, head_mask_type)
    else:
        kernel = flash_mla_varlen_func_kernel(
            nnz_qo, nnz_kv, num_heads_q, head_dim_qk, head_dim_vo, softmax_scale, True, block_M=128, block_N=128, num_stages=1, threads=256
        )
        return kernel(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)



def zigzag_attn_with_kvcache(
    q: torch.Tensor,
    kv: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    dim_nope: int,
    softmax_scale: float,
    causal: bool = True,
    streaming_info: torch.Tensor = None,
    head_mask_type: torch.Tensor = None,
):
    assert causal == True

    batch_size, seqlen_q, num_heads_q, head_dim_qk = q.shape
    num_pages, page_size, num_heads_kv, head_dim_vo = kv.shape
    assert num_heads_kv == 1
    assert head_dim_qk == head_dim_vo

    # streaming_info = torch.tensor(
    #     [[1, 7]] * q.shape[2],
    #     device=q.device,
    #     dtype=torch.int32,
    # )
    # head_mask_type = torch.full(
    #     (q.shape[2],),
    #     -1,
    #     device=q.device,
    #     dtype=torch.int32,
    # )

    if streaming_info is not None:
        block_H = 64
        block_N = 128
        # num_splits = get_splits(batch_size, num_heads_q, seqlen_q, torch.mean(cache_seqlens.float()).int().item(), block_H, block_N, streaming_info[0].cpu().tolist())
        num_splits = 2
        num_stages = 1
        threads = 256

        glse = torch.empty(batch_size, seqlen_q, num_heads_q, num_splits, dtype=q.dtype, device=q.device)
        out_partial = torch.empty(batch_size, seqlen_q, num_heads_q, num_splits, dim_nope, dtype=q.dtype, device=q.device)

        kernel = zigzag_attn_with_kvcache_kernel(
            batch_size, seqlen_q, num_heads_q, num_heads_kv, dim_nope, head_dim_qk - dim_nope, softmax_scale, num_pages, page_size, block_H, block_N, num_splits, num_stages, threads
        )
        return kernel(q, kv, block_table, cache_seqlens, glse, out_partial, streaming_info, head_mask_type)
    else:
        block_H = 64
        block_N = 64
        # num_splits = get_splits(batch_size, num_heads_q, seqlen_q, torch.mean(cache_seqlens.float()).int().item(), block_H, block_N, None)
        num_splits = 2

        glse = torch.empty(batch_size, seqlen_q, num_heads_q, num_splits, dtype=q.dtype, device=q.device)
        out_partial = torch.empty(batch_size, seqlen_q, num_heads_q, num_splits, dim_nope, dtype=q.dtype, device=q.device)

        kernel = flash_mla_with_kvcache_kernel(
            batch_size, seqlen_q, num_heads_q, num_heads_kv, dim_nope, head_dim_qk - dim_nope, block_N, block_H, num_splits, page_size, num_pages, block_table.shape[1], softmax_scale
        )
        return kernel(q, kv, block_table, cache_seqlens, glse, out_partial)
