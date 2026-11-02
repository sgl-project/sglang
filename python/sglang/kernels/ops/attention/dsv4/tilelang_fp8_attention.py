"""TileLang FP8 sparse attention for DeepSeek-V4 (``flash_mla_with_kvcache``
signature, MODEL1_FP8Sparse KV cache)."""

from typing import Any, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

# Importing tilelang_kernel also applies its out_idx adapter patch, which the
# single/dual partial kernel below relies on.
from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    BF16,
    FP32,
    INT32,
    _is_gfx95_supported,
    _pick_inner_iter,
)


def _build_fp8_combined_view(k_cache: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Reinterpret a MODEL1_FP8Sparse KV cache as a contiguous uint32 view.
    Input:  k_cache (num_blocks, block_size, 1, d_qk)  fp8/uint8
            — per-block storage also holds scales + padding past d_qk.
    Output: (num_blocks, block_pad_u32) uint32 covering the full block
            stride. Same storage ashe input, no copy.
    """
    k_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    num_blocks = k_u8.shape[0]
    block_size = k_u8.shape[1]
    block_pad_u32 = k_u8.stride(0) // 4
    storage = k_u8.untyped_storage()
    flat_u32 = torch.empty(0, dtype=torch.uint32, device=k_u8.device).set_(
        storage, 0, (storage.nbytes() // 4,), (1,)
    )
    k_combined = torch.as_strided(
        flat_u32,
        size=(num_blocks, block_pad_u32),
        stride=(block_pad_u32, 1),
        storage_offset=k_u8.storage_offset() // 4,
    )
    return k_combined, num_blocks, block_size


_TOPK_LEN_SENTINEL_CACHE: dict = {}
_INT32_MAX = 2**30


def _topk_length_sentinel(device: torch.device, batch: int) -> torch.Tensor:
    """Cached `(batch,) int32 INT_MAX` tensor used when `topk_length` is None."""
    cur = _TOPK_LEN_SENTINEL_CACHE.get(device)
    if cur is None or cur.numel() < batch:
        cur = torch.full(
            (max(batch, 256),), _INT32_MAX, dtype=torch.int32, device=device
        )
        _TOPK_LEN_SENTINEL_CACHE[device] = cur
    return cur[:batch]


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def dpsk_v4_fp8_partial_kernel(
    num_heads: int,
    topk_1: int,
    block_size_kv_1: int,
    topk_2: int = 0,
    block_size_kv_2: int = 0,
    *,
    dim: int = 448,
    tail_dim: int = 64,
    sm_scale: float = 0.0,
    block_I: int = 64,
    inner_iter_1: int = 1,
    inner_iter_2: int = 0,
    num_stages: int = 0,
    threads: int = 512,
) -> Any:
    """
    Read FP8 K cache directly, dequantise to BF16 in-kernel, do flash-attn
    online softmax with split-K. Supports a second cache (`topk_2>0`) and
    `attn_sink` is folded later by the combine kernel.
    """
    log2e: float = 1.44269504
    if sm_scale <= 0.0:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * log2e
    else:
        sm_scale = sm_scale * log2e
    assert dim == 448 and tail_dim == 64
    assert topk_1 % block_I == 0
    assert (topk_1 // block_I) % inner_iter_1 == 0, (
        f"NI_1={topk_1 // block_I} must be divisible by inner_iter_1={inner_iter_1}"
    )
    assert block_size_kv_1 > 0 and (block_size_kv_1 & (block_size_kv_1 - 1)) == 0

    is_dual = topk_2 > 0
    if is_dual:
        assert inner_iter_2 > 0, "dual-cache call requires inner_iter_2 > 0"
        assert topk_2 % block_I == 0
        assert (topk_2 // block_I) % inner_iter_2 == 0, (
            f"NI_2={topk_2 // block_I} must be divisible by inner_iter_2={inner_iter_2}"
        )
        assert block_size_kv_2 > 0 and (block_size_kv_2 & (block_size_kv_2 - 1)) == 0

    PACKED_W = dim + 2 * tail_dim
    NOPE_TILE = 64
    NUM_TILES = dim // NOPE_TILE
    SCALE_W = 8
    PACKED_W4 = PACKED_W // 4
    SCALE_W4 = SCALE_W // 4

    kv_group = 1
    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    num_blocks_kv_1 = T.symbolic("num_blocks_kv_1")
    block_pad_u32_1 = T.symbolic("block_pad_u32_1")
    if is_dual:
        num_blocks_kv_2 = T.symbolic("num_blocks_kv_2")
        block_pad_u32_2 = T.symbolic("block_pad_u32_2")

    head_kv = num_heads // kv_group
    D = dim
    D_tail = tail_dim
    BI = block_I
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if head_kv > 64:
        assert head_kv % 64 == 0
    REPLICATE_H = (head_kv + 63) // 64 if head_kv > 64 else 1
    H_per_block = 64 if REPLICATE_H > 1 else padded_H

    NI_1 = topk_1 // BI
    n_groups_1 = NI_1 // inner_iter_1
    NI_2 = (topk_2 // BI) if is_dual else 0
    n_groups_2 = (NI_2 // inner_iter_2) if is_dual else 0
    n_groups = n_groups_1 + n_groups_2

    BS_KV_1 = block_size_kv_1
    NOPE_ROPE_U32_PER_BLOCK_1 = BS_KV_1 * PACKED_W4
    if is_dual:
        BS_KV_2 = block_size_kv_2
        NOPE_ROPE_U32_PER_BLOCK_2 = BS_KV_2 * PACKED_W4

    q_shape = [batch, seq_len, num_heads, D + D_tail]
    k1_shape = [num_blocks_kv_1, block_pad_u32_1]
    indices1_shape = [batch, seq_len, topk_1]
    topk_length_shape = [batch]
    partial_o_shape = [batch, seq_len, n_groups, num_heads, D + D_tail]
    partial_lse_shape = [batch, seq_len, n_groups, num_heads]
    if is_dual:
        k2_shape = [num_blocks_kv_2, block_pad_u32_2]
        indices2_shape = [batch, seq_len, topk_2]

    accum_dtype = "float"
    indices_dtype = INT32

    if is_dual:

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, BF16),  # type: ignore
            K_combined_1: T.Tensor(k1_shape, "uint32"),  # type: ignore
            Indices_1: T.Tensor(indices1_shape, indices_dtype),  # type: ignore
            Topk_length_1: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
            K_combined_2: T.Tensor(k2_shape, "uint32"),  # type: ignore
            Indices_2: T.Tensor(indices2_shape, indices_dtype),  # type: ignore
            Topk_length_2: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
            Partial_O: T.Tensor(partial_o_shape, BF16),  # type: ignore
            Partial_LSE: T.Tensor(partial_lse_shape, accum_dtype),  # type: ignore
        ) -> None:
            """
            grid: (seq_len * REPLICATE_H * n_groups, batch, 1)
            Each block processes `inner_iter_1` (or `inner_iter_2`) consecutive
            KV tiles of one phase and writes one (partial_o, partial_lse) entry.
            """
            with T.Kernel(
                seq_len * REPLICATE_H * n_groups, batch, kv_group, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_fragment([H_per_block, D], BF16)
                Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], BF16)
                K_packed_shared = T.alloc_shared([BI, PACKED_W4], "uint32")
                K_scale_shared = T.alloc_shared([BI, SCALE_W4], "uint32")
                KV_shared = T.alloc_shared([BI, D], BF16)
                K_tail_shared = T.alloc_shared([BI, D_tail], BF16)
                S_shared = T.alloc_shared([H_per_block, BI], BF16)
                page_idx_shared = T.alloc_shared([BI], INT32)

                mask = T.alloc_fragment([BI], "bool")
                scale_byte_local = T.alloc_fragment([BI, NUM_TILES], "uint32")

                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_o_tail = T.alloc_fragment([H_per_block, D_tail], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(acc_o_tail, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_i, g_i = by, bz
                # bx encodes (s_i, h_replicate, group_i).
                spans_per_seq = REPLICATE_H * n_groups
                s_i = bx // spans_per_seq
                rest = bx % spans_per_seq
                group_i = rest // REPLICATE_H
                h_rep = rest % REPLICATE_H
                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else h_rep * 64)
                H1 = H0 + H_per_block

                tk_len_1 = Topk_length_1[b_i]
                tk_len_2 = Topk_length_2[b_i]
                actual_n_groups_1 = T.ceildiv(tk_len_1, BI * inner_iter_1)
                actual_n_groups_2 = T.ceildiv(tk_len_2, BI * inner_iter_2)

                if (group_i < n_groups_1) & (group_i < actual_n_groups_1):
                    # Phase 1 active: SWA cache work + Partial_O write.
                    T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                    T.copy(Q[b_i, s_i, H0:H1, D : D + D_tail], Q_tail_shared)
                    for k_i in T.Pipelined(inner_iter_1, num_stages=num_stages):
                        iter_i = group_i * inner_iter_1 + k_i
                        for bi_i in T.Parallel(BI):
                            pos = iter_i * BI + bi_i
                            idx = Indices_1[b_i, s_i, pos]
                            valid = (idx >= 0) & (pos < tk_len_1)
                            page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                            mask[bi_i] = valid

                        for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_1
                            t_in_block = page % BS_KV_1
                            K_packed_shared[bi_i, w_i] = K_combined_1[
                                block_id, t_in_block * PACKED_W4 + w_i
                            ]

                        for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_1
                            t_in_block = page % BS_KV_1
                            K_scale_shared[bi_i, w_i] = K_combined_1[
                                block_id,
                                NOPE_ROPE_U32_PER_BLOCK_1 + t_in_block * SCALE_W4 + w_i,
                            ]

                        for bi_i, ti in T.Parallel(BI, NUM_TILES):
                            word_idx = ti // 4
                            byte_in_word = ti % 4
                            word = K_scale_shared[bi_i, word_idx]
                            scale_byte_local[bi_i, ti] = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)

                        for bi_i, d_i in T.Parallel(BI, D):
                            word_idx = d_i // 4
                            byte_in_word = d_i % 4
                            word = K_packed_shared[bi_i, word_idx]
                            b_u32 = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)
                            sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                            exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                            mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                            scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                            exp_combined = exp_e4 + scale_byte - T.uint32(7)
                            bf16_bits = (
                                sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                            )
                            KV_shared[bi_i, d_i] = T.reinterpret(
                                BF16, T.Cast("uint16", bf16_bits)
                            )

                        for bi_i, j in T.Parallel(BI, D_tail):
                            abs_off = D + 2 * j
                            word_idx = abs_off // 4
                            word_off = abs_off % 4
                            word = K_packed_shared[bi_i, word_idx]
                            half_u32 = T.if_then_else(
                                word_off == 0,
                                word & T.uint32(0xFFFF),
                                (word >> T.uint32(16)) & T.uint32(0xFFFF),
                            )
                            K_tail_shared[bi_i, j] = T.reinterpret(
                                BF16, T.Cast("uint16", half_u32)
                            )

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D):
                            acc_o[h_i, d_i] *= alpha[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D_tail):
                            acc_o_tail[h_i, d_i] *= alpha[h_i]
                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared,
                            KV_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            S_shared,
                            K_tail_shared,
                            acc_o_tail,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    # ---- finalize phase 1 (active) ----
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                            sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                        )
                    for h_i, d_i in T.Parallel(H_per_block, D_tail):
                        acc_o_tail[h_i, d_i] = acc_o_tail[h_i, d_i] / T.if_then_else(
                            sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                        )
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.if_then_else(
                            sumexp[h_i] == 0.0,
                            -(2.0**30),
                            T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                        )
                    T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :D])
                    T.copy(
                        acc_o_tail,
                        Partial_O[b_i, s_i, group_i, H0:H1, D : D + D_tail],
                    )
                    T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])
                elif group_i < n_groups_1:
                    # Phase 1 skipped: m_i is still the -2^30
                    T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])
                elif (group_i - n_groups_1) < actual_n_groups_2:
                    # Phase 2 active: c128 cache work + Partial_O write.
                    T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                    T.copy(Q[b_i, s_i, H0:H1, D : D + D_tail], Q_tail_shared)
                    for k_i in T.Pipelined(inner_iter_2, num_stages=num_stages):
                        iter_i = (group_i - n_groups_1) * inner_iter_2 + k_i
                        for bi_i in T.Parallel(BI):
                            pos = iter_i * BI + bi_i
                            idx = Indices_2[b_i, s_i, pos]
                            valid = (idx >= 0) & (pos < tk_len_2)
                            page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                            mask[bi_i] = valid

                        for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_2
                            t_in_block = page % BS_KV_2
                            K_packed_shared[bi_i, w_i] = K_combined_2[
                                block_id, t_in_block * PACKED_W4 + w_i
                            ]

                        for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                            page = page_idx_shared[bi_i]
                            block_id = page // BS_KV_2
                            t_in_block = page % BS_KV_2
                            K_scale_shared[bi_i, w_i] = K_combined_2[
                                block_id,
                                NOPE_ROPE_U32_PER_BLOCK_2 + t_in_block * SCALE_W4 + w_i,
                            ]

                        for bi_i, ti in T.Parallel(BI, NUM_TILES):
                            word_idx = ti // 4
                            byte_in_word = ti % 4
                            word = K_scale_shared[bi_i, word_idx]
                            scale_byte_local[bi_i, ti] = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)

                        for bi_i, d_i in T.Parallel(BI, D):
                            word_idx = d_i // 4
                            byte_in_word = d_i % 4
                            word = K_packed_shared[bi_i, word_idx]
                            b_u32 = (
                                word >> T.Cast("uint32", byte_in_word * 8)
                            ) & T.uint32(0xFF)
                            sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                            exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                            mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                            scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                            exp_combined = exp_e4 + scale_byte - T.uint32(7)
                            bf16_bits = (
                                sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                            )
                            KV_shared[bi_i, d_i] = T.reinterpret(
                                BF16, T.Cast("uint16", bf16_bits)
                            )

                        for bi_i, j in T.Parallel(BI, D_tail):
                            abs_off = D + 2 * j
                            word_idx = abs_off // 4
                            word_off = abs_off % 4
                            word = K_packed_shared[bi_i, word_idx]
                            half_u32 = T.if_then_else(
                                word_off == 0,
                                word & T.uint32(0xFFFF),
                                (word >> T.uint32(16)) & T.uint32(0xFFFF),
                            )
                            K_tail_shared[bi_i, j] = T.reinterpret(
                                BF16, T.Cast("uint16", half_u32)
                            )

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D):
                            acc_o[h_i, d_i] *= alpha[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D_tail):
                            acc_o_tail[h_i, d_i] *= alpha[h_i]
                        T.copy(acc_s, S_shared)
                        T.gemm(
                            S_shared,
                            KV_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            S_shared,
                            K_tail_shared,
                            acc_o_tail,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    # ---- finalize phase 2 (active) ----
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                            sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                        )
                    for h_i, d_i in T.Parallel(H_per_block, D_tail):
                        acc_o_tail[h_i, d_i] = acc_o_tail[h_i, d_i] / T.if_then_else(
                            sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                        )
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.if_then_else(
                            sumexp[h_i] == 0.0,
                            -(2.0**30),
                            T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                        )
                    T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :D])
                    T.copy(
                        acc_o_tail,
                        Partial_O[b_i, s_i, group_i, H0:H1, D : D + D_tail],
                    )
                    T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])
                else:
                    # Phase 2 skipped: m_i is still the -2^30
                    T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])

        return main

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, BF16),  # type: ignore
        K_combined_1: T.Tensor(k1_shape, "uint32"),  # type: ignore
        Indices_1: T.Tensor(indices1_shape, indices_dtype),  # type: ignore
        Topk_length_1: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
        Partial_O: T.Tensor(partial_o_shape, BF16),  # type: ignore
        Partial_LSE: T.Tensor(partial_lse_shape, accum_dtype),  # type: ignore
    ) -> None:
        """
        grid: (seq_len * REPLICATE_H * n_groups, batch, 1)
        Each block processes `inner_iter_1` consecutive KV tiles and writes
        one (partial_o, partial_lse) entry.
        """
        with T.Kernel(
            seq_len * REPLICATE_H * n_groups, batch, kv_group, threads=threads
        ) as (bx, by, bz):
            Q_shared = T.alloc_fragment([H_per_block, D], BF16)
            Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], BF16)
            K_packed_shared = T.alloc_shared([BI, PACKED_W4], "uint32")
            K_scale_shared = T.alloc_shared([BI, SCALE_W4], "uint32")
            KV_shared = T.alloc_shared([BI, D], BF16)
            K_tail_shared = T.alloc_shared([BI, D_tail], BF16)
            S_shared = T.alloc_shared([H_per_block, BI], BF16)
            page_idx_shared = T.alloc_shared([BI], INT32)

            mask = T.alloc_fragment([BI], "bool")
            scale_byte_local = T.alloc_fragment([BI, NUM_TILES], "uint32")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_o_tail = T.alloc_fragment([H_per_block, D_tail], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(acc_o_tail, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            spans_per_seq = REPLICATE_H * n_groups
            s_i = bx // spans_per_seq
            rest = bx % spans_per_seq
            group_i = rest // REPLICATE_H
            h_rep = rest % REPLICATE_H
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else h_rep * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D : D + D_tail], Q_tail_shared)

            tk_len_1 = Topk_length_1[b_i]

            for k_i in T.Pipelined(inner_iter_1, num_stages=num_stages):
                iter_i = group_i * inner_iter_1 + k_i
                for bi_i in T.Parallel(BI):
                    pos = iter_i * BI + bi_i
                    idx = Indices_1[b_i, s_i, pos]
                    valid = (idx >= 0) & (pos < tk_len_1)
                    page_idx_shared[bi_i] = T.if_then_else(valid, idx, 0)
                    mask[bi_i] = valid

                for bi_i, w_i in T.Parallel(BI, PACKED_W4):
                    page = page_idx_shared[bi_i]
                    block_id = page // BS_KV_1
                    t_in_block = page % BS_KV_1
                    K_packed_shared[bi_i, w_i] = K_combined_1[
                        block_id, t_in_block * PACKED_W4 + w_i
                    ]

                for bi_i, w_i in T.Parallel(BI, SCALE_W4):
                    page = page_idx_shared[bi_i]
                    block_id = page // BS_KV_1
                    t_in_block = page % BS_KV_1
                    K_scale_shared[bi_i, w_i] = K_combined_1[
                        block_id,
                        NOPE_ROPE_U32_PER_BLOCK_1 + t_in_block * SCALE_W4 + w_i,
                    ]

                for bi_i, ti in T.Parallel(BI, NUM_TILES):
                    word_idx = ti // 4
                    byte_in_word = ti % 4
                    word = K_scale_shared[bi_i, word_idx]
                    scale_byte_local[bi_i, ti] = (
                        word >> T.Cast("uint32", byte_in_word * 8)
                    ) & T.uint32(0xFF)

                for bi_i, d_i in T.Parallel(BI, D):
                    word_idx = d_i // 4
                    byte_in_word = d_i % 4
                    word = K_packed_shared[bi_i, word_idx]
                    b_u32 = (word >> T.Cast("uint32", byte_in_word * 8)) & T.uint32(
                        0xFF
                    )
                    sign_bf = (b_u32 & T.uint32(0x80)) * T.uint32(0x100)
                    exp_e4 = (b_u32 & T.uint32(0x78)) >> T.uint32(3)
                    mant_bf = (b_u32 & T.uint32(0x7)) * T.uint32(0x10)
                    scale_byte = scale_byte_local[bi_i, d_i // NOPE_TILE]
                    exp_combined = exp_e4 + scale_byte - T.uint32(7)
                    bf16_bits = sign_bf | (exp_combined << T.uint32(7)) | mant_bf
                    KV_shared[bi_i, d_i] = T.reinterpret(
                        BF16, T.Cast("uint16", bf16_bits)
                    )

                for bi_i, j in T.Parallel(BI, D_tail):
                    abs_off = D + 2 * j
                    word_idx = abs_off // 4
                    word_off = abs_off % 4
                    word = K_packed_shared[bi_i, word_idx]
                    half_u32 = T.if_then_else(
                        word_off == 0,
                        word & T.uint32(0xFFFF),
                        (word >> T.uint32(16)) & T.uint32(0xFFFF),
                    )
                    K_tail_shared[bi_i, j] = T.reinterpret(
                        BF16, T.Cast("uint16", half_u32)
                    )

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] *= alpha[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D_tail):
                    acc_o_tail[h_i, d_i] *= alpha[h_i]
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(
                    S_shared, K_tail_shared, acc_o_tail, policy=T.GemmWarpPolicy.FullRow
                )

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            for h_i, d_i in T.Parallel(H_per_block, D_tail):
                acc_o_tail[h_i, d_i] = acc_o_tail[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2.0**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )
            T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :D])
            T.copy(acc_o_tail, Partial_O[b_i, s_i, group_i, H0:H1, D : D + D_tail])
            T.copy(m_i, Partial_LSE[b_i, s_i, group_i, H0:H1])

    return main


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def dpsk_v4_combine_kernel(
    num_heads: int,
    n_groups_1: int,
    n_groups_2: int = 0,
    *,
    block_I: int = 64,
    inner_iter_1: int = 1,
    inner_iter_2: int = 1,
    dim: int = 448,
    tail_dim: int = 64,
    head_per_block: int = 16,
    threads: int = 256,
    use_attn_sink: bool = False,
) -> Any:
    """
    Combine `n_groups` flash-attention partials into the final output.

    Inputs:
      Partial_O    : (batch, seq_len, n_groups, num_heads, dim+tail_dim) bf16
      Partial_LSE  : (batch, seq_len, n_groups, num_heads) fp32, log2 form
      Topk_length_1: (batch,) int32, actual phase-1 length
      Topk_length_2: (batch,) int32, actual phase-2 length (dual only)
      Attn_sink    : (num_heads,) fp32
    Outputs:
      Output : (batch, seq_len, num_heads, dim+tail_dim) bf16
      LSE    : (batch, seq_len, num_heads) fp32, natural log

    Each grid block handles `head_per_block` heads of one (batch, seq) row.
    """
    log2e: float = 1.44269504
    ln2: float = 0.69314718
    assert num_heads % head_per_block == 0

    is_dual = n_groups_2 > 0
    n_groups = n_groups_1 + n_groups_2

    H_per_block = head_per_block
    HEAD_BLOCKS = num_heads // H_per_block
    DT = dim + tail_dim

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")

    accum_dtype = "float"

    if is_dual:

        @T.prim_func
        def main(
            Partial_O: T.Tensor([batch, seq_len, n_groups, num_heads, DT], BF16),  # type: ignore
            Partial_LSE: T.Tensor([batch, seq_len, n_groups, num_heads], accum_dtype),  # type: ignore
            Topk_length_1: T.Tensor([batch], INT32),  # type: ignore
            Topk_length_2: T.Tensor([batch], INT32),  # type: ignore
            Attn_sink: T.Tensor([num_heads], FP32),  # type: ignore
            Output: T.Tensor([batch, seq_len, num_heads, DT], BF16),  # type: ignore
            LSE: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(seq_len * HEAD_BLOCKS, batch, threads=threads) as (
                bx,
                by,
            ):
                shared_lse = T.alloc_shared([n_groups, H_per_block], accum_dtype)
                lse_max = T.alloc_fragment([H_per_block], accum_dtype)
                lse_sum = T.alloc_fragment([H_per_block], accum_dtype)
                scale = T.alloc_fragment([H_per_block, n_groups], accum_dtype)
                acc_o = T.alloc_fragment([H_per_block, DT], accum_dtype)
                attn_sink_frag = T.alloc_fragment([H_per_block], accum_dtype)
                o_scale_frag = T.alloc_fragment([H_per_block], accum_dtype)
                final_lse = T.alloc_fragment([H_per_block], accum_dtype)

                b_i = by
                s_i = bx // HEAD_BLOCKS
                head_block = bx % HEAD_BLOCKS
                H0 = head_block * H_per_block
                H1 = H0 + H_per_block

                # Clamp to the captured-shape upper bounds so callers passing
                # the INT32_MAX sentinel (= "all valid") still iterate exactly
                # n_groups groups, not 33M.
                actual_n_groups_1 = T.min(
                    T.ceildiv(Topk_length_1[b_i], block_I * inner_iter_1),
                    n_groups_1,
                )
                actual_n_groups_2 = T.min(
                    T.ceildiv(Topk_length_2[b_i], block_I * inner_iter_2),
                    n_groups - n_groups_1,
                )
                actual_n_groups = actual_n_groups_1 + actual_n_groups_2

                # Pass 1: load only active groups' LSE into compact slots.
                for k_c in T.serial(actual_n_groups):
                    k = T.if_then_else(
                        k_c < actual_n_groups_1,
                        k_c,
                        n_groups_1 + (k_c - actual_n_groups_1),
                    )
                    T.copy(Partial_LSE[b_i, s_i, k, H0:H1], shared_lse[k_c, :])

                T.fill(lse_max, -(2**30))
                for k_c in T.serial(actual_n_groups):
                    for h_i in T.Parallel(H_per_block):
                        lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k_c, h_i])
                T.fill(lse_sum, 0)
                for k_c in T.serial(actual_n_groups):
                    for h_i in T.Parallel(H_per_block):
                        lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                            shared_lse[k_c, h_i] - lse_max[h_i]
                        )
                for k_c in T.serial(actual_n_groups):
                    for h_i in T.Parallel(H_per_block):
                        scale[h_i, k_c] = T.exp2(
                            shared_lse[k_c, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                        )

                T.fill(acc_o, 0)
                for k_c in T.serial(actual_n_groups):
                    k = T.if_then_else(
                        k_c < actual_n_groups_1,
                        k_c,
                        n_groups_1 + (k_c - actual_n_groups_1),
                    )
                    for h_i, d_i in T.Parallel(H_per_block, DT):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k_c] * Partial_O[
                            b_i, s_i, k, H0 + h_i, d_i
                        ].astype(accum_dtype)

                for h_i in T.Parallel(H_per_block):
                    empty = lse_max[h_i] <= -(2**29)
                    final_lse[h_i] = T.if_then_else(
                        empty,
                        T.infinity(accum_dtype),
                        (lse_max[h_i] + T.log2(lse_sum[h_i])) * ln2,
                    )

                if use_attn_sink:
                    for h_i in T.Parallel(H_per_block):
                        attn_sink_frag[h_i] = Attn_sink[H0 + h_i]
                    for h_i in T.Parallel(H_per_block):
                        empty = lse_max[h_i] <= -(2**29)
                        o_scale_frag[h_i] = T.if_then_else(
                            empty,
                            0.0,
                            1.0
                            / (
                                1.0
                                + T.exp2((attn_sink_frag[h_i] - final_lse[h_i]) * log2e)
                            ),
                        )
                    for h_i, d_i in T.Parallel(H_per_block, DT):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * o_scale_frag[h_i]

                T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
                T.copy(final_lse, LSE[b_i, s_i, H0:H1])

        return main

    @T.prim_func
    def main(
        Partial_O: T.Tensor([batch, seq_len, n_groups, num_heads, DT], BF16),  # type: ignore
        Partial_LSE: T.Tensor([batch, seq_len, n_groups, num_heads], accum_dtype),  # type: ignore
        Attn_sink: T.Tensor([num_heads], FP32),  # type: ignore
        Output: T.Tensor([batch, seq_len, num_heads, DT], BF16),  # type: ignore
        LSE: T.Tensor([batch, seq_len, num_heads], accum_dtype),  # type: ignore
    ) -> None:
        with T.Kernel(seq_len * HEAD_BLOCKS, batch, threads=threads) as (bx, by):
            shared_lse = T.alloc_shared([n_groups, H_per_block], accum_dtype)

            lse_max = T.alloc_fragment([H_per_block], accum_dtype)
            lse_sum = T.alloc_fragment([H_per_block], accum_dtype)
            scale = T.alloc_fragment([H_per_block, n_groups], accum_dtype)
            acc_o = T.alloc_fragment([H_per_block, DT], accum_dtype)
            attn_sink_frag = T.alloc_fragment([H_per_block], accum_dtype)
            o_scale_frag = T.alloc_fragment([H_per_block], accum_dtype)
            final_lse = T.alloc_fragment([H_per_block], accum_dtype)

            b_i = by
            s_i = bx // HEAD_BLOCKS
            head_block = bx % HEAD_BLOCKS
            H0 = head_block * H_per_block
            H1 = H0 + H_per_block

            for k in T.serial(n_groups):
                T.copy(Partial_LSE[b_i, s_i, k, H0:H1], shared_lse[k, :])

            T.fill(lse_max, -(2**30))
            for k in T.serial(n_groups):
                for h_i in T.Parallel(H_per_block):
                    lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k, h_i])
            T.fill(lse_sum, 0)
            for k in T.serial(n_groups):
                for h_i in T.Parallel(H_per_block):
                    lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i]
                    )
            for k in T.serial(n_groups):
                for h_i in T.Parallel(H_per_block):
                    scale[h_i, k] = T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                    )

            T.fill(acc_o, 0)
            for k in T.serial(n_groups):
                for h_i, d_i in T.Parallel(H_per_block, DT):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k] * Partial_O[
                        b_i, s_i, k, H0 + h_i, d_i
                    ].astype(accum_dtype)

            for h_i in T.Parallel(H_per_block):
                empty = lse_max[h_i] <= -(2**29)
                final_lse[h_i] = T.if_then_else(
                    empty,
                    T.infinity(accum_dtype),
                    (lse_max[h_i] + T.log2(lse_sum[h_i])) * ln2,
                )

            if use_attn_sink:
                for h_i in T.Parallel(H_per_block):
                    attn_sink_frag[h_i] = Attn_sink[H0 + h_i]
                for h_i in T.Parallel(H_per_block):
                    empty = lse_max[h_i] <= -(2**29)
                    o_scale_frag[h_i] = T.if_then_else(
                        empty,
                        0.0,
                        1.0
                        / (
                            1.0 + T.exp2((attn_sink_frag[h_i] - final_lse[h_i]) * log2e)
                        ),
                    )
                for h_i, d_i in T.Parallel(H_per_block, DT):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * o_scale_frag[h_i]

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(final_lse, LSE[b_i, s_i, H0:H1])

    return main


"""
2-stage attention kernel (partial + combine) over an FP8 KV cache,
with optional second cache (`extra_k_cache`).
"""


def dpsk_v4_fp8_attention_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Follows the original `flash_mla.flash_mla_with_kvcache` signature.
    """
    if _is_gfx95_supported:
        block_I, threads, num_stages, block_per_cu, cu = 64, 512, 0, 2, 256
    else:
        block_I, threads, num_stages, block_per_cu, cu = 32, 128, 1, 1, 304

    batch, seq_len, num_heads, _ = q.shape
    # Partial grid is (seq_len * REPLICATE_H * n_groups, batch, kv_group); the
    # heuristic in _pick_inner_iter assumes `total_blocks = seq * ni / inner_iter`,
    # so `seq` must include REPLICATE_H or n_groups doubles for medium batches.
    replicate_h = max((num_heads + 63) // 64, 1)
    seq = batch * seq_len * replicate_h

    k1, _, bs_kv_1 = _build_fp8_combined_view(k_cache)
    topk_1 = indices.shape[-1]
    ni_1 = topk_1 // block_I
    tk_len_1 = (
        topk_length
        if topk_length is not None
        else _topk_length_sentinel(q.device, batch)
    )
    if attn_sink is None:
        attn_sink = torch.full(
            (num_heads,), float("-inf"), dtype=torch.float32, device=q.device
        )

    has_extra = extra_k_cache is not None
    if not has_extra:
        inner_iter_1 = _pick_inner_iter(seq, ni_1, cu, block_per_cu)
        inner_iter_2 = 1
        n_groups_1 = ni_1 // inner_iter_1
        n_groups_2 = 0
        partial = dpsk_v4_fp8_partial_kernel(
            num_heads,
            topk_1,
            bs_kv_1,
            sm_scale=softmax_scale,
            block_I=block_I,
            inner_iter_1=inner_iter_1,
            num_stages=num_stages,
            threads=threads,
        )
        partial_o, partial_lse = partial(q, k1, indices, tk_len_1)
    else:
        k2, _, bs_kv_2 = _build_fp8_combined_view(extra_k_cache)
        topk_2 = extra_indices_in_kvcache.shape[-1]
        ni_2 = topk_2 // block_I
        # Each phase picks its own optimal split-K independently — kernel
        # body uses two T.Pipelined loops with separate compile-time iter
        # counts, no shared-divisor constraint.
        inner_iter_1 = _pick_inner_iter(seq, ni_1, cu, block_per_cu)
        inner_iter_2 = _pick_inner_iter(seq, ni_2, cu, block_per_cu)
        n_groups_1 = ni_1 // inner_iter_1
        n_groups_2 = ni_2 // inner_iter_2
        tk_len_2 = (
            extra_topk_length
            if extra_topk_length is not None
            else _topk_length_sentinel(q.device, batch)
        )
        partial = dpsk_v4_fp8_partial_kernel(
            num_heads,
            topk_1,
            bs_kv_1,
            topk_2,
            bs_kv_2,
            sm_scale=softmax_scale,
            block_I=block_I,
            inner_iter_1=inner_iter_1,
            inner_iter_2=inner_iter_2,
            num_stages=num_stages,
            threads=threads,
        )
        partial_o, partial_lse = partial(
            q,
            k1,
            indices,
            tk_len_1,
            k2,
            extra_indices_in_kvcache,
            tk_len_2,
        )

    combine = dpsk_v4_combine_kernel(
        num_heads,
        n_groups_1,
        n_groups_2,
        block_I=block_I,
        inner_iter_1=inner_iter_1,
        inner_iter_2=inner_iter_2,
        head_per_block=4,
        threads=256,
        use_attn_sink=True,
    )
    if has_extra:
        return combine(partial_o, partial_lse, tk_len_1, tk_len_2, attn_sink)
    return combine(partial_o, partial_lse, attn_sink)
