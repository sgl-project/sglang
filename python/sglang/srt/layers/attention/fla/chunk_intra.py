# Adapted from flash-linear-attention project.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.chunk_intra_token_parallel import (
    chunk_kda_fwd_intra_token_parallel,
)
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.srt.layers.attention.fla.op import exp, exp2, gather
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
    is_gather_supported,
    is_tf32_supported,
)

if is_tf32_supported:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": 64}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC", "V", "FUSE_RECOMPUTE", "FUSE_DIAGONAL"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    v_in,
    w_out,
    u_out,
    kg_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
    FUSE_RECOMPUTE: tl.constexpr,
    FUSE_DIAGONAL: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    if FUSE_DIAGONAL:
        b_Aqk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d0 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d1 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d2 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Aqk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        b_Akk_d3 = tl.zeros([BC, BC], dtype=tl.float32)
        m_tc0 = (i_tc0 + o_i) < T

    ################################################################################
    # off-diagonal blocks (+ optional diagonal blocks)
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        p_g0 = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if FUSE_DIAGONAL:
            p_q0 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            b_q0 = tl.load(p_q0, boundary_check=(0, 1)).to(tl.float32)
            b_gn0 = tl.load(g + i_tc0 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gm0 = tl.clamp(b_g0 - b_gn0[None, :], -126.0, 126.0)
            b_gq0 = tl.where(m_tc0[:, None], exp2(b_gm0), 0.0)
            b_gk0 = tl.where(m_tc0[:, None], exp2(-b_gm0), 0.0)
            b_kgt_d0 = tl.trans(b_k0 * b_gk0)
            b_Aqk_d0 += tl.dot(b_q0 * b_gq0, b_kgt_d0)
            b_Akk_d0 += tl.dot(b_k0 * b_gq0, b_kgt_d0)

        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_g1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0)).to(tl.bfloat16)
            # [BC, BC]
            b_qg1 = (b_q1 * b_gqn).to(tl.bfloat16)
            b_kg1 = (b_k1 * b_gqn).to(tl.bfloat16)
            b_Aqk10 += tl.dot(b_qg1, b_kgt)
            b_Akk10 += tl.dot(b_kg1, b_kgt)

            if FUSE_DIAGONAL:
                b_gm1_d = tl.clamp(b_gn1[None, :] - b_g1, -126.0, 126.0)
                b_gk1_d = tl.where(m_tc1[:, None], exp2(b_gm1_d), 0.0)
                b_kgt_d1 = tl.trans(b_k1 * b_gk1_d)
                b_Aqk_d1 += tl.dot(b_q1 * b_gqn, b_kgt_d1)
                b_Akk_d1 += tl.dot(b_k1 * b_gqn, b_kgt_d1)

            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(
                    q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                p_g2 = tl.make_block_ptr(
                    g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                # [BC, BK]
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                # [BK]
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                # [BC, BK]
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = (b_q2 * b_gqn2).to(tl.bfloat16)
                b_kg2 = (b_k2 * b_gqn2).to(tl.bfloat16)
                # [BK, BC]
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0)).to(tl.bfloat16)
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                # [BC, BC]
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1)).to(tl.bfloat16)
                # [BC, BC]
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if FUSE_DIAGONAL:
                    b_gm2_d = tl.clamp(b_gn2[None, :] - b_g2, -126.0, 126.0)
                    b_gk2_d = tl.where(m_tc2[:, None], exp2(b_gm2_d), 0.0)
                    b_kgt_d2 = tl.trans(b_k2 * b_gk2_d)
                    b_Aqk_d2 += tl.dot(b_q2 * b_gqn2, b_kgt_d2)
                    b_Akk_d2 += tl.dot(b_k2 * b_gqn2, b_kgt_d2)

                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(
                        q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    p_g3 = tl.make_block_ptr(
                        g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    # [BC, BK]
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    # [BK]
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    # [BC, BK]
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = (b_q3 * b_gqn3).to(tl.bfloat16)
                    b_kg3 = (b_k3 * b_gqn3).to(tl.bfloat16)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    # [BK, BC]
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2)).to(tl.bfloat16)
                    # [BC, BC]
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

                    if FUSE_DIAGONAL:
                        b_gm3_d = tl.clamp(b_gn3[None, :] - b_g3, -126.0, 126.0)
                        b_gk3_d = tl.where(m_tc3[:, None], exp2(b_gm3_d), 0.0)
                        b_kgt_d3 = tl.trans(b_k3 * b_gk3_d)
                        b_Aqk_d3 += tl.dot(b_q3 * b_gqn3, b_kgt_d3)
                        b_Akk_d3 += tl.dot(b_k3 * b_gqn3, b_kgt_d3)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Aqk21 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Aqk31 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Aqk32 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_b3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    if FUSE_DIAGONAL:
        m_Aqk_diag = o_i[:, None] >= o_i[None, :]
        m_Akk_diag = o_i[:, None] > o_i[None, :]

        b_Aqk_d0 = tl.where(m_Aqk_diag, b_Aqk_d0, 0.0)
        b_Akk_d0 = tl.where(m_Akk_diag, b_Akk_d0, 0.0)
        b_Aqk_d1 = tl.where(m_Aqk_diag, b_Aqk_d1, 0.0)
        b_Akk_d1 = tl.where(m_Akk_diag, b_Akk_d1, 0.0)
        b_Aqk_d2 = tl.where(m_Aqk_diag, b_Aqk_d2, 0.0)
        b_Akk_d2 = tl.where(m_Akk_diag, b_Akk_d2, 0.0)
        b_Aqk_d3 = tl.where(m_Aqk_diag, b_Aqk_d3, 0.0)
        b_Akk_d3 = tl.where(m_Akk_diag, b_Akk_d3, 0.0)

        p_Aqk_d0 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Aqk_d1 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
        )
        p_Aqk_d2 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
        )
        p_Aqk_d3 = tl.make_block_ptr(
            Aqk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
        )
        tl.store(
            p_Aqk_d0, (b_Aqk_d0 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d1, (b_Aqk_d1 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d2, (b_Aqk_d2 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(
            p_Aqk_d3, (b_Aqk_d3 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1)
        )

        p_bd0 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
        )
        p_bd1 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        p_bd2 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        p_bd3 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_bd0 = tl.load(p_bd0, boundary_check=(0,)).to(tl.float32)
        b_bd1 = tl.load(p_bd1, boundary_check=(0,)).to(tl.float32)
        b_bd2 = tl.load(p_bd2, boundary_check=(0,)).to(tl.float32)
        b_bd3 = tl.load(p_bd3, boundary_check=(0,)).to(tl.float32)
        b_Akk_d0 = b_Akk_d0 * b_bd0[:, None]
        b_Akk_d1 = b_Akk_d1 * b_bd1[:, None]
        b_Akk_d2 = b_Akk_d2 * b_bd2[:, None]
        b_Akk_d3 = b_Akk_d3 * b_bd3[:, None]

        p_Akkd00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akkd11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akkd22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akkd33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        tl.store(p_Akkd00, b_Akk_d0.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd11, b_Akk_d1.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd22, b_Akk_d2.to(Akkd.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akkd33, b_Akk_d3.to(Akkd.dtype.element_ty), boundary_check=(0, 1))

        b_Ai00 = b_Akk_d0
        b_Ai11 = b_Akk_d1
        b_Ai22 = b_Akk_d2
        b_Ai33 = b_Akk_d3
    else:
        p_Akk00 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
        b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    # Diagonal blocks are RAW (need substitution) when:
    #   - FUSE_DIAGONAL=True: blocks were computed fresh above as gated k·k.
    #   - FUSE_DIAGONAL=False with USE_SAFE_GATE=False: token_parallel wrote raw.
    # They are pre-inverted only by the safe_gate diagonal kernel
    # (USE_SAFE_GATE=True, FUSE_DIAGONAL=False).
    ################################################################################

    if FUSE_DIAGONAL or not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ################################################################################
    # Output: store Akk_inv OR compute w, u, kg from registers
    ################################################################################

    if FUSE_RECOMPUTE:
        # Cast A-inverse sub-blocks to input dtype for dot products
        b_Ai00_h = b_Ai00.to(k.dtype.element_ty)
        b_Ai10_h = b_Ai10.to(k.dtype.element_ty)
        b_Ai11_h = b_Ai11.to(k.dtype.element_ty)
        b_Ai20_h = b_Ai20.to(k.dtype.element_ty)
        b_Ai21_h = b_Ai21.to(k.dtype.element_ty)
        b_Ai22_h = b_Ai22.to(k.dtype.element_ty)
        b_Ai30_h = b_Ai30.to(k.dtype.element_ty)
        b_Ai31_h = b_Ai31.to(k.dtype.element_ty)
        b_Ai32_h = b_Ai32.to(k.dtype.element_ty)
        b_Ai33_h = b_Ai33.to(k.dtype.element_ty)

        # Load beta for all 4 sub-chunks
        p_b0 = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,)
        )
        b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
        p_b1r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,)
        )
        b_b1r = tl.load(p_b1r, boundary_check=(0,)).to(tl.float32)
        p_b2r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,)
        )
        b_b2r = tl.load(p_b2r, boundary_check=(0,)).to(tl.float32)
        p_b3r = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,)
        )
        b_b3r = tl.load(p_b3r, boundary_check=(0,)).to(tl.float32)

        # ---- u = A_inv @ (v * beta) ----
        v_base = v_in + (bos * H + i_h) * V
        u_base = u_out + (bos * H + i_h) * V
        for i_v in range(tl.cdiv(V, BV)):
            p_v0 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
            )
            p_v1 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
            )
            p_v2 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
            )
            p_v3 = tl.make_block_ptr(
                v_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
            )

            b_v0 = tl.load(p_v0, boundary_check=(0, 1))
            b_v1 = tl.load(p_v1, boundary_check=(0, 1))
            b_v2 = tl.load(p_v2, boundary_check=(0, 1))
            b_v3 = tl.load(p_v3, boundary_check=(0, 1))

            b_vb0 = (b_v0 * b_b0[:, None]).to(b_v0.dtype)
            b_vb1 = (b_v1 * b_b1r[:, None]).to(b_v1.dtype)
            b_vb2 = (b_v2 * b_b2r[:, None]).to(b_v2.dtype)
            b_vb3 = (b_v3 * b_b3r[:, None]).to(b_v3.dtype)

            b_u0 = tl.dot(b_Ai00_h, b_vb0)
            b_u1 = tl.dot(b_Ai10_h, b_vb0) + tl.dot(b_Ai11_h, b_vb1)
            b_u2 = (
                tl.dot(b_Ai20_h, b_vb0)
                + tl.dot(b_Ai21_h, b_vb1)
                + tl.dot(b_Ai22_h, b_vb2)
            )
            b_u3 = (
                tl.dot(b_Ai30_h, b_vb0)
                + tl.dot(b_Ai31_h, b_vb1)
                + tl.dot(b_Ai32_h, b_vb2)
                + tl.dot(b_Ai33_h, b_vb3)
            )

            p_u0 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc0, i_v * BV), (BC, BV), (1, 0)
            )
            p_u1 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc1, i_v * BV), (BC, BV), (1, 0)
            )
            p_u2 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc2, i_v * BV), (BC, BV), (1, 0)
            )
            p_u3 = tl.make_block_ptr(
                u_base, (T, V), (H * V, 1), (i_tc3, i_v * BV), (BC, BV), (1, 0)
            )
            tl.store(p_u0, b_u0.to(p_u0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u1, b_u1.to(p_u1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u2, b_u2.to(p_u2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_u3, b_u3.to(p_u3.dtype.element_ty), boundary_check=(0, 1))

        # ---- w = A_inv @ (k * beta * exp(gk)), kg = k * exp(gn - gk) ----
        w_base = w_out + (bos * H + i_h) * K
        kg_base = kg_out + (bos * H + i_h) * K
        last_idx = min(i_t * BT + BT, T) - 1

        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(g + last_idx * H * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )

            p_k0 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_k1 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_k2 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_k3 = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )

            p_gk0 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk1 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk2 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_gk3 = tl.make_block_ptr(
                g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )

            b_k0r = tl.load(p_k0, boundary_check=(0, 1))
            b_k1r = tl.load(p_k1, boundary_check=(0, 1))
            b_k2r = tl.load(p_k2, boundary_check=(0, 1))
            b_k3r = tl.load(p_k3, boundary_check=(0, 1))

            b_gk0r = tl.load(p_gk0, boundary_check=(0, 1)).to(tl.float32)
            b_gk1r = tl.load(p_gk1, boundary_check=(0, 1)).to(tl.float32)
            b_gk2r = tl.load(p_gk2, boundary_check=(0, 1)).to(tl.float32)
            b_gk3r = tl.load(p_gk3, boundary_check=(0, 1)).to(tl.float32)

            b_kb0 = (b_k0r * b_b0[:, None] * exp(b_gk0r)).to(b_k0r.dtype)
            b_kb1 = (b_k1r * b_b1r[:, None] * exp(b_gk1r)).to(b_k1r.dtype)
            b_kb2 = (b_k2r * b_b2r[:, None] * exp(b_gk2r)).to(b_k2r.dtype)
            b_kb3 = (b_k3r * b_b3r[:, None] * exp(b_gk3r)).to(b_k3r.dtype)

            b_w0 = tl.dot(b_Ai00_h, b_kb0)
            b_w1 = tl.dot(b_Ai10_h, b_kb0) + tl.dot(b_Ai11_h, b_kb1)
            b_w2 = (
                tl.dot(b_Ai20_h, b_kb0)
                + tl.dot(b_Ai21_h, b_kb1)
                + tl.dot(b_Ai22_h, b_kb2)
            )
            b_w3 = (
                tl.dot(b_Ai30_h, b_kb0)
                + tl.dot(b_Ai31_h, b_kb1)
                + tl.dot(b_Ai32_h, b_kb2)
                + tl.dot(b_Ai33_h, b_kb3)
            )

            p_w0 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_w1 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_w2 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_w3 = tl.make_block_ptr(
                w_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )
            tl.store(p_w0, b_w0.to(p_w0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_w3, b_w3.to(p_w3.dtype.element_ty), boundary_check=(0, 1))

            b_kg0 = b_k0r * exp(b_gn[None, :] - b_gk0r)
            b_kg1 = b_k1r * exp(b_gn[None, :] - b_gk1r)
            b_kg2 = b_k2r * exp(b_gn[None, :] - b_gk2r)
            b_kg3 = b_k3r * exp(b_gn[None, :] - b_gk3r)

            p_kg0 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg1 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg2 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
            )
            p_kg3 = tl.make_block_ptr(
                kg_base, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
            )
            tl.store(p_kg0, b_kg0.to(p_kg0.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg1, b_kg1.to(p_kg1.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg2, b_kg2.to(p_kg2.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_kg3, b_kg3.to(p_kg3.dtype.element_ty), boundary_check=(0, 1))
    else:
        p_Akk00 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0)
        )
        p_Akk10 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0)
        )
        p_Akk11 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0)
        )
        p_Akk20 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0)
        )
        p_Akk21 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0)
        )
        p_Akk22 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
        )
        p_Akk30 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0)
        )
        p_Akk31 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0)
        )
        p_Akk32 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
        )
        p_Akk33 = tl.make_block_ptr(
            Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
        )

        tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * H + i_h) * K
    beta = beta + bos * H + i_h
    Aqk = Aqk + (bos * H + i_h) * BT
    Akk = Akk + (bos * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0))

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    if USE_GATHER:
        b_gn = gather(
            b_g, tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16), axis=0
        )
    else:
        # calculate offset
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * H * K + tl.arange(0, BK)
        b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
        b_gn = b_gn[None, :]

    # current block, keep numerical stability by subtracting the left boundary
    # less than 85 to avoid overflow in exp2
    b_gm = (b_g - b_gn).to(tl.float32)

    b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

    b_kgt = tl.trans(b_k * b_gk)

    b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
    b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_ti, i_i * BC), (BC, BC), (1, 0)
    )
    p_Akk = tl.make_block_ptr(Akk, (T, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()

    ################################################################################
    # forward substitution
    ################################################################################

    b_Ai = -b_Akk
    for i in range(2, min(BC, T - i_ti)):
        b_a = -tl.load(Akk + (i_ti + i) * H * BC + o_i)
        b_a = tl.where(o_i < i, b_a, 0.0)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    fuse_recompute: bool = False,
    fuse_diagonal: bool = False,
):
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    if fuse_diagonal:
        Aqk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    else:
        Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: compute diagonal blocks into Akkd (fp32)
    # When fuse_diagonal=True, diagonal blocks are computed inside inter_solve
    if not fuse_diagonal:
        if safe_gate:
            grid = (NT, NC, B * H)
            BK = triton.next_power_of_2(K)
            chunk_kda_fwd_kernel_intra_sub_chunk[grid](
                q=q,
                k=k,
                g=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                H=H,
                K=K,
                BT=BT,
                BC=BC,
                BK=BK,
                USE_GATHER=is_gather_supported,
            )
        else:
            Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
                q=q,
                k=k,
                gk=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                sub_chunk_size=BC,
            )

    # Step 2: inter_solve (+ optional fused recompute)
    grid = (NT, B * H)

    if fuse_recompute:
        w = torch.empty_like(k)
        u = torch.empty_like(v)
        kg = torch.empty_like(k)
        chunk_kda_fwd_kernel_inter_solve_fused[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akkd=Akkd,
            Akk=k,  # unused placeholder when FUSE_RECOMPUTE=True (dead branch)
            scale=scale,
            v_in=v,
            w_out=w,
            u_out=u,
            kg_out=kg,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BC=BC,
            USE_SAFE_GATE=safe_gate,
            FUSE_RECOMPUTE=True,
            FUSE_DIAGONAL=fuse_diagonal,
        )
        return w, u, None, kg, Aqk, None

    # Non-fused path: inter_solve stores Akk, then separate recompute
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        # v_in/w_out/u_out/kg_out unused when FUSE_RECOMPUTE=False (dead branch)
        v_in=k,
        w_out=k,
        u_out=k,
        kg_out=k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=0,
        BT=BT,
        BC=BC,
        USE_SAFE_GATE=safe_gate,
        FUSE_RECOMPUTE=False,
        FUSE_DIAGONAL=fuse_diagonal,
    )

    from sglang.srt.layers.attention.fla.kda import (
        recompute_w_u_fwd as kda_recompute_w_u_fwd,
    )

    w, u, qg, kg = kda_recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk
