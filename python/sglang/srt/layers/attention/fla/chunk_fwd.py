# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk_fwd.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.op import safe_exp
from sglang.srt.layers.attention.fla.utils import (
    autotune_cache_kwargs,
    is_intel,
    is_tf32_supported,
)
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd

# TF32 for the block-merge dot products (16x16 matmuls) is safe and ~2x faster on SM90.
# The numerically sensitive forward-substitution uses scalar ops, not tl.dot.
if is_tf32_supported:
    _MERGE_DOT_PRECISION = tl.constexpr("tf32")
else:
    _MERGE_DOT_PRECISION = tl.constexpr("ieee")


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [16, 32, 64]
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["H", "Hg", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.

    This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel,
    avoiding the HBM round-trip for the intermediate A matrix.

    Steps:
    1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
    2. Apply gate and beta scaling
    3. Forward substitution on diagonal blocks
    4. Block merge to get full (I+A)^{-1}
    5. Write result to A (output)
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

    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # load beta for each sub-chunk
    p_b0 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
    p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
    p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
    p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
    b_b0 = tl.load(p_b0, boundary_check=(0,)).to(tl.float32)
    b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
    b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
    b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)

    # load gate if used
    if USE_G:
        p_g0 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc0,), (BC,), (0,))
        p_g1 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        p_g2 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        p_g3 = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))

        b_g0 = tl.load(p_g0, boundary_check=(0,)).to(tl.float32)
        b_g1 = tl.load(p_g1, boundary_check=(0,)).to(tl.float32)
        b_g2 = tl.load(p_g2, boundary_check=(0,)).to(tl.float32)
        b_g3 = tl.load(p_g3, boundary_check=(0,)).to(tl.float32)

    ############################################################################
    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    # 4 diagonal blocks
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(
            k, (T, K), (Hg * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0)
        )
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        # diagonal block 0
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0)
            )
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            # diagonal block 1
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0)
                )
                b_k2 = tl.load(p_k2, boundary_check=(0, 1))
                # diagonal block 2
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = tl.make_block_ptr(
                        k, (T, K), (Hg * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0)
                    )
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1))
                    # diagonal block 3
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: apply gate and beta scaling
    ############################################################################

    if USE_G:
        # diagonal blocks: g_diff = g_i - g_j within sub-chunk
        b_A00 *= safe_exp(b_g0[:, None] - b_g0[None, :])
        b_A11 *= safe_exp(b_g1[:, None] - b_g1[None, :])
        b_A22 *= safe_exp(b_g2[:, None] - b_g2[None, :])
        b_A33 *= safe_exp(b_g3[:, None] - b_g3[None, :])

        # off-diagonal blocks: g_diff = g_row - g_col (cross sub-chunk)
        b_A10 *= safe_exp(b_g1[:, None] - b_g0[None, :])
        b_A20 *= safe_exp(b_g2[:, None] - b_g0[None, :])
        b_A21 *= safe_exp(b_g2[:, None] - b_g1[None, :])
        b_A30 *= safe_exp(b_g3[:, None] - b_g0[None, :])
        b_A31 *= safe_exp(b_g3[:, None] - b_g1[None, :])
        b_A32 *= safe_exp(b_g3[:, None] - b_g2[None, :])

    # apply beta to row dimension and mask
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # diagonal blocks: strictly lower triangular within sub-chunk, scaled by beta
    b_A00 = (
        tl.where(m_d & (m_tc0[:, None] & m_tc0[None, :]), b_A00, 0.0) * b_b0[:, None]
    )
    b_A11 = (
        tl.where(m_d & (m_tc1[:, None] & m_tc1[None, :]), b_A11, 0.0) * b_b1[:, None]
    )
    b_A22 = (
        tl.where(m_d & (m_tc2[:, None] & m_tc2[None, :]), b_A22, 0.0) * b_b2[:, None]
    )
    b_A33 = (
        tl.where(m_d & (m_tc3[:, None] & m_tc3[None, :]), b_A33, 0.0) * b_b3[:, None]
    )

    # off-diagonal blocks: full block, scaled by beta
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    ############################################################################
    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    #
    # Same algorithm as solve_tril, but rows are extracted from in-register
    # [BC, BC] tensor via tl.sum(tl.where(mask, tensor, 0), 0) instead of
    # tl.load from HBM.
    ############################################################################

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 4: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=_MERGE_DOT_PRECISION),
        b_Ai00,
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=_MERGE_DOT_PRECISION),
        b_Ai11,
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=_MERGE_DOT_PRECISION),
        b_Ai22,
        input_precision=_MERGE_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=_MERGE_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=_MERGE_DOT_PRECISION),
        input_precision=_MERGE_DOT_PRECISION,
    )

    ############################################################################
    # Step 5: store full (I + A)^{-1} to output A
    ############################################################################

    p_A00 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_A10 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_A11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_A20 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_A21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_A22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0)
    )
    p_A30 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_A31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_A32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0)
    )
    p_A33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0)
    )

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [16, 32, 64]
        for num_warps in [2, 4, 8, 16, 32]
    ],
    key=["H", "Hg", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Low-reg version: one [BC,BC] accumulator at a time to minimise register pressure.

    Pass 1: loop over 4 diagonal blocks (tl.static_range unrolls to 4 K-loops).
    Pass 2: nested loop over off-diagonal distance d=1,2,3 and column j.
      d=1 (nearest): Ai_{ij} = -Ai_ii @ A_ij_raw @ Ai_jj
      d>1 (farther):  Ai_{ij} = -(Ai_ii @ A_ij_raw
                                  + sum_{m=j+1}^{i-1} Ai_im @ A_mj_raw) @ Ai_jj
    Each K-loop holds exactly one [BC,BC] accumulator.  Raw blocks needed by
    later correction terms are spilled to upper-triangular scratch slots in A
    (see _KKT_SCRATCH_COL for the layout; boundary_check makes out-of-bounds
    stores/loads safe so no runtime `if i_tcX < T` guards are needed).
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
    k += (bos * Hg + i_h // (H // Hg)) * K
    A += (bos * H + i_h) * BT

    o_i = tl.arange(0, BC)
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    ############################################################################
    # Pass 1: diagonal blocks — one K-loop per sub-chunk (tl.static_range → 4)
    ############################################################################
    for i_b in tl.static_range(4):
        i_tci = i_tc0 + i_b * BC
        m_tci = (i_tci + o_i) < T
        p_bi = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
        )
        b_bi = tl.load(p_bi, boundary_check=(0,)).to(tl.float32)
        if USE_G:
            p_gi = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
            )
            b_gi = tl.load(p_gi, boundary_check=(0,)).to(tl.float32)
        b_A = tl.zeros([BC, BC], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k, (T, K), (Hg * K, 1), (i_tci, i_k * BK), (BC, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_k, tl.trans(b_k))
        if USE_G:
            b_A *= safe_exp(b_gi[:, None] - b_gi[None, :])
        b_A = (
            tl.where(m_d & (m_tci[:, None] & m_tci[None, :]), b_A, 0.0) * b_bi[:, None]
        )
        # Forward substitution: solve (I + A_diag) x = I column by column.
        # Extra iterations for out-of-bounds rows are no-ops (b_A rows are zero).
        b_Ai = -b_A
        for i in range(2, BC):
            b_a = tl.sum(tl.where((o_i == i)[:, None], -b_A, 0.0), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a = b_a + tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        p_Aii = tl.make_block_ptr(
            A, (T, BT), (H * BT, 1), (i_tci, i_b * BC), (BC, BC), (1, 0)
        )
        tl.store(p_Aii, b_Ai.to(A.dtype.element_ty), boundary_check=(0, 1))

    ############################################################################
    # Pass 2: off-diagonal blocks — one K-loop per (i,j) pair.
    # Outer loop: d = i-j (diagonal distance) from 1 to 3.
    # Inner loop: j from 0 to 3-d  (so i = j+d).
    # Processing order ensures Ai_im (m in j+1..i-1) is already in A when needed.
    # Formula: Ai_ij = -(Ai_ii @ A_ij_raw
    #                    + sum_{m=j+1}^{i-1} Ai_im @ A_mj_raw) @ Ai_jj
    ############################################################################
    for d in tl.static_range(1, 4):
        for j in tl.static_range(0, 4 - d):
            i = j + d  # compile-time ints from static_range
            i_tci = i_tc0 + i * BC
            i_tcj = i_tc0 + j * BC
            p_bi = tl.make_block_ptr(
                beta + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
            )
            b_bi = tl.load(p_bi, boundary_check=(0,)).to(tl.float32)
            if USE_G:
                p_gi = tl.make_block_ptr(
                    g + bos * H + i_h, (T,), (H,), (i_tci,), (BC,), (0,)
                )
                p_gj = tl.make_block_ptr(
                    g + bos * H + i_h, (T,), (H,), (i_tcj,), (BC,), (0,)
                )
                b_gi = tl.load(p_gi, boundary_check=(0,)).to(tl.float32)
                b_gj = tl.load(p_gj, boundary_check=(0,)).to(tl.float32)
            # K-loop: accumulate k_i @ k_j^T
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_ki = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tci, i_k * BK), (BC, BK), (1, 0)
                )
                p_kj = tl.make_block_ptr(
                    k, (T, K), (Hg * K, 1), (i_tcj, i_k * BK), (BC, BK), (1, 0)
                )
                b_A += tl.dot(
                    tl.load(p_ki, boundary_check=(0, 1)),
                    tl.trans(tl.load(p_kj, boundary_check=(0, 1))),
                )
            if USE_G:
                b_A *= safe_exp(b_gi[:, None] - b_gj[None, :])
            b_A *= b_bi[:, None]
            # Scratch this raw block if a later row in this column needs it as
            # a correction term.  i < 3 means rows i+1..3 exist and will use it.
            # Scratch column layout (compile-time ternary, evaluated at trace time):
            #   (i=1,j=0)->col1  (i=2,j=1)->col2  (i=2,j=0)->col3
            if i < 3:
                sc = 1 if (i == 1 and j == 0) else (2 if (i == 2 and j == 1) else 3)
                p_s = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc * BC), (BC, BC), (1, 0)
                )
                tl.store(p_s, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
            # Correction sum: sum_{m=j+1}^{i-1} Ai_im @ A_mj_raw
            # Unrolled manually (d is a compile-time Python int from tl.static_range):
            #   d=1: no corrections; d=2: m=j+1; d=3: m=j+1 then m=j+2
            b_corr = tl.zeros([BC, BC], dtype=tl.float32)
            if d >= 2:
                m1 = j + 1
                sc_m1j = (
                    1 if (m1 == 1 and j == 0) else (2 if (m1 == 2 and j == 1) else 3)
                )
                p_s_m1j = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc_m1j * BC), (BC, BC), (1, 0)
                )
                b_A_m1j = tl.load(p_s_m1j, boundary_check=(0, 1)).to(tl.float32)
                p_Ai_im1 = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tci, m1 * BC), (BC, BC), (1, 0)
                )
                b_corr += tl.dot(
                    tl.load(p_Ai_im1, boundary_check=(0, 1)).to(tl.float32),
                    b_A_m1j,
                    input_precision=_MERGE_DOT_PRECISION,
                )
            if d >= 3:
                m2 = j + 2
                sc_m2j = (
                    1 if (m2 == 1 and j == 0) else (2 if (m2 == 2 and j == 1) else 3)
                )
                p_s_m2j = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tc0, sc_m2j * BC), (BC, BC), (1, 0)
                )
                b_A_m2j = tl.load(p_s_m2j, boundary_check=(0, 1)).to(tl.float32)
                p_Ai_im2 = tl.make_block_ptr(
                    A, (T, BT), (H * BT, 1), (i_tci, m2 * BC), (BC, BC), (1, 0)
                )
                b_corr += tl.dot(
                    tl.load(p_Ai_im2, boundary_check=(0, 1)).to(tl.float32),
                    b_A_m2j,
                    input_precision=_MERGE_DOT_PRECISION,
                )
            # Compute and store Ai_ij
            p_Ai_ii = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tci, i * BC), (BC, BC), (1, 0)
            )
            p_Ai_jj = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tcj, j * BC), (BC, BC), (1, 0)
            )
            b_Ai_ii = tl.load(p_Ai_ii, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_jj = tl.load(p_Ai_jj, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_ij = -tl.dot(
                tl.dot(b_Ai_ii, b_A, input_precision=_MERGE_DOT_PRECISION) + b_corr,
                b_Ai_jj,
                input_precision=_MERGE_DOT_PRECISION,
            )
            p_Ai_ij = tl.make_block_ptr(
                A, (T, BT), (H * BT, 1), (i_tci, j * BC), (BC, BC), (1, 0)
            )
            tl.store(p_Ai_ij, b_Ai_ij.to(A.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    GDN intra-chunk forward: fused kkt + solve_tril + recompute_w_u.

    Equivalent to:
        A = chunk_scaled_dot_kkt_fwd(k, g, beta, ...)       # kernel 1
        A = solve_tril(A, ...)                                # kernel 2
        w, u = recompute_w_u_fwd(k, v, beta, A, g, ...)      # kernel 3

    Fuses kernels 1+2 into a single kernel, reducing from 3 to 2 kernel launches
    and eliminating the HBM round-trip for the intermediate A matrix.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        v (torch.Tensor):
            The value tensor of shape `[B, T, H, V]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths. Default: `None`.
        chunk_size (int):
            The chunk size. Default: 64.
        chunk_indices (torch.LongTensor):
            Precomputed chunk indices. Default: `None`.

    Returns:
        w (torch.Tensor): shape `[B, T, H, K]`
        u (torch.Tensor): shape `[B, T, H, V]`
        A (torch.Tensor): shape `[B, T, H, BT]`, the solved (I+A)^{-1} matrix
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Step 1: fused kkt + solve_tril
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    kernel = (
        chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg
        if is_intel
        else chunk_gated_delta_rule_fwd_kkt_solve_kernel
    )
    kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        BC=BC,
    )

    # Step 2: recompute_w_u
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, A
