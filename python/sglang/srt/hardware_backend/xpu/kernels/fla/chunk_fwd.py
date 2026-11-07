import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import safe_exp
from sglang.kernels.ops.attention.fla.utils import (
    autotune_cache_kwargs,
)
from sglang.kernels.ops.attention.fla.wy_fast import recompute_w_u_fwd

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
        for num_warps in [2, 4, 8, 16, 32]
    ],
    key=["H", "Hg", "K", "BC", "BK", "USE_G", "IS_VARLEN"],
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
    (see _KKT_SCRATCH_COL for the layout; masked stores/loads make
    out-of-bounds accesses safe so no runtime `if i_tcX < T` guards are needed).
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
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
        p_bi = beta + bos * H + i_h + (i_tci + o_i) * H
        b_bi = tl.load(p_bi, mask=m_tci, other=0.0).to(tl.float32)
        if USE_G:
            p_gi = g + bos * H + i_h + (i_tci + o_i) * H
            b_gi = tl.load(p_gi, mask=m_tci, other=0.0).to(tl.float32)
        b_A = tl.zeros([BC, BC], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            p_k = k + (i_tci + o_i)[:, None] * (Hg * K) + o_k[None, :]
            b_k = tl.load(p_k, mask=m_tci[:, None] & (o_k[None, :] < K), other=0.0)
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
        p_Aii = A + (i_tci + o_i)[:, None] * (H * BT) + (i_b * BC + o_i)[None, :]
        tl.store(
            p_Aii,
            b_Ai.to(A.dtype.element_ty),
            mask=m_tci[:, None] & ((i_b * BC + o_i)[None, :] < BT),
        )

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
            m_tci = (i_tci + o_i) < T
            m_tcj = (i_tcj + o_i) < T
            p_bi = beta + bos * H + i_h + (i_tci + o_i) * H
            b_bi = tl.load(p_bi, mask=m_tci, other=0.0).to(tl.float32)
            if USE_G:
                p_gi = g + bos * H + i_h + (i_tci + o_i) * H
                p_gj = g + bos * H + i_h + (i_tcj + o_i) * H
                b_gi = tl.load(p_gi, mask=m_tci, other=0.0).to(tl.float32)
                b_gj = tl.load(p_gj, mask=m_tcj, other=0.0).to(tl.float32)
            # K-loop: accumulate k_i @ k_j^T
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_kc = o_k[None, :] < K
                p_ki = k + (i_tci + o_i)[:, None] * (Hg * K) + o_k[None, :]
                p_kj = k + (i_tcj + o_i)[:, None] * (Hg * K) + o_k[None, :]
                b_A += tl.dot(
                    tl.load(p_ki, mask=m_tci[:, None] & m_kc, other=0.0),
                    tl.trans(tl.load(p_kj, mask=m_tcj[:, None] & m_kc, other=0.0)),
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
                m_tc0 = (i_tc0 + o_i) < T
                p_s = A + (i_tc0 + o_i)[:, None] * (H * BT) + (sc * BC + o_i)[None, :]
                tl.store(
                    p_s,
                    b_A.to(A.dtype.element_ty),
                    mask=m_tc0[:, None] & ((sc * BC + o_i)[None, :] < BT),
                )
            # Correction sum: sum_{m=j+1}^{i-1} Ai_im @ A_mj_raw
            # Unrolled manually (d is a compile-time Python int from tl.static_range):
            #   d=1: no corrections; d=2: m=j+1; d=3: m=j+1 then m=j+2
            b_corr = tl.zeros([BC, BC], dtype=tl.float32)
            if d >= 2:
                m1 = j + 1
                sc_m1j = (
                    1 if (m1 == 1 and j == 0) else (2 if (m1 == 2 and j == 1) else 3)
                )
                m_tc0 = (i_tc0 + o_i) < T
                p_s_m1j = (
                    A + (i_tc0 + o_i)[:, None] * (H * BT) + (sc_m1j * BC + o_i)[None, :]
                )
                b_A_m1j = tl.load(
                    p_s_m1j,
                    mask=m_tc0[:, None] & ((sc_m1j * BC + o_i)[None, :] < BT),
                    other=0.0,
                ).to(tl.float32)
                p_Ai_im1 = (
                    A + (i_tci + o_i)[:, None] * (H * BT) + (m1 * BC + o_i)[None, :]
                )
                b_corr += tl.dot(
                    tl.load(
                        p_Ai_im1,
                        mask=m_tci[:, None] & ((m1 * BC + o_i)[None, :] < BT),
                        other=0.0,
                    ).to(tl.float32),
                    b_A_m1j,
                    input_precision=_MERGE_DOT_PRECISION,
                )
            if d >= 3:
                m2 = j + 2
                sc_m2j = (
                    1 if (m2 == 1 and j == 0) else (2 if (m2 == 2 and j == 1) else 3)
                )
                p_s_m2j = (
                    A + (i_tc0 + o_i)[:, None] * (H * BT) + (sc_m2j * BC + o_i)[None, :]
                )
                b_A_m2j = tl.load(
                    p_s_m2j,
                    mask=m_tc0[:, None] & ((sc_m2j * BC + o_i)[None, :] < BT),
                    other=0.0,
                ).to(tl.float32)
                p_Ai_im2 = (
                    A + (i_tci + o_i)[:, None] * (H * BT) + (m2 * BC + o_i)[None, :]
                )
                b_corr += tl.dot(
                    tl.load(
                        p_Ai_im2,
                        mask=m_tci[:, None] & ((m2 * BC + o_i)[None, :] < BT),
                        other=0.0,
                    ).to(tl.float32),
                    b_A_m2j,
                    input_precision=_MERGE_DOT_PRECISION,
                )
            # Compute and store Ai_ij
            p_Ai_ii = A + (i_tci + o_i)[:, None] * (H * BT) + (i * BC + o_i)[None, :]
            p_Ai_jj = A + (i_tcj + o_i)[:, None] * (H * BT) + (j * BC + o_i)[None, :]
            b_Ai_ii = tl.load(
                p_Ai_ii, mask=m_tci[:, None] & ((i * BC + o_i)[None, :] < BT), other=0.0
            ).to(tl.float32)
            b_Ai_jj = tl.load(
                p_Ai_jj, mask=m_tcj[:, None] & ((j * BC + o_i)[None, :] < BT), other=0.0
            ).to(tl.float32)
            b_Ai_ij = -tl.dot(
                tl.dot(b_Ai_ii, b_A, input_precision=_MERGE_DOT_PRECISION) + b_corr,
                b_Ai_jj,
                input_precision=_MERGE_DOT_PRECISION,
            )
            p_Ai_ij = A + (i_tci + o_i)[:, None] * (H * BT) + (j * BC + o_i)[None, :]
            tl.store(
                p_Ai_ij,
                b_Ai_ij.to(A.dtype.element_ty),
                mask=m_tci[:, None] & ((j * BC + o_i)[None, :] < BT),
            )

    # Clean up scratch slots: Pass 2 stored raw A_ij blocks in the upper-triangular
    # part of row i_tc0 (cols BC..3*BC). These must be zeroed because
    # recompute_w_u_fwd reads the full BT×BT block.
    b_zero = tl.zeros([BC, BC], dtype=tl.float32)
    m_tc0 = (i_tc0 + o_i) < T
    for sc in tl.static_range(1, BT // BC):
        p_scratch = A + (i_tc0 + o_i)[:, None] * (H * BT) + (sc * BC + o_i)[None, :]
        tl.store(
            p_scratch,
            b_zero.to(A.dtype.element_ty),
            mask=m_tc0[:, None] & ((sc * BC + o_i)[None, :] < BT),
        )


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
    kernel = chunk_gated_delta_rule_fwd_kkt_solve_kernel_low_reg
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
