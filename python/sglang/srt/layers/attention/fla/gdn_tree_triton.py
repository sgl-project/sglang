"""K1-K3 fused tree-GDN target-verify — Triton implementation.

Semantics identical to gdn_tree_fused.tree_gdn_fused_verify (the validated
torch spec); see that file for the math. Four launches per layer, all
shape-static (CUDA-graph friendly), every matmul a tl.dot:

  K0 scalars   grid (N, NB, HV/16)    beta; prefix = P @ g for ALL heads at
                                      once (per-head P loads waste 48x traffic)
  K1 gram      grid (tri-tiles, N*H)  per K-HEAD: KK/QK dots once, then a
                                      static G loop applies each grouped
                                      v-head's beta/prefix masks (3x fewer
                                      dots); diagonal tiles also invert
                                      (I + A_bb) via nilpotent squaring
  K2 solve     grid (N*HV, NV)        RHS, then block forward substitution:
                                      U_b = inv_b @ (rhs_b - A[b,<b] @ U[<b])
                                      NB sequential dense block steps (T/32),
                                      not T scalar steps
  K3 out       grid (NB, NV, N*HV)    O = exp(prefix)·(Q H0^T) + QKD @ U

Empirical notes (B200, production config T=129 H=16 HV=48 K=V=128,
CUDA-graph replay; stage split 10.1 / 28.5 / 29.5 / 19.0 us = 87 us/layer
vs budget=1 floor 10.3 us; dense torch reference 385 us):
  - precision="tf32" for bf16 serving (error ~2e-4 abs, below input noise);
    "ieee" is fp32-exact for validation but Triton lowers it to scalar FMAs
    (~3.5x slower). K0's P@g dot is hardcoded tf32x3 (tensor cores at
    ~5e-7 rel; ieee there was 10x slower at its tiny grid).
  - num_warps: K1=4 (8 measured worse); K2/K3=8 with BV=64 (BV=64 at 4
    warps register-spills catastrophically — 777us; at 8 warps it halves
    A reloads and wins). BT=32 optimal: BT=64 measured worse at T=129
    (padding lanes dominate: 125 vs 87 us).
  - merging K3 into K2 was measured WORSE (the output stage loses its
    independent grid and serializes behind the solve) — separate launches.

Conventions pinned by the float64 reference harness (fp64 ref / production
chain kernel): state memory layout [*, HV, V, K] (element (k,v) at flat v*K+k);
g = -exp(A_log)·softplus(a+dt_bias); beta = sigmoid(b); q,k l2norm eps 1e-6
inside sqrt; q *= K^-0.5; ratio-form decay exp(prefix_i - prefix_j).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.gdn_tree_fused import (
    LazyGDNVerifyState,
    TreeStructure,
)

BT = 32  # tile/block size (rows of A per program, solve block size)

# Scratch reused across layers/calls (A/QKD/Ainv/beta are consumed within the
# call; safe on a single stream, and under CUDA graphs it avoids pinning one
# copy per captured layer). U and prefix are NOT here: the returned lazy
# state aliases them, so they must stay per-call.
_WORKSPACE: dict = {}
_TRI_CACHE: dict = {}


def _ws(key, alloc):
    buf = _WORKSPACE.get(key)
    if buf is None:
        buf = alloc()
        _WORKSPACE[key] = buf
    return buf


def _tri_tiles(NB: int, device) -> tuple[torch.Tensor, int]:
    """Flat [(i_t, i_j)] lookup for the lower-triangular K1 tile grid."""
    key = (NB, device)
    hit = _TRI_CACHE.get(key)
    if hit is None:
        pairs = [(it, jt) for it in range(NB) for jt in range(it + 1)]
        t = torch.tensor(pairs, dtype=torch.int32, device=device).flatten().contiguous()
        hit = (t, len(pairs))
        _TRI_CACHE[key] = hit
    return hit


@triton.jit
def _gate(a, A_log, dt_bias, sp_beta, sp_thr):
    x = (a + dt_bias) * sp_beta
    sp = tl.where(x <= sp_thr, tl.log(1.0 + tl.exp(x)) / sp_beta, a + dt_bias)
    return -tl.exp(A_log) * sp


@triton.jit
def _k0_scalars_kernel(
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, p_ptr,
    prefix_ptr, beta_ptr,
    T, HV: tl.constexpr, sp_beta, sp_thr,
    stride_a_n, stride_a_t,
    stride_p_n,
    stride_s_n, stride_s_t,
    NB: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
):
    # prefix = P @ g for ALL heads at once: one [BT,BT]@[BT,BH] dot per
    # t-chunk (P is per-sequence — loading it per head wastes 48x traffic).
    # tf32x3 dots: tensor-core speed at ~fp32 accuracy (~5e-7 rel) — ieee
    # lowers to scalar FMAs and is 10x slower at this tiny grid.
    i_n, i_rb, i_hc = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = i_rb * BT + tl.arange(0, BT)
    m_r = rows < T
    hv = i_hc * BH + tl.arange(0, BH)
    m_h = hv < HV
    A_log = tl.load(A_log_ptr + hv, mask=m_h, other=0.0).to(tl.float32)
    dtb = tl.load(dt_bias_ptr + hv, mask=m_h, other=0.0).to(tl.float32)

    acc = tl.zeros((BT, BH), dtype=tl.float32)
    for tb in tl.static_range(NB):
        tcols = tb * BT + tl.arange(0, BT)
        m_t = tcols < T
        p_tile = tl.load(
            p_ptr + i_n * stride_p_n + rows[:, None] * T + tcols[None, :],
            mask=m_r[:, None] & m_t[None, :], other=0,
        ).to(tl.float32)
        a_tile = tl.load(
            a_ptr + i_n * stride_a_n + tcols[:, None] * stride_a_t + hv[None, :],
            mask=m_t[:, None] & m_h[None, :], other=0.0,
        ).to(tl.float32)
        g_tile = tl.where(
            m_t[:, None] & m_h[None, :],
            _gate(a_tile, A_log[None, :], dtb[None, :], sp_beta, sp_thr), 0.0,
        )
        acc += tl.dot(p_tile, g_tile, input_precision="tf32x3")

    b_v = tl.load(
        b_ptr + i_n * stride_a_n + rows[:, None] * stride_a_t + hv[None, :],
        mask=m_r[:, None] & m_h[None, :], other=0.0,
    ).to(tl.float32)
    beta = 1.0 / (1.0 + tl.exp(-b_v))
    m_rh = m_r[:, None] & m_h[None, :]
    tl.store(prefix_ptr + i_n * stride_s_n + rows[:, None] * stride_s_t + hv[None, :], acc, mask=m_rh)
    tl.store(beta_ptr + i_n * stride_s_n + rows[:, None] * stride_s_t + hv[None, :], beta, mask=m_rh)


@triton.jit
def _k1_gram_kernel(
    q_ptr, k_ptr, p_ptr, prefix_ptr, beta_ptr,
    A_ptr, QKD_ptr, Ainv_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_q_n, stride_q_t, stride_q_h,
    stride_p_n,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_inv_nh, stride_inv_b,
    USE_L2NORM: tl.constexpr,
    NBT: tl.constexpr, BK: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr, G: tl.constexpr, MAX_DEPTH: tl.constexpr,
    tile_ptr=None,
):
    # lower-triangular tile grid: upper tiles (j > i) are identically zero
    # and never read by K2/K3, so they are not computed or stored.
    # grid is per K-HEAD (not per v-head): the G grouped v-heads share the
    # same KK/QK Grams and differ only in beta/prefix masks — computing the
    # dots once and applying G masks removes a Gx redundancy.
    i_l, i_nh_h = tl.program_id(0), tl.program_id(1)
    i_t = tl.load(tile_ptr + 2 * i_l)
    i_j = tl.load(tile_ptr + 2 * i_l + 1)
    i_n, i_h = i_nh_h // H, i_nh_h % H

    rows = i_t * BT + tl.arange(0, BT)
    cols = i_j * BT + tl.arange(0, BT)
    m_r, m_c = rows < T, cols < T
    offs_k = tl.arange(0, BK)
    m_k = offs_k < K

    k_i = tl.load(
        k_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    k_j = tl.load(
        k_ptr + i_n * stride_q_n + cols[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_c[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    q_i = tl.load(
        q_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        k_i = k_i / tl.sqrt(tl.sum(k_i * k_i, 1) + 1e-6)[:, None]
        k_j = k_j / tl.sqrt(tl.sum(k_j * k_j, 1) + 1e-6)[:, None]
        q_i = q_i / tl.sqrt(tl.sum(q_i * q_i, 1) + 1e-6)[:, None]
    q_i = q_i * scale

    kk = tl.dot(k_i, tl.trans(k_j), input_precision=PREC)
    qk = tl.dot(q_i, tl.trans(k_j), input_precision=PREC)

    p_tile = tl.load(
        p_ptr + i_n * stride_p_n + rows[:, None] * T + cols[None, :],
        mask=m_r[:, None] & m_c[None, :], other=0,
    )
    incl = p_tile > 0
    strict = incl & (rows[:, None] != cols[None, :])
    eye = (tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]).to(tl.float32)
    ij = tl.arange(0, BT)[:, None] * BT + tl.arange(0, BT)[None, :]

    for g_i in tl.static_range(G):
        i_hv = i_h * G + g_i
        i_nh = i_n * HV + i_hv
        pre_i = tl.load(prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        pre_j = tl.load(prefix_ptr + i_n * stride_s_n + cols * stride_s_t + i_hv, mask=m_c, other=0.0)
        beta_i = tl.load(beta_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        dexp = tl.exp(pre_i[:, None] - pre_j[None, :])

        a_tile = tl.where(strict, beta_i[:, None] * dexp * kk, 0.0)
        qkd_tile = tl.where(incl, dexp * qk, 0.0)

        base = i_nh * stride_A_nh + rows[:, None] * stride_A_i + cols[None, :]
        tl.store(A_ptr + base, a_tile)
        tl.store(QKD_ptr + base, qkd_tile)

        if i_t == i_j:
            # (I + A_bb)^-1 = sum_j (-A_bb)^j, A_bb nilpotent (strictly
            # lower) — repeated squaring covers j < 32 = BT
            n1 = -a_tile
            x = eye + n1
            m2 = tl.dot(n1, n1, input_precision=PREC)
            x = x + tl.dot(x, m2, input_precision=PREC)
            if MAX_DEPTH > 3:
                m2 = tl.dot(m2, m2, input_precision=PREC)
                x = x + tl.dot(x, m2, input_precision=PREC)
            if MAX_DEPTH > 7:
                m2 = tl.dot(m2, m2, input_precision=PREC)
                x = x + tl.dot(x, m2, input_precision=PREC)
            if MAX_DEPTH > 15:
                m2 = tl.dot(m2, m2, input_precision=PREC)
                x = x + tl.dot(x, m2, input_precision=PREC)
            if MAX_DEPTH > 31:
                m2 = tl.dot(m2, m2, input_precision=PREC)
                x = x + tl.dot(x, m2, input_precision=PREC)
            inv_base = i_nh * stride_inv_nh + i_t * stride_inv_b
            tl.store(Ainv_ptr + inv_base + ij, x)


@triton.jit
def _k2_solve_kernel(
    k_ptr, v_ptr, prefix_ptr, beta_ptr,
    A_ptr, Ainv_ptr, h0_ptr, h0_idx_ptr,
    U_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_k_n, stride_k_t, stride_k_h,
    stride_v_n, stride_v_t, stride_v_h,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_inv_nh, stride_inv_b,
    stride_U_nh, stride_U_t,
    USE_L2NORM: tl.constexpr, USE_H0: tl.constexpr,
    NB: tl.constexpr, BK: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = offs_k < K, offs_v < V

    h0 = tl.zeros((BK, BV), dtype=tl.float32)
    if USE_H0:
        idx = tl.load(h0_idx_ptr + i_n).to(tl.int32)
        if idx >= 0:
            h0 = tl.load(
                h0_ptr + idx * HV * V * K + i_hv * V * K
                + offs_v[None, :] * K + offs_k[:, None],
                mask=m_k[:, None] & m_v[None, :], other=0.0,
            ).to(tl.float32)

    ij = tl.arange(0, BT)[:, None] * BT + tl.arange(0, BT)[None, :]
    for cur in tl.static_range(NB):
        rows = cur * BT + tl.arange(0, BT)
        m_r = rows < T
        k_b = tl.load(
            k_ptr + i_n * stride_k_n + rows[:, None] * stride_k_t + i_h * stride_k_h + offs_k[None, :],
            mask=m_r[:, None] & m_k[None, :], other=0.0,
        ).to(tl.float32)
        if USE_L2NORM:
            k_b = k_b / tl.sqrt(tl.sum(k_b * k_b, 1) + 1e-6)[:, None]
        v_b = tl.load(
            v_ptr + i_n * stride_v_n + rows[:, None] * stride_v_t + i_hv * stride_v_h + offs_v[None, :],
            mask=m_r[:, None] & m_v[None, :], other=0.0,
        ).to(tl.float32)
        pre_b = tl.load(prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        beta_b = tl.load(beta_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0)
        kh0 = tl.dot(k_b, h0, input_precision=PREC)
        acc = beta_b[:, None] * (v_b - tl.exp(pre_b)[:, None] * kh0)
        for pb in tl.static_range(cur):
            pcols = pb * BT + tl.arange(0, BT)
            a_blk = tl.load(
                A_ptr + i_nh * stride_A_nh + rows[:, None] * stride_A_i + pcols[None, :]
            )
            u_blk = tl.load(
                U_ptr + i_nh * stride_U_nh + pcols[:, None] * stride_U_t + offs_v[None, :]
            )
            acc -= tl.dot(a_blk, u_blk, input_precision=PREC)
        inv = tl.load(Ainv_ptr + i_nh * stride_inv_nh + cur * stride_inv_b + ij)
        u_cur = tl.dot(inv, acc, input_precision=PREC)
        tl.store(
            U_ptr + i_nh * stride_U_nh + rows[:, None] * stride_U_t + offs_v[None, :], u_cur
        )


@triton.jit
def _k3_out_kernel(
    q_ptr, prefix_ptr,
    QKD_ptr, U_ptr, h0_ptr, h0_idx_ptr,
    O_ptr,
    T, scale,
    H: tl.constexpr, HV: tl.constexpr,
    stride_q_n, stride_q_t, stride_q_h,
    stride_s_n, stride_s_t,
    stride_A_nh, stride_A_i,
    stride_U_nh, stride_U_t,
    stride_o_n, stride_o_t, stride_o_h,
    USE_L2NORM: tl.constexpr, USE_H0: tl.constexpr,
    NB: tl.constexpr, BK: tl.constexpr, K: tl.constexpr,
    BV: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    PREC: tl.constexpr,
):
    i_t, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    rows = i_t * BT + tl.arange(0, BT)
    m_r = rows < T
    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    m_k, m_v = offs_k < K, offs_v < V

    q_i = tl.load(
        q_ptr + i_n * stride_q_n + rows[:, None] * stride_q_t + i_h * stride_q_h + offs_k[None, :],
        mask=m_r[:, None] & m_k[None, :], other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        q_i = q_i / tl.sqrt(tl.sum(q_i * q_i, 1) + 1e-6)[:, None]
    q_i = q_i * scale

    acc = tl.zeros((BT, BV), dtype=tl.float32)
    if USE_H0:
        idx = tl.load(h0_idx_ptr + i_n).to(tl.int32)
        if idx >= 0:
            h0 = tl.load(
                h0_ptr + idx * HV * V * K + i_hv * V * K
                + offs_v[None, :] * K + offs_k[:, None],
                mask=m_k[:, None] & m_v[None, :], other=0.0,
            ).to(tl.float32)
            pre_i = tl.load(
                prefix_ptr + i_n * stride_s_n + rows * stride_s_t + i_hv, mask=m_r, other=0.0
            )
            acc = tl.exp(pre_i)[:, None] * tl.dot(q_i, h0, input_precision=PREC)

    for jt in range(0, i_t + 1):  # tiles with jt > i_t are zero (no ancestors above)
        cols = jt * BT + tl.arange(0, BT)
        qkd = tl.load(
            QKD_ptr + i_nh * stride_A_nh + rows[:, None] * stride_A_i + cols[None, :]
        )
        u_blk = tl.load(
            U_ptr + i_nh * stride_U_nh + cols[:, None] * stride_U_t + offs_v[None, :]
        )
        acc += tl.dot(qkd, u_blk, input_precision=PREC)

    tl.store(
        O_ptr + i_n * stride_o_n + rows[:, None] * stride_o_t + i_hv * stride_o_h + offs_v[None, :],
        acc.to(O_ptr.dtype.element_ty),
        mask=m_r[:, None] & m_v[None, :],
    )


def tree_gdn_triton_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    tree: TreeStructure,
    scale=None,
    use_qk_l2norm_in_kernel: bool = False,
    return_lazy_state: bool = False,
    precision: str = "ieee",
):
    """Same contract as gdn_tree_fused.tree_gdn_fused_verify. precision:
    "ieee" = fp32-exact dots (validation); "tf32" = tensor cores (serving,
    ~2e-4 abs vs ieee — below bf16 input noise)."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    NB = triton.cdiv(T, BT)
    NBT = NB * BT
    BK = max(16, triton.next_power_of_2(K))
    BV = min(64, max(16, triton.next_power_of_2(V)))
    BH = 16  # head-chunk width; chunked grid keeps K0 parallel
    NHC = triton.cdiv(HV, BH)
    NV = triton.cdiv(V, BV)
    G = HV // H
    dev = q.device

    p_u8 = tree.anc_u8  # [N, T, T], built once per verify step
    prefix = torch.empty(B, NBT, HV, device=dev, dtype=torch.float32)
    beta = _ws(("beta", B, NBT, HV, dev), lambda: torch.empty(B, NBT, HV, device=dev, dtype=torch.float32))
    A = _ws(("A", B * HV, NBT, dev), lambda: torch.empty(B * HV, NBT, NBT, device=dev, dtype=torch.float32))
    QKD = _ws(("QKD", B * HV, NBT, dev), lambda: torch.empty(B * HV, NBT, NBT, device=dev, dtype=torch.float32))
    Ainv = _ws(("Ainv", B * HV, NB, dev), lambda: torch.empty(B * HV, NB, BT, BT, device=dev, dtype=torch.float32))
    U = torch.empty(B * HV, NBT, V, device=dev, dtype=torch.float32)
    O = torch.empty(B, T, HV, V, device=dev, dtype=v.dtype)
    tile_t, n_tiles = _tri_tiles(NB, dev)

    _k0_scalars_kernel[(B, NB, NHC)](
        a, b, A_log, dt_bias, p_u8, prefix, beta,
        T, HV, softplus_beta, softplus_threshold,
        a.stride(0), a.stride(1),
        p_u8.stride(0),
        prefix.stride(0), prefix.stride(1),
        NB=NB, BT=BT, BH=BH,
        num_warps=4,
    )
    _k1_gram_kernel[(n_tiles, B * H)](
        q, k, p_u8, prefix, beta, A, QKD, Ainv,
        T, scale, H, HV,
        q.stride(0), q.stride(1), q.stride(2),
        p_u8.stride(0),
        prefix.stride(0), prefix.stride(1),
        A.stride(0), A.stride(1),
        Ainv.stride(0), Ainv.stride(1),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        NBT=NBT, BK=BK, K=K, BT=BT, PREC=precision, G=G,
        MAX_DEPTH=tree.max_depth,
        tile_ptr=tile_t,
        num_warps=4,
    )
    _k2_solve_kernel[(B * HV, NV)](
        k, v, prefix, beta, A, Ainv,
        initial_state_source, initial_state_indices, U,
        T, scale, H, HV,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        prefix.stride(0), prefix.stride(1),
        A.stride(0), A.stride(1),
        Ainv.stride(0), Ainv.stride(1),
        U.stride(0), U.stride(1),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        USE_H0=initial_state_source is not None,
        NB=NB, BK=BK, K=K, BV=BV, V=V, BT=BT, PREC=precision,
        num_warps=8,
    )
    _k3_out_kernel[(NB, NV, B * HV)](
        q, prefix, QKD, U,
        initial_state_source, initial_state_indices, O,
        T, scale, H, HV,
        q.stride(0), q.stride(1), q.stride(2),
        prefix.stride(0), prefix.stride(1),
        QKD.stride(0), QKD.stride(1),
        U.stride(0), U.stride(1),
        O.stride(0), O.stride(1), O.stride(2),
        USE_L2NORM=use_qk_l2norm_in_kernel,
        USE_H0=initial_state_source is not None,
        NB=NB, BK=BK, K=K, BV=BV, V=V, BT=BT, PREC=precision,
        num_warps=4,
    )

    if return_lazy_state:
        lazy = LazyGDNVerifyState(
            k=k,
            u=U.view(B, HV, NBT, V)[:, :, :T],
            prefix=prefix[:, :T],
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            retrieve_parent_token=tree.parent,
            max_tree_depth=tree.max_depth,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=None,
        )
        return O, lazy
    return O
