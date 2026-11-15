# SPDX-License-Identifier: Apache-2.0
"""GDN tree verify without per-node state snapshots.

The state after node n is the checkpoint H0 decayed along the path to n plus
one rank-1 term per node on that path:

    H_n = exp(c_n) * H0 + sum_{i in path(n)} exp(c_n - c_i) * k_i (x) d_i

with c the cumulative log-decay and d_i the node's delta-rule update vector.
Each program loads its H0 tile and all node inputs once, forms H0^T k, H0^T q
and the k.k / q.k Gram matrices, and then resolves the nodes in registers. No
state is written. For a chain the ancestor mask is the strictly-lower causal
mask.

Each node's raw inputs go to the ReplaySSM rings at the node's index, and
``gdn_replayssm_spec_fold`` replays the accepted path from them on commit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.fla.utils import is_tf32_supported


@triton.jit(do_not_specialize=["T"])
def gdn_tree_verify_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    ring_indices,  # [B] ring row per request
    parent_token,  # [B, T], parent < node
    rawv,  # [rows, HV, RL, V]
    rawk,  # [rows, H, RL, K]
    ring_g,  # [rows, HV, RL]
    ring_beta,  # [rows, HV, RL]
    scale,
    T,
    stride_h0_slot,
    stride_parent_seq,
    stride_parent_tok,
    stride_rawv_row,
    stride_rawk_row,
    stride_g_row,
    stride_beta_row,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    RING_LEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_t = o_t < T
    mask_tk = mask_t[:, None] & mask_k[None, :]
    mask_tv = mask_t[:, None] & mask_v[None, :]
    tok = i_n.to(tl.int64) * T + o_t
    p_o = o + (tok[:, None] * HV + i_hv) * V + o_v[None, :]

    idx = tl.load(h0_indices + i_n).to(tl.int64)
    row = tl.load(ring_indices + i_n).to(tl.int64)
    if (idx < 0) | (row < 0):
        # Graph padding: no state to read, clear the output rows.
        tl.store(p_o, 0.0, mask=mask_tv)
        return

    # All node inputs at once: [BT, BK] keys and queries, [BT, BV] values.
    b_q = tl.load(
        q + tok[:, None] * stride_q + i_h * K + o_k[None, :], mask=mask_tk, other=0
    ).to(tl.float32)
    b_k = tl.load(
        k + tok[:, None] * stride_k + i_h * K + o_k[None, :], mask=mask_tk, other=0
    ).to(tl.float32)
    b_v = tl.load(
        v + tok[:, None] * stride_v + i_hv * V + o_v[None, :], mask=mask_tv, other=0
    ).to(tl.float32)
    b_a = tl.load(a + tok * stride_a + i_hv, mask=mask_t, other=0).to(tl.float32)
    b_b = tl.load(b + tok * stride_b + i_hv, mask=mask_t, other=0).to(tl.float32)
    parents = tl.load(
        parent_token + i_n.to(tl.int64) * stride_parent_seq + o_t * stride_parent_tok,
        mask=mask_t,
        other=0,
    ).to(tl.int32)

    x = b_a + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    b_g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    p_ring = i_hv * RING_LEN + o_t
    tl.store(
        rawv + row * stride_rawv_row + p_ring[:, None] * V + o_v[None, :],
        b_v.to(rawv.dtype.element_ty),
        mask=mask_tv,
    )
    if i_v == 0:
        tl.store(
            rawk
            + row * stride_rawk_row
            + (i_h * RING_LEN + o_t)[:, None] * K
            + o_k[None, :],
            b_k.to(rawk.dtype.element_ty),
            mask=mask_tk,
        )
        tl.store(ring_g + row * stride_g_row + p_ring, b_g, mask=mask_t)
        tl.store(ring_beta + row * stride_beta_row + p_ring, b_beta, mask=mask_t)

    # Normalization is applied to the products, so the tensor-core dots below
    # see the raw q and k, which TF32 holds exactly when they are half precision.
    q_norm = scale / tl.sqrt(tl.sum(b_q * b_q, 1) + 1e-6)
    k_norm = 1.0 / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)

    b_h0 = tl.load(
        h0_source
        + idx * stride_h0_slot
        + i_hv * K * V
        + o_v[None, :] * K
        + o_k[:, None],
        mask=mask_k[:, None] & mask_v[None, :],
        other=0,
    ).to(tl.float32)
    # TF32 keeps 10 mantissa bits: split the fp32 checkpoint into a high part
    # that TF32 holds exactly and a remainder about 2^-10 times smaller.
    h_hi = (b_h0.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)
    h_lo = b_h0 - h_hi
    b_kt = tl.trans(b_k)
    # [BT, BV]: H0^T k and H0^T q per node.
    h0_k = (
        tl.dot(b_k, h_hi, input_precision=DOT_PRECISION)
        + tl.dot(b_k, h_lo, input_precision=DOT_PRECISION)
    ) * k_norm[:, None]
    h0_q = (
        tl.dot(b_q, h_hi, input_precision=DOT_PRECISION)
        + tl.dot(b_q, h_lo, input_precision=DOT_PRECISION)
    ) * q_norm[:, None]
    # [BT, BT]: k_t . k_i and q_t . k_i.
    kk = tl.dot(b_k, b_kt, input_precision=DOT_PRECISION)
    kk = kk * k_norm[:, None] * k_norm[None, :]
    qk = tl.dot(b_q, b_kt, input_precision=DOT_PRECISION)
    qk = qk * q_norm[:, None] * k_norm[None, :]

    b_c = tl.zeros([BT], dtype=tl.float32)  # cumulative log-decay per node
    # ancestors[t, i]: i is an ancestor of t
    ancestors = tl.zeros([BT, BT], dtype=tl.float32)
    b_d = tl.zeros([BT, BV], dtype=tl.float32)  # update vector per node
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for t in range(T):
        is_t = o_t == t
        parent = tl.sum(tl.where(is_t, parents, 0))
        is_parent = (o_t == parent) & (parent < t)
        c_parent = tl.sum(tl.where(is_parent, b_c, 0.0))
        ancestors_t = tl.sum(
            tl.where(is_parent[:, None], ancestors, 0.0), 0
        ) + tl.where(is_parent, 1.0, 0.0)
        # Nodes on the path to the parent are selected, not weighted by zero:
        # a non-finite node must not reach the nodes outside its subtree.
        on_path = ancestors_t > 0
        w = tl.exp(c_parent - b_c)  # on the path: the decay from node i to the parent
        decay = tl.exp(c_parent)
        row_t = is_t[:, None]
        g_t = tl.sum(tl.where(is_t, b_g, 0.0))
        gate = tl.exp(g_t)
        wk = tl.where(on_path, w * tl.sum(tl.where(row_t, kk, 0.0), 0), 0.0)
        wq = tl.where(on_path, w * tl.sum(tl.where(row_t, qk, 0.0), 0), 0.0)
        d_path = tl.where(on_path[:, None], b_d, 0.0)
        hk = decay * tl.sum(tl.where(row_t, h0_k, 0.0), 0) + tl.sum(
            wk[:, None] * d_path, 0
        )
        hq = decay * tl.sum(tl.where(row_t, h0_q, 0.0), 0) + tl.sum(
            wq[:, None] * d_path, 0
        )
        v_t = tl.sum(tl.where(row_t, b_v, 0.0), 0)
        d_t = tl.sum(tl.where(is_t, b_beta, 0.0)) * (v_t - gate * hk)
        o_row = (
            gate * hq + tl.sum(tl.where(is_t[:, None] & is_t[None, :], qk, 0.0)) * d_t
        )
        b_d = tl.where(row_t, d_t[None, :], b_d)
        b_o = tl.where(row_t, o_row[None, :], b_o)
        b_c = tl.where(is_t, c_parent + g_t, b_c)
        ancestors = tl.where(row_t, ancestors_t[None, :], ancestors)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_tv)


def gdn_tree_verify(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,  # [bs * T, HV]
    dt_bias: torch.Tensor,
    q: torch.Tensor,  # [1, bs * T, H, K]
    k: torch.Tensor,  # [1, bs * T, H, K]
    v: torch.Tensor,  # [1, bs * T, HV, V]
    b: torch.Tensor,  # [bs * T, HV]
    initial_state_source: torch.Tensor,  # [slots, HV, V, K], read only
    initial_state_indices: torch.Tensor,  # [bs]
    ring_indices: torch.Tensor,  # [bs]
    retrieve_parent_token: torch.Tensor,  # [bs, T]
    replayssm_rawv: torch.Tensor,  # [rows, HV, RL, V]
    replayssm_rawk: torch.Tensor,  # [rows, H, RL, K]
    replayssm_g: torch.Tensor,  # [rows, HV, RL]
    replayssm_beta: torch.Tensor,  # [rows, HV, RL]
) -> torch.Tensor:
    """Returns the core attention output with ``v``'s shape."""
    _, T_all, H, K = k.shape
    HV, V = v.shape[2:]
    T = retrieve_parent_token.shape[1]
    bs = T_all // T
    assert T_all == bs * T, f"{T_all=} is not a multiple of the tree size {T}"
    ring_len = replayssm_rawv.shape[-2]
    assert ring_len >= T, f"{ring_len=} cannot hold a tree of {T} nodes"
    # tl.dot needs power-of-two tiles of at least 16.
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, min(triton.next_power_of_2(V), 32))
    tf32 = is_tf32_supported and q.dtype in (torch.bfloat16, torch.float16)
    o = q.new_empty(v.shape)
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    grid = (triton.cdiv(V, BV), bs, HV)
    gdn_tree_verify_kernel[grid](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_source,
        initial_state_indices,
        ring_indices,
        retrieve_parent_token,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        K**-0.5,
        T,
        initial_state_source.stride(0),
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
        replayssm_rawv.stride(0),
        replayssm_rawk.stride(0),
        replayssm_g.stride(0),
        replayssm_beta.stride(0),
        a.stride(-2),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        b.stride(-2),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BT=max(16, triton.next_power_of_2(T)),
        RING_LEN=ring_len,
        DOT_PRECISION="tf32" if tf32 else "ieee",
        USE_GDC=use_pdl,
        num_warps=1,
        num_stages=3,
        **pdl_kwargs,
    )
    return o
