"""Fused UBTree causal-conv and KDA node transform.

The production path is restricted to BS=1, 32/49 verify nodes, convolution
width 4, H == HV, and K == V. It replaces the tree causal-conv launch plus
``fused_sigmoid_gating_delta_rule_tree_transform_kernel`` while preserving the
materialized post-conv Q/K/V tensor consumed by factor generation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kda_conv_tree_transform_kernel(
    x,
    conv_weight,
    conv_bias,
    conv_state,
    conv_state_indices,
    intermediate_conv_window,
    factor_cache_indices,
    retrieve_parent_token,
    a,
    b,
    A_log,
    dt_bias,
    out,
    processed_k,
    processed_decay,
    processed_beta,
    tree_rawv,
    tree_rawk,
    tree_g,
    tree_beta,
    stride_x_batch: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_win: tl.constexpr,
    stride_inter_slot: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_parent_batch: tl.constexpr,
    stride_parent_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_dim: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    TREE_WIDTH: tl.constexpr,
    CACHE_TREE_WIDTH: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    CACHE_TREE_FACTORS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    batch = tl.program_id(0)
    node = tl.program_id(1)
    hv = tl.program_id(2)
    offsets = tl.arange(0, BK)
    mask = offsets < K
    state_slot = tl.load(conv_state_indices + batch).to(tl.int64)
    factor_slot = tl.load(factor_cache_indices + batch).to(tl.int64)

    q_dim_base = hv * K
    k_dim_base = HV * K + hv * K
    v_dim_base = 2 * HV * K + hv * K
    current = node.to(tl.int64)

    if HAS_BIAS:
        acc_q = tl.load(conv_bias + q_dim_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        acc_k = tl.load(conv_bias + k_dim_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        acc_v = tl.load(conv_bias + v_dim_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    else:
        acc_q = tl.zeros((BK,), dtype=tl.float32)
        acc_k = tl.zeros((BK,), dtype=tl.float32)
        acc_v = tl.zeros((BK,), dtype=tl.float32)

    # Preserve the node-local raw input used by accept-path conv commit.
    raw_q = tl.load(
        x
        + batch * stride_x_batch
        + (q_dim_base + offsets) * stride_x_dim
        + node * stride_x_token,
        mask=mask,
        other=0.0,
    )
    raw_k_input = tl.load(
        x
        + batch * stride_x_batch
        + (k_dim_base + offsets) * stride_x_dim
        + node * stride_x_token,
        mask=mask,
        other=0.0,
    )
    raw_v_input = tl.load(
        x
        + batch * stride_x_batch
        + (v_dim_base + offsets) * stride_x_dim
        + node * stride_x_token,
        mask=mask,
        other=0.0,
    )
    inter_base = (
        intermediate_conv_window
        + factor_slot * stride_inter_slot
        + node * stride_inter_step
    )
    tl.store(
        inter_base + (q_dim_base + offsets) * stride_inter_dim + 2 * stride_inter_win,
        raw_q,
        mask=mask & (factor_slot >= 0),
    )
    tl.store(
        inter_base + (k_dim_base + offsets) * stride_inter_dim + 2 * stride_inter_win,
        raw_k_input,
        mask=mask & (factor_slot >= 0),
    )
    tl.store(
        inter_base + (v_dim_base + offsets) * stride_inter_dim + 2 * stride_inter_win,
        raw_v_input,
        mask=mask & (factor_slot >= 0),
    )

    value_q = raw_q
    value_k = raw_k_input
    value_v = raw_v_input
    for tap in tl.static_range(4):
        weight_col = 3 - tap
        wq = tl.load(
            conv_weight + (q_dim_base + offsets) * 4 + weight_col, mask=mask, other=0.0
        ).to(tl.float32)
        wk = tl.load(
            conv_weight + (k_dim_base + offsets) * 4 + weight_col, mask=mask, other=0.0
        ).to(tl.float32)
        wv = tl.load(
            conv_weight + (v_dim_base + offsets) * 4 + weight_col, mask=mask, other=0.0
        ).to(tl.float32)
        acc_q += value_q.to(tl.float32) * wq
        acc_k += value_k.to(tl.float32) * wk
        acc_v += value_v.to(tl.float32) * wv

        if tap < 3:
            # Keep the ancestor walk branch-free.  ``current > 0`` means the
            # next value comes from a tree node; ``current <= 0`` encodes the
            # persistent conv-state columns 2, 1, 0.  Triton runtime branches
            # around a loop-carried scalar produced incorrect SSA merges for
            # sibling programs, while masked loads + tl.where are deterministic.
            use_tree_node = current > 0
            safe_node = tl.maximum(current, 0)
            parent = tl.load(
                retrieve_parent_token
                + batch * stride_parent_batch
                + safe_node * stride_parent_token
            ).to(tl.int64)
            history = tl.maximum(0, tl.minimum(2, 2 + current))

            tree_q = tl.load(
                x
                + batch * stride_x_batch
                + (q_dim_base + offsets) * stride_x_dim
                + parent * stride_x_token,
                mask=mask & use_tree_node,
                other=0.0,
            )
            tree_k = tl.load(
                x
                + batch * stride_x_batch
                + (k_dim_base + offsets) * stride_x_dim
                + parent * stride_x_token,
                mask=mask & use_tree_node,
                other=0.0,
            )
            tree_v = tl.load(
                x
                + batch * stride_x_batch
                + (v_dim_base + offsets) * stride_x_dim
                + parent * stride_x_token,
                mask=mask & use_tree_node,
                other=0.0,
            )
            state_q = tl.load(
                conv_state
                + state_slot * stride_state_slot
                + (q_dim_base + offsets) * stride_state_dim
                + history * stride_state_win,
                mask=mask & (~use_tree_node) & (state_slot >= 0),
                other=0.0,
            )
            state_k = tl.load(
                conv_state
                + state_slot * stride_state_slot
                + (k_dim_base + offsets) * stride_state_dim
                + history * stride_state_win,
                mask=mask & (~use_tree_node) & (state_slot >= 0),
                other=0.0,
            )
            state_v = tl.load(
                conv_state
                + state_slot * stride_state_slot
                + (v_dim_base + offsets) * stride_state_dim
                + history * stride_state_win,
                mask=mask & (~use_tree_node) & (state_slot >= 0),
                other=0.0,
            )
            value_q = tl.where(use_tree_node, tree_q, state_q)
            value_k = tl.where(use_tree_node, tree_k, state_k)
            value_v = tl.where(use_tree_node, tree_v, state_v)
            current = tl.where(use_tree_node, parent, current - 1)

    # Match the two-kernel reference boundary: causal-conv stores BF16, then
    # the transform kernel reloads and promotes that rounded value to FP32.
    q_conv = (acc_q * tl.sigmoid(acc_q)).to(out.dtype.element_ty)
    k_conv = (acc_k * tl.sigmoid(acc_k)).to(out.dtype.element_ty)
    v_conv = (acc_v * tl.sigmoid(acc_v)).to(out.dtype.element_ty)
    tl.store(
        out
        + batch * stride_out_batch
        + (q_dim_base + offsets) * stride_out_dim
        + node * stride_out_token,
        q_conv,
        mask=mask,
    )
    tl.store(
        out
        + batch * stride_out_batch
        + (k_dim_base + offsets) * stride_out_dim
        + node * stride_out_token,
        k_conv,
        mask=mask,
    )
    tl.store(
        out
        + batch * stride_out_batch
        + (v_dim_base + offsets) * stride_out_dim
        + node * stride_out_token,
        v_conv,
        mask=mask,
    )

    raw_k_conv = k_conv.to(tl.float32)
    k_conv_fp32 = k_conv.to(tl.float32)
    v_conv_fp32 = v_conv.to(tl.float32)
    norm = tl.sqrt(tl.sum(k_conv_fp32 * k_conv_fp32, axis=0) + 1e-6)
    k_norm = k_conv_fp32 / norm
    token = batch * TREE_WIDTH + node
    a_value = tl.load(
        a + token * stride_a_token + hv * K + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    b_value = tl.load(b + token * stride_b_token + hv).to(tl.float32)
    a_log = tl.load(A_log + hv).to(tl.float32)
    dt = tl.load(dt_bias + hv * K + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_input = a_value + dt
    if USE_LOWER_BOUND:
        g = LOWER_BOUND * tl.sigmoid(tl.exp(a_log) * gate_input)
    else:
        beta_x = SOFTPLUS_BETA * gate_input
        softplus_x = tl.where(
            beta_x <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(beta_x)) / SOFTPLUS_BETA,
            gate_input,
        )
        g = -tl.exp(a_log) * softplus_x
    decay = tl.exp(g)
    beta = tl.sigmoid(b_value)

    transform_base = ((batch * HV + hv) * TREE_WIDTH + node) * K
    tl.store(processed_k + transform_base + offsets, k_norm, mask=mask)
    tl.store(processed_decay + transform_base + offsets, decay, mask=mask)
    tl.store(processed_beta + (batch * HV + hv) * TREE_WIDTH + node, beta)

    if CACHE_TREE_FACTORS:
        cache_mask = mask & (factor_slot >= 0)
        tl.store(
            tree_rawv
            + factor_slot * stride_rawv_slot
            + (hv * CACHE_TREE_WIDTH + node) * K
            + offsets,
            v_conv_fp32,
            mask=cache_mask,
        )
        tl.store(
            tree_rawk
            + factor_slot * stride_rawk_slot
            + (hv * CACHE_TREE_WIDTH + node) * K
            + offsets,
            raw_k_conv,
            mask=cache_mask,
        )
        tl.store(
            tree_g
            + factor_slot * stride_g_slot
            + (hv * CACHE_TREE_WIDTH + node) * K
            + offsets,
            g,
            mask=cache_mask,
        )
        tl.store(
            tree_beta + factor_slot * stride_beta_slot + hv * CACHE_TREE_WIDTH + node,
            beta,
            mask=factor_slot >= 0,
        )


def fused_kda_conv_tree_transform(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_parent_token: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    processed_k: torch.Tensor,
    processed_decay: torch.Tensor,
    processed_beta: torch.Tensor,
    factor_cache_indices: torch.Tensor,
    tree_rawv: torch.Tensor,
    tree_rawk: torch.Tensor,
    tree_g: torch.Tensor,
    tree_beta: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Run fused conv + node transform for the fixed-width BS=1 path."""
    if x.ndim != 3:
        raise ValueError("x must be [batch, 3 * HV * K, tree_width]")
    batch, dim, tree_width = x.shape
    if batch != 1:
        raise ValueError(f"factorized UBTree conv supports only BS=1, got {batch}")
    if tree_width not in (32, 49):
        raise ValueError(f"tree_width must be 32 or 49, got {tree_width}")
    hv, key_dim = processed_beta.shape[1], processed_k.shape[-1]
    if dim != 3 * hv * key_dim or processed_k.shape != (batch, hv, tree_width, key_dim):
        raise ValueError("only equal Q/K/V head geometry is supported")
    if conv_weight.shape != (dim, 4) or conv_state.shape[-1] != 3:
        raise ValueError("only conv width 4 is supported")
    if processed_decay.shape != processed_k.shape or processed_beta.shape != (
        batch,
        hv,
        tree_width,
    ):
        raise ValueError("processed transform shapes do not match")
    if retrieve_parent_token.shape != (batch, tree_width):
        raise ValueError("retrieve_parent_token shape mismatch")
    cache_tree_width = tree_rawv.shape[-2]
    if cache_tree_width < tree_width:
        raise ValueError("tree factor cache does not cover logical tree width")
    if x.dtype != torch.bfloat16 or conv_state.dtype != torch.bfloat16:
        raise ValueError("the fused kernel requires BF16 activations and conv state")
    if conv_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("the fused kernel requires BF16 or FP32 conv weights")
    if conv_bias is not None and conv_bias.dtype != torch.bfloat16:
        raise ValueError("the fused kernel requires a BF16 conv bias")
    out = torch.empty_like(x)
    bk = triton.next_power_of_2(key_dim)
    _fused_kda_conv_tree_transform_kernel[(batch, tree_width, hv)](
        x,
        conv_weight,
        conv_bias,
        conv_state,
        conv_state_indices,
        intermediate_conv_window,
        factor_cache_indices,
        retrieve_parent_token,
        a,
        b,
        A_log,
        dt_bias,
        out,
        processed_k,
        processed_decay,
        processed_beta,
        tree_rawv,
        tree_rawk,
        tree_g,
        tree_beta,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        intermediate_conv_window.stride(3),
        retrieve_parent_token.stride(0),
        retrieve_parent_token.stride(1),
        (a.stride(1) if a.ndim == 4 else a.stride(-2)),
        b.stride(-2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        tree_rawv.stride(0),
        tree_rawk.stride(0),
        tree_g.stride(0),
        tree_beta.stride(0),
        SOFTPLUS_BETA=softplus_beta,
        SOFTPLUS_THRESHOLD=softplus_threshold,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        TREE_WIDTH=tree_width,
        CACHE_TREE_WIDTH=cache_tree_width,
        HV=hv,
        K=key_dim,
        BK=bk,
        USE_LOWER_BOUND=lower_bound is not None,
        CACHE_TREE_FACTORS=True,
        HAS_BIAS=conv_bias is not None,
        num_warps=2,
        num_stages=1,
    )
    return out
