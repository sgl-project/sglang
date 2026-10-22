"""Bitwise regression tests for fused UBTree causal-conv + KDA transform."""

import pytest
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.linear.fused_kda_conv_tree_transform import (
    fused_kda_conv_tree_transform,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@triton.jit
def fused_sigmoid_gating_delta_rule_tree_transform_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    k,
    v,
    b,
    cu_seqlens,
    processed_k,
    processed_decay,
    processed_beta,
    factor_cache_indices,
    tree_rawv,
    tree_rawk,
    tree_g,
    tree_beta,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_a,
    stride_k,
    stride_v,
    stride_b,
    TREE_WIDTH: tl.constexpr,
    CACHE_TREE_WIDTH: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    CACHE_TREE_FACTORS: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Precompute node-local transforms once for all descendant factor walks."""
    i_nt, i_hv = tl.program_id(0), tl.program_id(1)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    i_n = i_nt // TREE_WIDTH
    node_idx = i_nt % TREE_WIDTH
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    active = node_idx < eos - bos
    token_idx = bos + node_idx
    o_k = tl.arange(0, BK)
    mask_k = o_k < K

    b_rawk = tl.load(
        k + token_idx * stride_k + i_h * K + o_k,
        mask=active & mask_k,
        other=0.0,
    ).to(tl.float32)
    b_k = b_rawk
    if USE_QK_L2NORM_IN_KERNEL:
        b_k /= tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_a = tl.load(
        a + token_idx * stride_a + i_hv * K + o_k,
        mask=active & mask_k,
        other=0.0,
    ).to(tl.float32)
    b_b = tl.load(
        b + token_idx * stride_b + i_hv,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    x = b_a + b_dt_bias
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
    else:
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
    b_decay = tl.exp(b_g)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    transform_base = (i_n * HV * TREE_WIDTH + i_hv * TREE_WIDTH + node_idx) * K
    tl.store(processed_k + transform_base + o_k, b_k, mask=active & mask_k)
    tl.store(processed_decay + transform_base + o_k, b_decay, mask=active & mask_k)
    tl.store(
        processed_beta + (i_n * HV + i_hv) * TREE_WIDTH + node_idx,
        b_beta,
        mask=active,
    )

    if CACHE_TREE_FACTORS:
        factor_slot = tl.load(factor_cache_indices + i_n).to(tl.int64)
        if factor_slot >= 0:
            b_rawv = tl.load(
                v + token_idx * stride_v + i_hv * V + o_k,
                mask=active & mask_k,
                other=0.0,
            )
            tl.store(
                tree_rawv
                + factor_slot * stride_rawv_slot
                + (i_hv * CACHE_TREE_WIDTH + node_idx) * V
                + o_k,
                b_rawv,
                mask=active & mask_k,
            )
            tl.store(
                tree_g
                + factor_slot * stride_g_slot
                + (i_hv * CACHE_TREE_WIDTH + node_idx) * K
                + o_k,
                b_g,
                mask=active & mask_k,
            )
            if i_hv % (HV // H) == 0:
                tl.store(
                    tree_rawk
                    + factor_slot * stride_rawk_slot
                    + (i_h * CACHE_TREE_WIDTH + node_idx) * K
                    + o_k,
                    b_rawk,
                    mask=active & mask_k,
                )
            tl.store(
                tree_beta
                + factor_slot * stride_beta_slot
                + i_hv * CACHE_TREE_WIDTH
                + node_idx,
                b_beta,
                mask=active,
            )


def _outputs(batch, heads, nodes, dim, device):
    pk = torch.empty(batch, heads, nodes, dim, device=device, dtype=torch.float32)
    pd = torch.empty_like(pk)
    pb = torch.empty(batch, heads, nodes, device=device, dtype=torch.float32)
    rv = torch.empty(batch, heads, nodes, dim, device=device, dtype=torch.bfloat16)
    rk = torch.empty_like(rv)
    g = torch.empty(batch, heads, nodes, dim, device=device, dtype=torch.float32)
    beta = torch.empty(batch, heads, nodes, device=device, dtype=torch.float32)
    return pk, pd, pb, rv, rk, g, beta


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("heads,nodes", [(8, 32), (16, 49)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_fused_conv_tree_transform_matches_reference(
    batch, heads, nodes, weight_dtype, lower_bound
):
    torch.manual_seed(20260918 + nodes)
    device = "cuda"
    head_dim, cache_nodes = 128, nodes
    model_dim = 3 * heads * head_dim
    x = torch.randn(batch, model_dim, nodes, device=device, dtype=torch.bfloat16)
    weight = torch.randn(model_dim, 4, device=device, dtype=weight_dtype) * 0.1
    state = torch.randn(batch, model_dim, 3, device=device, dtype=torch.bfloat16)
    state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    factor_indices = state_indices.clone()
    parent_row = [-1] + [(i - 1) // 4 for i in range(1, nodes)]
    parents = torch.tensor([parent_row] * batch, dtype=torch.int32, device=device)
    tree_marker = torch.zeros_like(parents)
    inter_ref = torch.empty(
        batch, nodes, model_dim, 3, device=device, dtype=torch.bfloat16
    )
    inter_fused = torch.empty_like(inter_ref)
    a = torch.randn(
        1, batch * nodes, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    b = torch.randn(1, batch * nodes, heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, head_dim, device=device, dtype=torch.bfloat16)
    cu = torch.arange(0, (batch + 1) * nodes, nodes, device=device, dtype=torch.int32)
    ref = _outputs(batch, heads, nodes, head_dim, device)
    fused = list(_outputs(batch, heads, nodes, head_dim, device))
    fused[3] = torch.full(
        (batch, heads, cache_nodes, head_dim),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    fused[4] = torch.full_like(fused[3], float("nan"))
    fused[5] = torch.full(
        (batch, heads, cache_nodes, head_dim),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )
    fused[6] = torch.full(
        (batch, heads, cache_nodes),
        float("nan"),
        device=device,
        dtype=torch.float32,
    )

    out_ref = causal_conv1d_update(
        x,
        state,
        weight,
        None,
        activation="silu",
        conv_state_indices=state_indices,
        intermediate_conv_window=inter_ref,
        intermediate_state_indices=factor_indices,
        retrieve_next_token=tree_marker,
        retrieve_parent_token=parents,
        direct_parent=True,
        save_tree_raw_only=True,
    )
    flat = out_ref.transpose(1, 2).reshape(batch * nodes, model_dim).contiguous()
    _, k, v = flat.split([heads * head_dim] * 3, dim=-1)
    pk, pd, pb, rv, rk, g, beta = ref
    fused_sigmoid_gating_delta_rule_tree_transform_kernel[(batch * nodes, heads)](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        lower_bound=0.0 if lower_bound is None else lower_bound,
        k=k.view(1, batch * nodes, heads, head_dim),
        v=v.view(1, batch * nodes, heads, head_dim),
        b=b,
        cu_seqlens=cu,
        processed_k=pk,
        processed_decay=pd,
        processed_beta=pb,
        factor_cache_indices=factor_indices,
        tree_rawv=rv,
        tree_rawk=rk,
        tree_g=g,
        tree_beta=beta,
        stride_rawv_slot=rv.stride(0),
        stride_rawk_slot=rk.stride(0),
        stride_g_slot=g.stride(0),
        stride_beta_slot=beta.stride(0),
        stride_a=a.stride(-3),
        stride_k=k.view(1, batch * nodes, heads, head_dim).stride(-3),
        stride_v=v.view(1, batch * nodes, heads, head_dim).stride(-3),
        stride_b=b.stride(-2),
        TREE_WIDTH=nodes,
        CACHE_TREE_WIDTH=nodes,
        H=heads,
        HV=heads,
        K=head_dim,
        V=head_dim,
        BK=head_dim,
        USE_QK_L2NORM_IN_KERNEL=True,
        USE_LOWER_BOUND=lower_bound is not None,
        CACHE_TREE_FACTORS=True,
        num_warps=2,
        num_stages=1,
    )

    fpk, fpd, fpb, frv, frk, fg, fbeta = fused
    out_fused = fused_kda_conv_tree_transform(
        x=x,
        conv_state=state,
        conv_weight=weight,
        conv_bias=None,
        conv_state_indices=state_indices,
        intermediate_conv_window=inter_fused,
        retrieve_parent_token=parents,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        processed_k=fpk,
        processed_decay=fpd,
        processed_beta=fpb,
        factor_cache_indices=factor_indices,
        tree_rawv=frv,
        tree_rawk=frk,
        tree_g=fg,
        tree_beta=fbeta,
        lower_bound=lower_bound,
    )
    torch.cuda.synchronize()

    pairs = {
        "out": (out_fused, out_ref),
        "inter_raw": (inter_fused[..., 2], inter_ref[..., 2]),
        "processed_k": (fused[0], ref[0]),
        "processed_decay": (fused[1], ref[1]),
        "processed_beta": (fused[2], ref[2]),
        "rawv": (fused[3][:, :, :nodes], ref[3]),
        "rawk": (fused[4][:, :, :nodes], ref[4]),
        "g": (fused[5][:, :, :nodes], ref[5]),
        "beta": (fused[6][:, :, :nodes], ref[6]),
    }
    if cache_nodes > nodes:
        for padded in fused[3:]:
            assert torch.isnan(padded[:, :, nodes:]).all()

    for name, (got, expected) in pairs.items():
        if weight_dtype == torch.float32:
            # Production Ling KDA weights are FP32. Keep this path bitwise exact:
            # changing the transform launch from two warps changes the L2-reduction
            # order and is expected to fail this assertion.
            assert torch.equal(got, expected), name
        else:
            # The BF16-weight compatibility path accumulates the four-tap conv in
            # a different order from causal_conv1d_update, so one-BF16-ULP output
            # differences are expected while the result remains numerically equal.
            torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2, msg=name)
