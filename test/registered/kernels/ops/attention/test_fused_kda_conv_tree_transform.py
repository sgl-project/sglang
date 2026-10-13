"""Bitwise regression tests for fused UBTree causal-conv + KDA transform."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.fused_kda_conv_tree_transform import (
    fused_kda_conv_tree_transform,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_tree_transform_kernel,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
@pytest.mark.parametrize("nodes,cache_nodes", [(32, 32), (49, 49)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_fused_conv_tree_transform_matches_reference(
    batch, nodes, cache_nodes, weight_dtype
):
    torch.manual_seed(20260918 + nodes)
    device = "cuda"
    heads, head_dim = 16, 128
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
        lower_bound=0.0,
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
        USE_LOWER_BOUND=False,
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
        intermediate_state_indices=factor_indices,
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
