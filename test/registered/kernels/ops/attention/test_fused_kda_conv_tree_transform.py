"""Tests for the fused UBTree causal-conv + KDA transform."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.linear.fused_kda_conv_tree_transform import (
    fused_kda_conv_tree_transform,
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


@pytest.mark.parametrize("heads,nodes", [(8, 32), (16, 49)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_fused_conv_tree_transform_matches_reference(
    heads, nodes, weight_dtype, lower_bound
):
    torch.manual_seed(20260918 + nodes)
    device = "cuda"
    head_dim = 128
    model_dim = 3 * heads * head_dim
    x = torch.randn(1, model_dim, nodes, device=device, dtype=torch.bfloat16)
    weight = torch.randn(model_dim, 4, device=device, dtype=weight_dtype) * 0.1
    state = torch.randn(1, model_dim, 3, device=device, dtype=torch.bfloat16)
    state_indices = torch.zeros(1, dtype=torch.int32, device=device)
    parent_row = [-1] + [(i - 1) // 4 for i in range(1, nodes)]
    parents = torch.tensor([parent_row], dtype=torch.int32, device=device)
    inter_ref = torch.empty(1, nodes, model_dim, 3, device=device, dtype=torch.bfloat16)
    inter_fused = torch.empty_like(inter_ref)
    a = torch.randn(1, nodes, heads, head_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(1, nodes, heads, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, head_dim, device=device, dtype=torch.bfloat16)

    out_ref = causal_conv1d_update(
        x,
        state,
        weight,
        None,
        activation="silu",
        conv_state_indices=state_indices,
        intermediate_conv_window=inter_ref,
        intermediate_state_indices=state_indices,
        retrieve_next_token=torch.zeros_like(parents),
        retrieve_parent_token=parents,
        direct_parent=True,
        save_tree_raw_only=True,
    )
    pk, pd, pb, rv, rk, g, beta = _outputs(1, heads, nodes, head_dim, device)
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
        processed_k=pk,
        processed_decay=pd,
        processed_beta=pb,
        factor_cache_indices=state_indices,
        tree_rawv=rv,
        tree_rawk=rk,
        tree_g=g,
        tree_beta=beta,
        lower_bound=lower_bound,
    )

    if weight_dtype == torch.float32:
        # FP32 production weights sum the four taps in causal_conv1d_update's order.
        assert torch.equal(out_fused, out_ref)
    else:
        # BF16 weights reorder the four-tap sum, so one-ULP gaps are expected.
        torch.testing.assert_close(out_fused, out_ref, rtol=1e-2, atol=1e-2)
    assert torch.equal(inter_fused[..., 2], inter_ref[..., 2])

    _, k, v = out_fused[0].t().reshape(nodes, 3, heads, head_dim).unbind(1)
    assert torch.equal(rk[0], k.transpose(0, 1))
    assert torch.equal(rv[0], v.transpose(0, 1))
    k = k.double()
    gate_input = a[0].double() + dt_bias.double()
    if lower_bound is None:
        gate = -A_log.double().exp()[:, None] * F.softplus(gate_input)
    else:
        gate = lower_bound * torch.sigmoid(A_log.double().exp()[:, None] * gate_input)
    beta_ref = torch.sigmoid(b[0].double())
    for name, got, expected in (
        ("processed_k", pk, k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)),
        ("processed_decay", pd, gate.exp()),
        ("processed_beta", pb, beta_ref),
        ("g", g, gate),
        ("beta", beta, beta_ref),
    ):
        torch.testing.assert_close(
            got[0].double(), expected.transpose(0, 1), rtol=1e-5, atol=1e-5, msg=name
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
