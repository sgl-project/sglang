import sys
import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")


def _reference(mixed_qkvz, mixed_ba, nq, nv, head_qk, head_v):
    """Contiguous [all_q|all_k|all_v|all_z] / [all_b|all_a] split, mirroring
    Qwen3_5GatedDeltaNet.fix_query_key_value_ordering plus the fused path's
    cat/reshape."""
    k_tp = nq * head_qk
    v_tp = nv * head_v
    q, k, v, z = mixed_qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
    b, a = mixed_ba.split([nv, nv], dim=-1)
    mixed_qkv = torch.cat([q, k, v], dim=-1).contiguous()
    return (
        mixed_qkv,
        z.reshape(z.size(0), nv, head_v).contiguous(),
        b.contiguous(),
        a.contiguous(),
    )


@pytest.mark.parametrize(
    "nq,nv",
    [
        (4, 12),  # qwen4-exp TP4: head-group ratio 3 (non-power-of-2)
        (4, 8),  # ratio-2 regression
        (16, 48),  # qwen4-exp TP1: ratio 3 at full width
    ],
)
@pytest.mark.parametrize("batch", [1, 4, 33])
def test_contiguous_split_matches_reference(nq, nv, batch):
    torch.manual_seed(0)
    head_qk = head_v = 128
    qkvz = torch.randn(
        batch, 2 * nq * head_qk + 2 * nv * head_v, dtype=torch.bfloat16, device="cuda"
    )
    ba = torch.randn(batch, 2 * nv, dtype=torch.bfloat16, device="cuda")
    got = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, nq, nv, head_qk, head_v)
    ref = _reference(qkvz, ba, nq, nv, head_qk, head_v)
    for name, g, r in zip(("qkv", "z", "b", "a"), got, ref):
        assert g.shape == r.shape, name
        assert torch.equal(g, r), name


@pytest.mark.parametrize("batch", [1, 4, 33])
def test_strided_views_from_fused_in_proj_output(batch):
    """The merged in_proj GEMM emits one [batch, qkvz+ba] tensor whose qkvz/ba
    column views share a row stride wider than their own width; the split
    kernel must honor those strides instead of assuming contiguity."""
    torch.manual_seed(0)
    nq, nv, head_qk, head_v = 4, 12, 128, 128
    qkvz_width = 2 * nq * head_qk + 2 * nv * head_v
    ba_width = 2 * nv
    fused = torch.randn(
        batch, qkvz_width + ba_width, dtype=torch.bfloat16, device="cuda"
    )
    qkvz, ba = fused[:, :qkvz_width], fused[:, qkvz_width:]
    got = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, nq, nv, head_qk, head_v)
    ref = _reference(qkvz.contiguous(), ba.contiguous(), nq, nv, head_qk, head_v)
    for name, g, r in zip(("qkv", "z", "b", "a"), got, ref):
        assert g.shape == r.shape, name
        assert torch.equal(g, r), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
