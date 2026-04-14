import sys

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=300, suite="nightly-kernel-1-gpu", nightly=True)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation
# ---------------------------------------------------------------------------


def biased_grouped_topk_ref(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
):
    """Pure-PyTorch reference for biased grouped top-k MoE routing."""
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    # Group scores: sum top-2 within each group
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [num_token, num_expert_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )

    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    topk_excl = topk - num_fused_shared_experts
    _, routed_topk_ids = torch.topk(tmp_scores, k=topk_excl, dim=-1, sorted=False)
    routed_topk_weights = scores.gather(1, routed_topk_ids)

    if num_fused_shared_experts > 0:
        topk_ids = torch.empty(
            (num_token, topk),
            dtype=routed_topk_ids.dtype,
            device=routed_topk_ids.device,
        )
        topk_weights = torch.empty(
            (num_token, topk),
            dtype=routed_topk_weights.dtype,
            device=routed_topk_weights.device,
        )
        topk_ids[:, :topk_excl] = routed_topk_ids
        topk_weights[:, :topk_excl] = routed_topk_weights

        scale = float(routed_scaling_factor)
        routed_sum = routed_topk_weights.sum(dim=-1, keepdim=True)

        for i in range(num_fused_shared_experts):
            topk_ids[:, topk_excl + i] = num_experts + i
            topk_weights[:, topk_excl + i] = routed_sum[:, 0] / scale
    else:
        topk_ids = routed_topk_ids
        topk_weights = routed_topk_weights

    if renormalize:
        if num_fused_shared_experts > 0:
            topk_weights_sum = topk_weights[:, :topk_excl].sum(dim=-1, keepdim=True)
        else:
            topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

SEQ_LENGTHS = get_ci_test_range(
    full_range=list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    ci_range=list(range(1, 5)) + [16, 128, 1024, 4096],
)

MOE_PARAMS = [
    # (num_experts, num_expert_group, topk_group, topk_excl)
    (128, 4, 2, 4),
    (256, 8, 4, 8),  # DeepSeek V3 / R1
    (512, 16, 8, 16),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

NUM_FUSED_SHARED = [0, 1, 2]

APPLY_SCALING = [False, True]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("moe_params", MOE_PARAMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_fused_shared_experts", NUM_FUSED_SHARED)
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", APPLY_SCALING)
def test_moe_fused_gate(
    seq_length: int,
    moe_params,
    dtype: torch.dtype,
    num_fused_shared_experts: int,
    apply_routed_scaling_factor_on_output: bool,
):
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    num_experts, num_expert_group, topk_group, topk_excl = moe_params
    topk = topk_excl + num_fused_shared_experts
    routed_scaling_factor = 2.5

    torch.manual_seed(seq_length ^ (num_experts << 8))
    gating_output = torch.randn((seq_length, num_experts), dtype=dtype, device="cuda")
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    # JIT kernel
    jit_weights, jit_ids = moe_fused_gate(
        input=gating_output,
        bias=bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Reference
    ref_weights, ref_ids = biased_grouped_topk_ref(
        gating_output=gating_output,
        correction_bias=bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # For fused shared experts: only compare the routed (non-shared) columns
    # for exact index matching; shared slots just need valid range.
    if num_fused_shared_experts > 0:
        # Check shared expert index range
        shared_jit = jit_ids[:, topk_excl:]
        shared_ref = ref_ids[:, topk_excl:]
        assert torch.all(
            (shared_jit >= num_experts)
            & (shared_jit < num_experts + num_fused_shared_experts)
        ), (
            f"JIT shared expert indices out of range [{num_experts}, "
            f"{num_experts + num_fused_shared_experts}): {shared_jit}"
        )
        assert torch.all(
            (shared_ref >= num_experts)
            & (shared_ref < num_experts + num_fused_shared_experts)
        ), (
            f"Ref shared expert indices out of range [{num_experts}, "
            f"{num_experts + num_fused_shared_experts}): {shared_ref}"
        )

        # Compare routed columns only
        jit_ids_cmp = jit_ids[:, :topk_excl]
        ref_ids_cmp = ref_ids[:, :topk_excl]
    else:
        jit_ids_cmp = jit_ids
        ref_ids_cmp = ref_ids

    # Indices: sort each row before comparing (order within row can differ)
    idx_match = torch.allclose(
        jit_ids_cmp.sort(dim=-1)[0].to(torch.int32),
        ref_ids_cmp.sort(dim=-1)[0].to(torch.int32),
    )
    assert idx_match, (
        f"Expert index mismatch: seq={seq_length}, dtype={dtype}, params={moe_params}, "
        f"fused_shared={num_fused_shared_experts}, apply_scaling={apply_routed_scaling_factor_on_output}"
    )

    # Weights: sort and compare with tolerance
    weight_match = torch.allclose(
        jit_weights.sort(dim=-1)[0],
        ref_weights.sort(dim=-1)[0],
        rtol=1e-2,
        atol=1e-3,
    )
    assert weight_match, (
        f"Weight mismatch: seq={seq_length}, dtype={dtype}, params={moe_params}, "
        f"fused_shared={num_fused_shared_experts}, apply_scaling={apply_routed_scaling_factor_on_output}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
