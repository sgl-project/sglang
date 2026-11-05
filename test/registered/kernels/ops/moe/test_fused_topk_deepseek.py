import sys

import pytest
import torch

from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu, biased_grouped_topk_impl
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="nightly", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),  # 128 experts configuration
        (256, 8, 4, 8),  # DeepSeek V3 config - most important to test
        (64, 2, 2, 4),  # Smaller configuration
    ],
)
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_fused_topk_deepseek(seq_length, params, apply_routed_scaling_factor_on_output):
    """
    Test the fused_topk_deepseek code path in biased_grouped_topk_gpu.
    """
    num_experts, num_expert_group, topk_group, topk = params
    dtype = torch.float32
    device = get_device()

    torch.manual_seed(seq_length)
    hidden_states = torch.randn(seq_length, 128, dtype=dtype, device=device)
    gating_output = torch.randn(seq_length, num_experts, dtype=dtype, device=device)
    correction_bias = torch.randn(num_experts, dtype=dtype, device=device)

    routed_scaling_factor = 2.5 if apply_routed_scaling_factor_on_output else None

    # Fused implementation (uses fused_topk_deepseek when conditions are met)
    output, indices = biased_grouped_topk_gpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Reference implementation (pure PyTorch)
    ref_output, ref_indices = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Check 1: Row-wise sums should match (invariant to tie-breaking)
    output_sum = output.sum(dim=-1)
    ref_output_sum = ref_output.sum(dim=-1)
    sum_check = torch.allclose(output_sum, ref_output_sum, rtol=1e-03, atol=1e-04)

    # Check 2: Scatter-based comparison with allowance for tie-breaking
    res = torch.zeros(seq_length, num_experts, dtype=torch.float32, device=device)
    ref = torch.zeros(seq_length, num_experts, dtype=torch.float32, device=device)

    res.scatter_(1, indices.long(), output)
    ref.scatter_(1, ref_indices.long(), ref_output)

    diff = torch.abs(ref - res)
    atol = (
        5e-03
        if (seq_length >= 4096 and apply_routed_scaling_factor_on_output)
        else 1e-03
    )
    num_large_diffs = (diff > atol).sum().item()

    # Allow a small number of differences for tie-breaking situations
    max_allowed_diffs = max(16, seq_length // 500)
    scatter_check = num_large_diffs <= max_allowed_diffs

    assert sum_check and scatter_check, (
        f"Output mismatch at seq_length {seq_length}, params {params}, "
        f"apply_routed_scaling_factor_on_output {apply_routed_scaling_factor_on_output}"
    )


def test_grouped_topk_with_fused_shared_experts():
    """
    Test grouped_topk_gpu and biased_grouped_topk_impl when num_fused_shared_experts > 0.
    Ensures:
    1. Routed experts are correctly selected and normalized.
    2. Shared expert IDs and weights are appended deterministically (no random indices).
    """
    from sglang.srt.layers.moe.topk import grouped_topk_gpu

    num_experts = 64
    num_fused_shared_experts = 2
    topk = 8  # 6 routed + 2 shared
    num_tokens = 16
    device = torch.device("cpu")

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)

    weights, ids = grouped_topk_gpu(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
        num_fused_shared_experts=num_fused_shared_experts,
    )

    # Validate output shapes
    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)

    # Validate routed weights sum to 1.0 (renormalized)
    routed_weights = weights[:, : topk - num_fused_shared_experts]
    assert torch.allclose(routed_weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)

    # Validate shared expert IDs are [num_experts, num_experts + 1]
    expected_shared_ids = (
        torch.arange(
            num_experts, num_experts + num_fused_shared_experts, dtype=torch.int32
        )
        .unsqueeze(0)
        .expand(num_tokens, -1)
    )
    assert torch.equal(ids[:, topk - num_fused_shared_experts :], expected_shared_ids)

    # Validate shared expert weights are 1.0
    expected_shared_weights = torch.ones(
        (num_tokens, num_fused_shared_experts), dtype=torch.float32
    )
    assert torch.equal(
        weights[:, topk - num_fused_shared_experts :], expected_shared_weights
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
