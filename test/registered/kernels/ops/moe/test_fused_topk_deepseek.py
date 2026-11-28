import sys

import pytest
import torch

from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu, biased_grouped_topk_impl
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="nightly", runner_config="1-gpu-small")


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


@pytest.mark.parametrize(
    "impl_name,routed_scaling_factor,apply_routed_scaling_factor_on_output",
    [
        ("grouped_topk_gpu", None, False),
        ("grouped_topk_gpu", 2.5, False),
        ("grouped_topk_gpu", 2.5, True),
        ("biased_topk_impl", None, False),
        ("biased_topk_impl", 2.5, False),
        ("biased_topk_impl", 2.5, True),
        ("biased_grouped_topk_impl", None, False),
        ("biased_grouped_topk_impl", 2.5, False),
        ("biased_grouped_topk_impl", 2.5, True),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(
    "device_type", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_grouped_topk_with_fused_shared_experts(
    impl_name,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
    renormalize,
    device_type,
):
    """
    Test grouped_topk_gpu, biased_topk_impl, and biased_grouped_topk_impl
    when num_fused_shared_experts > 0.
    Ensures:
    1. Routed experts are distinct, normalized/scaled, and respect routed_scaling_factor.
    2. Shared expert IDs and weights are appended deterministically (no random indices).
    """
    from sglang.srt.layers.moe.topk import biased_topk_impl, grouped_topk_gpu

    torch.manual_seed(0)
    num_experts = 64
    num_fused_shared_experts = 2
    topk = 8  # 6 routed + 2 shared
    num_routed = topk - num_fused_shared_experts
    num_tokens = 16
    device = torch.device(device_type)

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.float32, device=device)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device)

    dispatch = {
        "grouped_topk_gpu": (
            grouped_topk_gpu,
            {
                "num_expert_group": 4,
                "topk_group": 2,
                "apply_routed_scaling_factor_on_output": (
                    apply_routed_scaling_factor_on_output
                ),
            },
        ),
        "biased_topk_impl": (
            biased_topk_impl,
            {
                "correction_bias": correction_bias,
                "apply_routed_scaling_factor_on_output": (
                    apply_routed_scaling_factor_on_output
                ),
            },
        ),
        "biased_grouped_topk_impl": (
            biased_grouped_topk_impl,
            {
                "correction_bias": correction_bias,
                "num_expert_group": 4,
                "topk_group": 2,
                "apply_routed_scaling_factor_on_output": (
                    apply_routed_scaling_factor_on_output
                ),
            },
        ),
    }
    fn, extra_kwargs = dispatch[impl_name]
    weights, ids = fn(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        **extra_kwargs,
    )

    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)

    # Validate routed expert IDs stay distinct and within routed range [0, num_experts).
    routed_ids = ids[:, :num_routed]
    assert ((routed_ids >= 0) & (routed_ids < num_experts)).all()
    for token_idx in range(num_tokens):
        assert len(torch.unique(routed_ids[token_idx])) == num_routed

    routed_weights = weights[:, :num_routed]
    if renormalize:
        expected_routed_sum = (
            routed_scaling_factor
            if (
                routed_scaling_factor is not None
                and apply_routed_scaling_factor_on_output
            )
            else 1.0
        )
        assert torch.allclose(
            routed_weights.sum(dim=-1),
            torch.full(
                (num_tokens,), expected_routed_sum, dtype=weights.dtype, device=device
            ),
            atol=1e-5,
        )
    else:
        raw_scores = (
            torch.softmax(gating_output, dim=-1)
            if impl_name == "grouped_topk_gpu"
            else gating_output.sigmoid()
        )
        expected_routed_weights = raw_scores.gather(1, routed_ids.long())
        if routed_scaling_factor is not None and apply_routed_scaling_factor_on_output:
            expected_routed_weights = expected_routed_weights * routed_scaling_factor
        assert torch.allclose(routed_weights, expected_routed_weights, atol=1e-5)

    # Validate shared expert IDs are [num_experts, num_experts + 1].
    expected_shared_ids = (
        torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            dtype=torch.int32,
            device=device,
        )
        .unsqueeze(0)
        .expand(num_tokens, -1)
    )
    assert torch.equal(ids[:, num_routed:], expected_shared_ids)

    expected_shared_weight = (
        1.0 / routed_scaling_factor
        if (
            routed_scaling_factor is not None
            and not apply_routed_scaling_factor_on_output
        )
        else 1.0
    )
    expected_shared_weights = torch.full(
        (num_tokens, num_fused_shared_experts),
        expected_shared_weight,
        dtype=weights.dtype,
        device=device,
    )
    assert torch.allclose(weights[:, num_routed:], expected_shared_weights, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
