import sys

import pytest
import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    _can_use_flashinfer_fused_topk_deepseek,
    biased_grouped_topk_gpu,
    biased_grouped_topk_impl,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="nightly", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "params",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),
        (64, 2, 2, 4),
        (256, 1, 1, 8),
    ],
)
def test_flashinfer_fused_topk_deepseek_accepts_contract_shapes(params):
    assert _can_use_flashinfer_fused_topk_deepseek(*params)


@pytest.mark.parametrize(
    "params",
    [
        (256, 16, 4, 8),  # too many groups
        (256, 8, 5, 8),  # too many selected groups
        (8, 8, 4, 8),  # fewer than two experts per group
        (512, 8, 4, 8),  # too many grouped experts
        (256, 8, 4, 9),  # too many routed experts
        (96, 3, 2, 5),  # FlashInfer route requires power-of-two expert count
    ],
)
def test_flashinfer_fused_topk_deepseek_rejects_outside_contract(params):
    assert not _can_use_flashinfer_fused_topk_deepseek(*params)


@pytest.mark.parametrize(
    "params",
    [
        (256, 16, 4, 8),
        (256, 8, 5, 8),
        (8, 8, 4, 8),
    ],
)
def test_outside_contract_falls_back_without_calling_flashinfer(monkeypatch, params):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("out-of-contract shape called FlashInfer")

    monkeypatch.setattr(topk_module, "fused_topk_deepseek", fail_if_called)

    num_experts, num_expert_group, topk_group, topk = params
    device = get_device()
    torch.manual_seed(num_experts + num_expert_group + topk_group + topk)
    hidden_states = torch.randn(8, 128, dtype=torch.float32, device=device)
    gating_output = torch.randn(8, num_experts, dtype=torch.float32, device=device)
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device)

    output, indices = biased_grouped_topk_gpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )
    ref_output, ref_indices = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )

    result = torch.zeros(8, num_experts, dtype=torch.float32, device=device)
    reference = torch.zeros_like(result)
    result.scatter_(1, indices.long(), output)
    reference.scatter_(1, ref_indices.long(), ref_output)
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
