import sys
from typing import Optional

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate_ungrouped import moe_fused_gate_ungrouped
from sglang.test.ci.ci_register import register_cuda_ci


def _reference_biased_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
    num_fused_shared_experts: int = 0,
):
    """Reference implementation mirroring biased_grouped_topk_impl (num_expert_group=1).
    Kept as a local copy so the reference is not affected by source code changes.
    """
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        tmp_scores,
        k=topk,
        dim=-1,
        sorted=(num_fused_shared_experts > 0),
    )
    topk_weights = scores.gather(1, topk_ids)

    # Fill shared expert slots (mirrors biased_grouped_topk_impl)
    if num_fused_shared_experts > 0:
        topk_ids[:, -num_fused_shared_experts:] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(num_token, num_fused_shared_experts),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            routed_topk = topk - num_fused_shared_experts
            topk_weights[:, -num_fused_shared_experts:] = (
                topk_weights[:, :routed_topk].sum(dim=-1, keepdim=True)
                / routed_scaling_factor
            )

    if renormalize:
        if num_fused_shared_experts == 0:
            topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        else:
            routed_topk = topk - num_fused_shared_experts
            topk_weights_sum = topk_weights[:, :routed_topk].sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    return topk_weights, topk_ids


register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")


def _call_kernel(
    tensor,
    bias,
    topk,
    renormalize,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
    num_fused_shared_experts=0,
    output=None,
    indices=None,
):
    """Helper mirroring _biased_grouped_topk_ungrouped: kernel + shared expert fill."""
    num_rows = tensor.size(0)
    num_experts = tensor.size(1)
    routed_topk = topk - num_fused_shared_experts

    if output is None:
        output = torch.empty(
            (num_rows, topk), dtype=torch.float32, device=tensor.device
        )
    if indices is None:
        indices = torch.empty((num_rows, topk), dtype=torch.int32, device=tensor.device)

    moe_fused_gate_ungrouped(
        tensor,
        bias,
        routed_topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
        output,
        indices,
    )

    # Fill shared expert slots (mirrors _biased_grouped_topk_ungrouped)
    if num_fused_shared_experts > 0:
        sf = routed_scaling_factor if routed_scaling_factor is not None else 1.0
        output[:, routed_topk:] = output[:, :routed_topk].sum(dim=-1, keepdim=True) / sf
        indices[:, routed_topk:] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(num_rows, num_fused_shared_experts),
            dtype=torch.int32,
            device=tensor.device,
        )

    return output, indices


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "num_experts,topk,routed_scaling_factor",
    [
        (384, 6, 2.827),  # Kimi K2: 384 experts, topk=6
        (256, 8, 2.5),  # GLM5: 256 experts, topk=8
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_moe_fused_gate_ungrouped(
    seq_length,
    num_experts,
    topk,
    routed_scaling_factor,
    dtype,
    apply_routed_scaling_factor_on_output,
    renormalize,
):

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    output, indices = _call_kernel(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    ref_output, ref_indices = _reference_biased_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"num_experts {num_experts}, topk {topk}, "
        f"apply_routed_scaling_factor_on_output {apply_routed_scaling_factor_on_output}"
    )

    # After renormalization, weights should sum to ~1.0
    if renormalize and not apply_routed_scaling_factor_on_output:
        weight_sums = output.sum(dim=-1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
        ), f"Weight sums mismatch at seq_length {seq_length}"


@pytest.mark.parametrize("seq_length", [1, 64, 512, 513, 515, 1024, 4096])
@pytest.mark.parametrize(
    "num_experts,topk,routed_scaling_factor",
    [
        (128, 5, 1.0),  # Minimal: 128 experts, topk=4+1 shared
        (384, 7, 2.827),  # Kimi K2: 384 experts, topk=6+1 shared
        (256, 8, 2.5),  # GLM5: 256 experts, topk=7+1 shared (topk<=8 constraint)
    ],
)
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_moe_fused_gate_ungrouped_shared_experts(
    seq_length,
    num_experts,
    topk,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
    renormalize,
):
    """End-to-end test mirroring _biased_grouped_topk_ungrouped vs biased_grouped_topk_impl
    with num_fused_shared_experts=1."""
    dtype = torch.float32
    num_fused_shared_experts = 1
    routed_topk = topk - num_fused_shared_experts

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    # Kernel path (mirrors _biased_grouped_topk_ungrouped)
    output, indices = _call_kernel(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        num_fused_shared_experts=num_fused_shared_experts,
    )

    # Reference path (mirrors biased_grouped_topk_impl)
    ref_output, ref_indices = _reference_biased_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        num_fused_shared_experts=num_fused_shared_experts,
    )

    assert output.shape == (seq_length, topk)
    assert indices.shape == (seq_length, topk)

    # Compare routed weights (shared expert IDs are random, so compare weights only)
    routed_output = output[:, :routed_topk]
    ref_routed_output = ref_output[:, :routed_topk]
    assert torch.allclose(
        ref_routed_output.sort()[0],
        routed_output.sort()[0],
        rtol=1e-02,
        atol=1e-03,
    ), "Routed weights mismatch"

    # Compare shared expert weights
    shared_weights = output[:, routed_topk:]
    ref_shared_weights = ref_output[:, routed_topk:]
    assert torch.allclose(
        shared_weights,
        ref_shared_weights,
        rtol=1e-03,
        atol=1e-04,
    ), f"Shared expert weight mismatch: kernel={shared_weights[0].item():.6f}, ref={ref_shared_weights[0].item():.6f}"

    # Verify shared expert IDs are in valid range
    shared_ids = indices[:, routed_topk:]
    assert (shared_ids >= num_experts).all(), "Shared expert IDs must be >= num_experts"
    assert (
        shared_ids < num_experts + num_fused_shared_experts
    ).all(), "Shared expert IDs must be < num_experts + num_fused_shared_experts"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
