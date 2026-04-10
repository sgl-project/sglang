import sys

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate_ungrouped import moe_fused_gate_ungrouped
from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "num_experts,topk,routed_scaling_factor",
    [
        (384, 6, 2.872),  # Kimi K2: 384 experts, topk=6
        (256, 8, 2.5),  # GLM5: 256 experts, topk=8
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_moe_fused_gate_ungrouped(
    seq_length,
    num_experts,
    topk,
    routed_scaling_factor,
    dtype,
    apply_routed_scaling_factor_on_output,
):
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    output, indices = moe_fused_gate_ungrouped(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    ref_output, ref_indices = kimi_k2_biased_topk_impl(
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


@pytest.mark.parametrize("seq_length", [1, 64, 512, 1024, 4096])
@pytest.mark.parametrize(
    "num_experts,routed_topk,routed_scaling_factor",
    [
        (384, 6, 2.872),  # Kimi K2
        (256, 8, 2.5),  # GLM5
    ],
)
def test_moe_fused_gate_ungrouped_output_stride(
    seq_length, num_experts, routed_topk, routed_scaling_factor
):
    """Test output_stride: kernel writes routed_topk columns into a wider pre-allocated tensor."""
    dtype = torch.float32
    renormalize = True
    total_topk = routed_topk + 1  # 1 extra slot for shared expert

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    # Pre-allocate wider tensors
    output = torch.empty((seq_length, total_topk), dtype=torch.float32, device="cuda")
    indices = torch.empty((seq_length, total_topk), dtype=torch.int32, device="cuda")

    moe_fused_gate_ungrouped(
        tensor,
        bias,
        topk=routed_topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
        output=output,
        indices=indices,
    )

    # Verify routed columns match reference
    ref_output, ref_indices = kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=routed_topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    routed_output = output[:, :routed_topk]
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        routed_output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )
    assert output_check, "Routed weights mismatch with output_stride"

    assert output.shape == (seq_length, total_topk)
    assert indices.shape == (seq_length, total_topk)


@pytest.mark.parametrize("seq_length", [1024, 4096])
@pytest.mark.parametrize(
    "num_experts,topk,routed_scaling_factor",
    [
        (384, 6, 2.872),  # Kimi K2
        (256, 8, 2.5),  # GLM5
    ],
)
def test_moe_fused_gate_ungrouped_specific(
    seq_length, num_experts, topk, routed_scaling_factor
):
    """Test for supported configurations: 384 experts (Kimi K2) and 256 experts (GLM5)"""
    dtype = torch.float32
    renormalize = True

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    output, indices = moe_fused_gate_ungrouped(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    ref_output, ref_indices = kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=False,
    )

    assert output.shape == (seq_length, topk)
    assert indices.shape == (seq_length, topk)
    assert output.dtype == torch.float32
    assert indices.dtype == torch.int32

    if renormalize:
        weight_sums = output.sum(dim=-1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
        )

    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert output_check, "Output mismatch for specific case"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
