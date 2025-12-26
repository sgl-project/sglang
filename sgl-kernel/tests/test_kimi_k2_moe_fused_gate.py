import pytest
import torch
from sgl_kernel import kimi_k2_moe_fused_gate

from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize("topk", [6])  # Kimi K2 uses topk=6
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_kimi_k2_moe_fused_gate(
    seq_length, topk, dtype, apply_routed_scaling_factor_on_output
):
    num_experts = 384  # Kimi K2: only support 384 experts
    renormalize = True
    routed_scaling_factor = 2.872  # Kimi K2's routed scaling factor

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    # Test our fused kernel
    output, indices = kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Reference implementation
    ref_output, ref_indices = kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Check weights match (after sorting)
    # Weights are the most important - they determine the actual MoE output
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


@pytest.mark.parametrize("seq_length", [1024, 4096])
@pytest.mark.parametrize("num_experts", [384])
@pytest.mark.parametrize("topk", [6])
def test_kimi_k2_specific_case(seq_length, num_experts, topk):
    """Test specifically for Kimi K2 configuration: 384 experts, topk=6"""
    dtype = torch.float32
    renormalize = True
    routed_scaling_factor = 2.872

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    output, indices = kimi_k2_moe_fused_gate(
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

    # Verify output shapes
    assert output.shape == (seq_length, topk)
    assert indices.shape == (seq_length, topk)
    assert output.dtype == torch.float32
    assert indices.dtype == torch.int32

    # Verify weights are normalized (sum to 1 per token if renormalize=True)
    if renormalize:
        weight_sums = output.sum(dim=-1)
        assert torch.allclose(
            weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
        )

    # Check weights match (after sorting)
    # Weights are the most important - they determine the actual MoE output
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-02,
        atol=1e-03,
    )

    assert output_check, f"Output mismatch for Kimi K2 specific case"


if __name__ == "__main__":
    pytest.main([__file__])
