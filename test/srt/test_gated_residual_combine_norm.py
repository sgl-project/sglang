import torch
import pytest

from sglang.kernels.ops.layernorm.gated_residual_combine_norm import (
    gated_residual_combine_norm,
)
from sglang.kernels.ops.elementwise.hc_combine import hc_combine


def reference_combine_norm(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
):
    """Reference: separate combine + grouped RMSNorm."""
    hc_count = inject_logits.shape[-1]
    hidden_size = block_output.shape[-1]

    # Combine
    combined = hc_combine(
        block_output,
        residual,
        inject_logits,
        hc_count,
        hidden_size,
    )

    # Grouped RMSNorm
    combined_reshaped = combined.reshape(-1, group_size)
    variance = combined_reshaped.pow(2).mean(dim=-1, keepdim=True)
    normed = combined_reshaped * torch.rsqrt(variance + eps)
    normed = normed.reshape(combined.shape)
    normed = normed * (1.0 + weight)

    return combined, normed


@pytest.mark.parametrize("hc_count", [4])
@pytest.mark.parametrize("hidden_size", [2560])
@pytest.mark.parametrize("group_size", [512, 2560])
@pytest.mark.parametrize("num_tokens", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gated_residual_combine_norm_matches_reference(
    hc_count: int,
    hidden_size: int,
    group_size: int,
    num_tokens: int,
    dtype: torch.dtype,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(42)

    block_output = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    residual = torch.randn(
        num_tokens, hc_count * hidden_size, dtype=dtype, device=device
    )
    inject_logits = torch.randn(
        num_tokens, hc_count, dtype=dtype, device=device
    )
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Fused kernel
    combined_fused, normed_fused = gated_residual_combine_norm(
        block_output,
        residual,
        inject_logits,
        weight,
        group_size,
    )

    # Reference
    combined_ref, normed_ref = reference_combine_norm(
        block_output,
        residual,
        inject_logits,
        weight,
        group_size,
    )

    # Compare
    torch.testing.assert_close(
        combined_fused, combined_ref, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        normed_fused, normed_ref, rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("hc_count", [4])
@pytest.mark.parametrize("hidden_size", [2560])
@pytest.mark.parametrize("num_tokens", [4])
def test_gated_residual_combine_norm_zero_tokens(
    hc_count: int,
    hidden_size: int,
    num_tokens: int,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    block_output = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    residual = torch.randn(
        num_tokens, hc_count * hidden_size, dtype=dtype, device=device
    )
    inject_logits = torch.randn(
        num_tokens, hc_count, dtype=dtype, device=device
    )
    weight = torch.randn(hidden_size, dtype=dtype, device=device)

    # Zero tokens
    block_output_0 = block_output[:0]
    residual_0 = residual[:0]
    inject_logits_0 = inject_logits[:0]

    combined, normed = gated_residual_combine_norm(
        block_output_0,
        residual_0,
        inject_logits_0,
        weight,
        group_size=512,
    )

    assert combined.shape == (0, hc_count * hidden_size)
    assert normed.shape == (0, hc_count * hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
