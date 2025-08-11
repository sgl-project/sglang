import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import dsv3_router_gemm


@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
@pytest.mark.parametrize("num_experts", [256, 384])
def test_dsv3_router_gemm(num_tokens, num_experts):
    hidden_dim = 7168

    mat_a = torch.randn(
        (num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    mat_b = torch.randn(
        (num_experts, hidden_dim), dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    bf16_ref = F.linear(mat_a, mat_b)
    float_ref = bf16_ref.to(torch.float32)

    bf16_output = dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.bfloat16)
    float_output = dsv3_router_gemm(mat_a, mat_b, out_dtype=torch.float32)

    assert torch.allclose(
        bf16_output, bf16_ref, rtol=1e-2, atol=1e-3
    ), "Router GEMM output in bf16 dtype mismatch with torch.nn.functional.linear reference"

    assert torch.allclose(
        float_output, float_ref, rtol=1e-2, atol=1e-3
    ), "Router GEMM output in float32 dtype mismatch with torch.nn.functional.linear reference"


if __name__ == "__main__":
    pytest.main([__file__])
