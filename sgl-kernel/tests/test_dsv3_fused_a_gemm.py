import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import dsv3_fused_a_gemm


def is_sm90_or_sm100_supported(device=None) -> bool:
    # Kernel needs ~192KB smem; consumer Blackwell (sm12x) caps at ~100KB (see #29317).
    return (torch.cuda.get_device_capability(device)[0] in (9, 10)) and (
        torch.version.cuda >= "12.3"
    )


@pytest.mark.skipif(
    not is_sm90_or_sm100_supported(),
    reason="dsv3_fused_a_gemm is only supported on SM90/SM100",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 15, 16])
def test_dsv3_fused_a_gemm(num_tokens):
    kHdIn = 7168
    kHdOut = 2112

    mat_a = torch.randn(
        (num_tokens, kHdIn), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    mat_b = torch.randn((kHdOut, kHdIn), dtype=torch.bfloat16, device="cuda").transpose(
        0, 1
    )
    output = torch.empty(
        (num_tokens, kHdOut), dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    ref = F.linear(mat_a, mat_b.T)

    output = dsv3_fused_a_gemm(mat_a, mat_b)

    assert torch.allclose(
        output, ref, rtol=1e-2, atol=1e-3
    ), "Fused GEMM output mismatch with torch.nn.functional.linear reference"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
