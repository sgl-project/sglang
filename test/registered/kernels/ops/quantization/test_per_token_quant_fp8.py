import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="nightly", runner_config="1-gpu-large")


PER_TOKEN_QUANT_CASES = get_ci_test_range(
    list(
        itertools.product(
            [1, 39, 128, 512, 1392, 7807],
            [512, 1076, 1368, 1536, 2048, 4096],
        )
    ),
    [(1, 512), (39, 1536), (128, 1076), (1392, 1368), (7807, 1536)],
)


def _reference(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    values = input.float() * scale.reciprocal()
    return values.clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens,hidden_dim", PER_TOKEN_QUANT_CASES)
def test_per_token_quant_fp8(dtype, num_tokens, hidden_dim):
    input = torch.rand((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")

    per_token_quant_fp8(input, output, scale)

    expected_scale = input.float().abs().amax(dim=1, keepdim=True) / 448.0
    torch.testing.assert_close(scale, expected_scale, rtol=1e-5, atol=1e-7)
    expected = _reference(input, scale).float()
    if dtype == torch.float32:
        # Fast reciprocal multiplication can move exact FP8 midpoint values to
        # the adjacent representable value. It must remain a rare, one-bin tie.
        mismatch = output.float() != expected
        assert mismatch.count_nonzero() <= max(1, output.numel() // 100_000)
        torch.testing.assert_close(
            output.float()[mismatch], expected[mismatch], rtol=0.13, atol=1e-3
        )
    else:
        torch.testing.assert_close(output.float(), expected, rtol=1e-3, atol=1e-3)


def test_per_token_quant_fp8_zero_rows():
    """A zero token must stay zero instead of producing FP8 NaNs from 0 / 0."""
    input = torch.zeros((2048, 512), dtype=torch.float16, device="cuda")
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = torch.empty((2048, 1), dtype=torch.float32, device="cuda")

    per_token_quant_fp8(input, output, scale)

    assert torch.count_nonzero(scale) == 0
    assert torch.count_nonzero(output.float()) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
