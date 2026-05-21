"""Tests for the softmax sampling kernel."""

import sys

import pytest
import torch

from sglang.jit_kernel.softmax import softmax_sampling
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)


DEVICE = "cuda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]

BATCH_SIZES = get_ci_test_range(
    full_range=[2**x for x in range(10)] + [2**x + 1 for x in range(1, 10)],
    ci_range=[1, 4, 7, 64, 128],
)
VOCAB_SIZES = get_ci_test_range(
    full_range=[1024, 4096, 8192, 32000, 32768, 65536, 128256, 151936, 262144],
    ci_range=[32000, 128256],
)
TEMPERATURES = [0.1, 0.6, 1.0, 2.0]


def _ref_softmax(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch.  Returns same dtype as input."""
    scaled = logits.float() / temperatures.unsqueeze(1)
    return torch.softmax(scaled, dim=-1).to(logits.dtype)


def _tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return dict(rtol=5e-4, atol=5e-5)
    return dict(rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(
    batch_size: int, vocab_size: int, temperature: float, dtype: torch.dtype
):
    """Test that temperature scaling works correctly."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=DEVICE)
    temperatures = torch.full(
        (batch_size,), temperature, dtype=torch.float32, device=DEVICE
    )
    result = softmax_sampling(logits, temperatures)
    expected = _ref_softmax(logits, temperatures)
    torch.testing.assert_close(result, expected, **_tolerances(dtype))


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_per_row_temperature(
    batch_size: int, vocab_size: int, dtype: torch.dtype
):
    """Test per-row (heterogeneous) temperature scaling."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=DEVICE)
    temperatures = torch.rand(batch_size, dtype=torch.float32, device=DEVICE)
    temperatures.clamp_(0.001, 2.0)
    result = softmax_sampling(logits, temperatures)
    expected = _ref_softmax(logits, temperatures)
    torch.testing.assert_close(result, expected, **_tolerances(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_math_properties(dtype: torch.dtype):
    """Verify output probabilities sum to ~1.0 for each row."""
    torch.manual_seed(42)
    batch_size, vocab_size = 256, 128256
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=DEVICE)
    temperatures = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    result = softmax_sampling(logits, temperatures)
    row_sums = result.float().sum(dim=-1)
    torch.testing.assert_close(
        row_sums,
        torch.ones(batch_size, dtype=torch.float32, device=DEVICE),
        **_tolerances(torch.float32),
    )
    assert torch.all(result >= 0), "All probabilities must be non-negative"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
