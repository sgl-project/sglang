import math
import sys

import pytest
import torch

from sglang.jit_kernel.new_gelu import new_gelu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _reference_new_gelu(x: torch.Tensor) -> torch.Tensor:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [1, 127, 128, 1024, 4096, 4097, 11008, 16384])
def test_new_gelu_correctness(dtype: torch.dtype, size: int) -> None:
    x = torch.randn(size, dtype=dtype, device="cuda")
    result = new_gelu(x)
    expected = _reference_new_gelu(x)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_new_gelu_multidim(dtype: torch.dtype) -> None:
    x = torch.randn(4, 128, 2048, dtype=dtype, device="cuda")
    result = new_gelu(x)
    expected = _reference_new_gelu(x)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_new_gelu_out_param(dtype: torch.dtype) -> None:
    x = torch.randn(1024, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    result = new_gelu(x, out=out)
    assert result is out
    expected = _reference_new_gelu(x)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_new_gelu_cpu_error() -> None:
    x = torch.randn(128, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="CUDA"):
        new_gelu(x)


def test_new_gelu_unsupported_dtype() -> None:
    x = torch.randint(0, 10, (128,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        new_gelu(x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
