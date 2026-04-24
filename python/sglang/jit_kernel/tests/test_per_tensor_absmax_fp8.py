import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.per_tensor_absmax_fp8 import per_tensor_absmax_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=60, suite="nightly-kernel-1-gpu", nightly=True)

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
FP8_MAX = torch.finfo(fp8_type_).max


def reference_absmax_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(x)).float() / FP8_MAX


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([1, 7, 128, 512], [64, 512, 2048, 4096])),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_absmax_correctness(num_tokens: int, hidden_dim: int, dtype: torch.dtype):
    device = torch.device("cuda")
    x = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

    scale = torch.zeros(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(x, scale)

    expected = reference_absmax_scale(x)
    torch.testing.assert_close(scale.squeeze(), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("numel", [1, 3, 127, 128, 1023, 1024, 4097, 100003])
def test_absmax_1d(numel: int):
    device = torch.device("cuda")
    x = torch.randn(numel, dtype=torch.bfloat16, device=device)

    scale = torch.zeros(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(x, scale)

    expected = reference_absmax_scale(x)
    torch.testing.assert_close(scale.squeeze(), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("shape", [(4, 8, 64), (2, 16, 128)])
def test_absmax_3d(shape):
    device = torch.device("cuda")
    x = torch.randn(shape, dtype=torch.float16, device=device)

    scale = torch.zeros(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(x, scale)

    expected = reference_absmax_scale(x)
    torch.testing.assert_close(scale.squeeze(), expected, rtol=1e-5, atol=1e-8)


def test_absmax_large():
    """Stress test with a large tensor to exercise multi-block atomic reduction."""
    device = torch.device("cuda")
    x = torch.randn(4096, 7168, dtype=torch.bfloat16, device=device)

    scale = torch.zeros(1, dtype=torch.float32, device=device)
    per_tensor_absmax_fp8(x, scale)

    expected = reference_absmax_scale(x)
    torch.testing.assert_close(scale.squeeze(), expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
