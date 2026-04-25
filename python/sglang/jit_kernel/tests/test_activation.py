import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.activation import SUPPORTED_ACTIVATIONS, run_activation
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


OPS = SUPPORTED_ACTIVATIONS
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = get_ci_test_range(
    full_range=[
        (7, 16),
        (83, 1024),
        (3, 5, 16),
        (2, 3, 512),
        (1, 17, 4096),
        *[(2**x, 2048) for x in range(0, 15, 2)],
        *[(2**x, 65536) for x in range(0, 5, 2)],
    ],
    ci_range=[(7, 16), (2, 3, 512)],
)


def _reference(op_name: str, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    lhs = x[..., :d].float()
    rhs = x[..., d:]
    if op_name == "silu":
        act = F.silu(lhs)
    elif op_name == "gelu":
        act = F.gelu(lhs, approximate="none")
    else:
        act = F.gelu(lhs, approximate="tanh")
    return act.to(dtype=x.dtype) * rhs


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_correctness(
    op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out = run_activation(op_name, x, None)
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_out_param(
    op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape[:-1] + (shape[-1] // 2,), dtype=dtype, device="cuda")
    result = run_activation(op_name, x, out)
    assert result is out
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
