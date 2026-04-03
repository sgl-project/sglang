import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


OPS = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = get_ci_test_range(
    full_range=[
        (7, 16),
        (83, 1024),
        (3, 5, 16),
        (2, 3, 512),
        (1, 17, 4096),
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
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device="cuda")

    out = OPS[op_name](x)
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_activation_out_param(op_name: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn((4, 7, 128), dtype=dtype, device="cuda")
    out = torch.empty((4, 7, 64), dtype=dtype, device="cuda")

    result = OPS[op_name](x, out)
    assert result is out

    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
