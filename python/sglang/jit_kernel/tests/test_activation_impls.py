"""Reference exemplar: correctness test for a multi-impl kernel.

Pairs 1:1 with ``benchmark/bench_activation.py`` (the marker bench pilot). The
bench compares the same three impls for *speed* with a loose tolerance; this
test is the correctness *contract*: every kernel impl is asserted against an
independent fp32 torch gold reference, plus a tight cross-impl check.

Impls under test:
  - ``aot``  : sgl_kernel AOT-compiled kernel
  - ``jit``  : sglang.jit_kernel triton kernel
  - ``torch``: eager fp32 reference (the gold oracle, not a kernel)

Copy this shape for any kernel that ships more than one implementation.
"""

import sys

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import gelu_and_mul as gelu_and_mul_aot
from sgl_kernel import gelu_tanh_and_mul as gelu_tanh_and_mul_aot
from sgl_kernel import silu_and_mul as silu_and_mul_aot

from sglang.jit_kernel.activation import gelu_and_mul as gelu_and_mul_jit
from sglang.jit_kernel.activation import gelu_tanh_and_mul as gelu_tanh_and_mul_jit
from sglang.jit_kernel.activation import silu_and_mul as silu_and_mul_jit
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


# impl name -> {op_name -> callable(x) -> tensor}
KERNELS = {
    "aot": {
        "silu": silu_and_mul_aot,
        "gelu": gelu_and_mul_aot,
        "gelu_tanh": gelu_tanh_and_mul_aot,
    },
    "jit": {
        "silu": silu_and_mul_jit,
        "gelu": gelu_and_mul_jit,
        "gelu_tanh": gelu_tanh_and_mul_jit,
    },
}
IMPLS = list(KERNELS)
OPS = ["silu", "gelu", "gelu_tanh"]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
# Adversarial shapes (odd batches/tiny/non-power-of-two), not the "nice" bench grid.
# The activation kernels require vector-aligned hidden dimensions.
SHAPES = get_ci_test_range(
    full_range=[
        (1, 16),
        (7, 32),
        (83, 1024),
        (3, 5, 16),
        (2, 3, 512),
        (1, 17, 4096),
        (4096, 8192),
    ],
    ci_range=[(7, 32), (2, 3, 512)],
)


def _gold(op_name: str, x: torch.Tensor) -> torch.Tensor:
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


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_impl_matches_gold(
    impl: str, op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """Each kernel impl must match the fp32 torch gold reference."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out = KERNELS[impl][op_name](x)
    expected = _gold(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_aot_matches_jit(
    op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    """The two kernel impls share an algorithm; they must agree tightly with
    each other (catches a shared-vs-gold drift that loose gold tolerance hides)."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out_aot = KERNELS["aot"][op_name](x)
    out_jit = KERNELS["jit"][op_name](x)
    atol = 0.0 if dtype == torch.float32 else 1e-3
    torch.testing.assert_close(out_aot, out_jit, atol=atol, rtol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
