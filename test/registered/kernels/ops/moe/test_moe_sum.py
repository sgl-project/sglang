"""Correctness tests for the JIT ``moe_sum`` kernel.

``moe_sum`` sums the top-k expert outputs of every token:
``output[num_tokens, hidden] = input[num_tokens, topk, hidden].sum(dim=1)``.

The JIT kernel is a host-side port of the AOT ``sgl_kernel.moe_sum``: the
device code is unchanged (topk 2/3/4 are specialised, every other topk falls
back to ``torch.sum``), so the comparison against the AOT kernel is exact.

Tests go through the public ``sglang.kernels.ops.moe.moe_sum`` wrapper, which
on CUDA is expected to dispatch to the JIT implementation; a dedicated test
pins that routing. Every case is also checked against a plain ``torch.sum``
reference, which accumulates in fp32 and therefore needs a dtype tolerance.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.moe import moe_sum
from sglang.kernels.selector import get_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")

try:
    from sgl_kernel import moe_sum as aot_moe_sum

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

DTYPES = get_ci_test_range(
    full_range=[torch.float16, torch.bfloat16, torch.float32],
    ci_range=[torch.bfloat16],
)
NUM_TOKENS = get_ci_test_range(full_range=[1, 7, 128, 1024], ci_range=[1, 128])
# 2/3/4 take the specialised kernel; 1/5/8 exercise the torch.sum fallback.
TOPKS = get_ci_test_range(full_range=[1, 2, 3, 4, 5, 8], ci_range=[2, 4, 5])
HIDDEN_SIZES = get_ci_test_range(
    full_range=[1, 31, 128, 511, 1024, 4096], ci_range=[31, 1024]
)

# The kernel accumulates in the input dtype (same as AOT); torch.sum accumulates
# in fp32, so a few ulps of drift are expected for the half-precision types.
_TOL = {
    torch.float32: dict(rtol=1e-6, atol=1e-6),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


def _make_inputs(num_tokens: int, topk: int, hidden_size: int, dtype: torch.dtype):
    torch.manual_seed(0)
    input = torch.randn((num_tokens, topk, hidden_size), dtype=dtype, device="cuda")
    output = torch.empty((num_tokens, hidden_size), dtype=dtype, device="cuda")
    return input, output


def test_moe_sum_routes_to_jit_on_cuda():
    kernel = get_kernel("moe.moe_sum")
    assert kernel.__module__ == "sglang.kernels.ops.moe.moe_sum"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_moe_sum_matches_torch(dtype, num_tokens, topk, hidden_size):
    input, output = _make_inputs(num_tokens, topk, hidden_size, dtype)

    moe_sum(input, output)
    ref = input.sum(dim=1)

    assert output.shape == ref.shape
    assert output.dtype == dtype
    torch.testing.assert_close(output, ref, **_TOL[dtype])


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_moe_sum_matches_aot(dtype, num_tokens, topk, hidden_size):
    input, jit_output = _make_inputs(num_tokens, topk, hidden_size, dtype)
    aot_output = torch.empty_like(jit_output)

    moe_sum(input, jit_output)
    aot_moe_sum(input, aot_output)

    # Same device code on both sides, so the match is bit-exact.
    torch.testing.assert_close(jit_output, aot_output, rtol=0, atol=0)


def test_moe_sum_zero_tokens():
    input, output = _make_inputs(0, 2, 128, torch.bfloat16)
    moe_sum(input, output)
    assert output.shape == (0, 128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
