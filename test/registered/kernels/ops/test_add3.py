"""Elementwise 3-way add: out = bf16(bf16(a + b) + c).

Callers (e.g. the K3 MoE tail folding the attn-res delayed +prefix_sum)
replace two unfused elementwise adds with this kernel under a bit-exactness
contract: fold vs no-fold must produce identical streams, so the kernel must
round (a + b) to bf16 BEFORE adding c, exactly like the unfused pair. The
adversarial round-to-even tie case turns red iff the kernel switches to a
single-rounded fp32 sum; the prefetch_bc entry point must compute the same
function (it only reorders loads around the PDL wait).
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.elementwise import add3
from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

NUM_TOKENS = [2**n for n in range(0, 15)]
NUM_TOKENS += [x + 1 + i for i, x in enumerate(NUM_TOKENS)]
NUM_TOKENS = get_ci_test_range(NUM_TOKENS, [1, 9, 256, 16399])
# All divisible by 16 so numel stays a vector-width multiple at any T.
HIDDEN_DIMS = get_ci_test_range([48, 64, 112, 512, 1024, 4096, 7168], [64, 112, 7168])
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _rand_bf16(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape, dtype=torch.float32, device=DEVICE) * 8.0).to(DTYPE)


def _assert_bit_exact(a, b, c, *, prefetch_bc: bool = False) -> None:
    assert add3.covered(a, b, c)
    out = add3.add3(a, b, c, prefetch_bc=prefetch_bc)
    ref = (a + b) + c  # the unfused pair: bf16-rounded at each step
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product(NUM_TOKENS, HIDDEN_DIMS)),
)
def test_bit_exact(num_tokens: int, hidden_dim: int) -> None:
    torch.manual_seed(num_tokens * 7 + hidden_dim)
    a = _rand_bf16(num_tokens, hidden_dim)
    b = _rand_bf16(num_tokens, hidden_dim)
    c = _rand_bf16(num_tokens, hidden_dim)
    _assert_bit_exact(a, b, c)


# Smaller subset for targeted tests below
REPR_TOKENS = get_ci_test_range([1, 7, 128, 4096], [1, 128])


@pytest.mark.parametrize("num_tokens", REPR_TOKENS)
def test_prefetch_bc(num_tokens: int) -> None:
    # The prefetch entry point only reorders loads around the PDL wait; it
    # must compute the same function as the default entry point.
    torch.manual_seed(num_tokens)
    a = _rand_bf16(num_tokens, 7168)
    b = _rand_bf16(num_tokens, 7168)
    c = _rand_bf16(num_tokens, 7168)
    _assert_bit_exact(a, b, c, prefetch_bc=True)


def test_any_rank() -> None:
    # The kernel is shape-agnostic (flattened by the wrapper): any rank is
    # accepted, and vectors may span row boundaries.
    torch.manual_seed(0)
    _assert_bit_exact(
        _rand_bf16(4, 3, 112), _rand_bf16(4, 3, 112), _rand_bf16(4, 3, 112)
    )


def test_double_rounding_order() -> None:
    # bf16 has a 7-bit mantissa, so ulp(1.0) = 2^-7 and 1 + 2^-8 is a
    # round-to-even tie back to 1.0: the unfused pair yields exactly 1.0
    # twice, while single-rounded fp32 a + b + c gives 1 + 2^-7, a
    # different bf16.
    a = torch.ones(2, 512, device=DEVICE, dtype=DTYPE)
    b = torch.full((2, 512), 2.0**-8, device=DEVICE, dtype=DTYPE)
    c = torch.full((2, 512), 2.0**-8, device=DEVICE, dtype=DTYPE)
    single_rounded = (a.float() + b.float() + c.float()).to(DTYPE)
    assert not torch.equal(single_rounded, torch.ones_like(a))
    out = add3.add3(a, b, c)
    assert torch.equal(out, torch.ones_like(a))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
