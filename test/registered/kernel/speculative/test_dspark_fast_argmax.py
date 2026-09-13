import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.fast_argmax import fast_row_argmax
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast_row_argmax requires CUDA"
)

VOCAB = 129280


def _check(x: torch.Tensor) -> None:
    got = fast_row_argmax(x)
    want = torch.argmax(x, dim=-1)
    assert got.dtype == want.dtype == torch.int64
    assert torch.equal(got, want), (got - want).nonzero()


@pytest.mark.parametrize("rows", [1, 6, 8, 64])
@pytest.mark.parametrize("vocab", [4096, 32000, VOCAB])
def test_matches_torch_argmax(rows: int, vocab: int):
    g = torch.Generator(device="cuda").manual_seed(rows * 1000 + vocab)
    x = torch.randn((rows, vocab), device="cuda", dtype=torch.float32, generator=g)
    _check(x)


def test_ties_resolve_to_the_lowest_index():
    # Every column equal: torch.argmax returns 0, and so must the split kernel.
    x = torch.zeros((6, VOCAB), device="cuda", dtype=torch.float32)
    _check(x)
    # A tie spanning two different partials, plus one strictly larger value in
    # a third, so the final stage has to break a tie and pick a winner.
    x = torch.full((6, VOCAB), -1.0, device="cuda", dtype=torch.float32)
    x[:, 100] = 5.0
    x[:, VOCAB // 2] = 5.0
    x[3, VOCAB - 7] = 9.0
    _check(x)


def test_infinities():
    x = torch.randn((6, VOCAB), device="cuda", dtype=torch.float32)
    x[0, :] = float("-inf")  # an all -inf row still has to return an index
    x[1, 77] = float("inf")
    x[2, VOCAB - 1] = float("inf")
    x[3, 5] = float("inf")
    x[3, 6] = float("inf")  # first +inf wins
    x[4, :] = float("-inf")
    x[4, VOCAB - 2] = 0.0
    _check(x)


def test_strided_rows():
    # A column slice of a padded buffer: only the innermost stride is unit.
    buf = torch.randn((6, VOCAB + 64), device="cuda", dtype=torch.float32)
    x = buf[:, :VOCAB]
    assert x.stride(1) == 1 and not x.is_contiguous()
    _check(x)


def test_out_parameter_is_filled():
    x = torch.randn((6, VOCAB), device="cuda", dtype=torch.float32)
    out = torch.empty((6,), device="cuda", dtype=torch.int64)
    got = fast_row_argmax(x, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(out, torch.argmax(x, dim=-1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
