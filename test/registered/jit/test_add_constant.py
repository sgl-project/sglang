import sys

import pytest
import torch

from sglang.jit_kernel.add_constant import add_constant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)


@pytest.mark.parametrize("size", [1, 2, 127, 128, 1024, 1025, 4096, 4097])
@pytest.mark.parametrize("constant", [0, 1, 7, 1024, -3])
def test_add_constant(size: int, constant: int) -> None:
    src = torch.arange(0, size, dtype=torch.int32, device="cuda")
    dst = add_constant(src, constant)
    assert torch.all(dst == src + constant)


def test_add_constant_unaligned_input() -> None:
    src = torch.arange(0, 4098, dtype=torch.int32, device="cuda")[1:]
    dst = add_constant(src, 7)
    assert torch.all(dst == src + 7)


@pytest.mark.parametrize("size", [2**20, 2**20 + 3])
def test_add_constant_large_aligned_input(size: int) -> None:
    src = torch.arange(0, size, dtype=torch.int32, device="cuda")
    dst = add_constant(src, -3)
    assert torch.all(dst == src - 3)


def test_add_constant_large_unaligned_input() -> None:
    src = torch.arange(0, 2**20 + 4, dtype=torch.int32, device="cuda")[1:]
    dst = add_constant(src, 7)
    assert torch.all(dst == src + 7)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
