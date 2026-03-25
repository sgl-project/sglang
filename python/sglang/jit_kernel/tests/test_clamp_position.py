import sys

import pytest
import torch

from sglang.jit_kernel.clamp_position import clamp_position_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _reference_clamp_position(seq_lens):
    return torch.clamp(seq_lens - 1, min=0).to(seq_lens.dtype)


@pytest.mark.parametrize("size", [1, 2, 127, 128, 255, 256, 1024, 4097])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
class TestClampPosition:
    def test_normal(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.randint(1, 10000, (size,), dtype=dtype, device="cuda")
        expected = _reference_clamp_position(seq_lens)
        result = clamp_position_cuda(seq_lens)
        assert torch.equal(result, expected)

    def test_zeros(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.zeros(size, dtype=dtype, device="cuda")
        expected = _reference_clamp_position(seq_lens)
        result = clamp_position_cuda(seq_lens)
        assert torch.equal(result, expected)

    def test_ones(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.ones(size, dtype=dtype, device="cuda")
        expected = _reference_clamp_position(seq_lens)
        result = clamp_position_cuda(seq_lens)
        assert torch.equal(result, expected)

    def test_mixed(self, size: int, dtype: torch.dtype) -> None:
        seq_lens = torch.randint(0, 10000, (size,), dtype=dtype, device="cuda")
        expected = _reference_clamp_position(seq_lens)
        result = clamp_position_cuda(seq_lens)
        assert torch.equal(result, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
