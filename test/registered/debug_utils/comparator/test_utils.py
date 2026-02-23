import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.debug_utils.comparator.utils import (
    argmax_coord,
    calc_rel_diff,
    compute_smaller_dtype,
    load_object,
    try_unify_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestCalcRelDiff:
    def test_identical_tensors(self):
        x = torch.randn(10, 10)
        assert calc_rel_diff(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_tensors(self):
        result = calc_rel_diff(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        ).item()
        assert result == pytest.approx(1.0, abs=1e-5)


class TestArgmaxCoord:
    def test_3d_tensor(self):
        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        assert argmax_coord(x) == (1, 2, 3)


class TestTryUnifyShape:
    def test_squeeze_leading_ones(self):
        target = torch.Size([3, 4])
        assert try_unify_shape(torch.randn(1, 1, 3, 4), target).shape == target

    def test_no_squeeze_when_leading_dim_not_one(self):
        target = torch.Size([3, 4])
        assert try_unify_shape(torch.randn(2, 3, 4), target).shape == (2, 3, 4)


class TestComputeSmallerDtype:
    def test_float32_bfloat16(self):
        assert compute_smaller_dtype(torch.float32, torch.bfloat16) == torch.bfloat16

    def test_reverse_order(self):
        assert compute_smaller_dtype(torch.bfloat16, torch.float32) == torch.bfloat16

    def test_same_dtype_returns_none(self):
        assert compute_smaller_dtype(torch.float32, torch.float32) is None


class TestLoadObject:
    def test_load_tensor(self, tmp_path):
        path = tmp_path / "tensor.pt"
        torch.save(torch.randn(5, 5), path)
        assert load_object(path).shape == (5, 5)

    def test_non_tensor_returns_none(self, tmp_path):
        path = tmp_path / "tensor.pt"
        torch.save({"dict": 1}, path)
        assert load_object(path) is None

    def test_nonexistent_returns_none(self):
        assert load_object(Path("/nonexistent.pt")) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
