import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.utils import (
    Pair,
    argmax_coord,
    calc_rel_diff,
    compute_smaller_dtype,
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

    def test_similar_tensors(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.01, 2.01, 3.01])
        result = calc_rel_diff(x, y).item()
        assert 0.0 < result < 0.01

    def test_negated_tensors(self):
        x = torch.tensor([1.0, 2.0])
        result = calc_rel_diff(x, -x).item()
        assert result == pytest.approx(2.0, abs=1e-5)


class TestArgmaxCoord:
    def test_1d_tensor(self):
        x = torch.tensor([0.0, 0.0, 5.0, 0.0])
        assert argmax_coord(x) == (2,)

    def test_2d_tensor(self):
        x = torch.zeros(3, 4)
        x[1, 2] = 10.0
        assert argmax_coord(x) == (1, 2)

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

    def test_same_shape_noop(self):
        target = torch.Size([3, 4])
        x = torch.randn(3, 4)
        result = try_unify_shape(x, target)
        assert result.shape == target
        assert result.data_ptr() == x.data_ptr()

    def test_trailing_dims_mismatch(self):
        target = torch.Size([5, 6])
        x = torch.randn(1, 3, 4)
        result = try_unify_shape(x, target)
        assert result.shape == (1, 3, 4)


class TestComputeSmallerDtype:
    def test_float32_bfloat16(self):
        assert (
            compute_smaller_dtype(Pair(x=torch.float32, y=torch.bfloat16))
            == torch.bfloat16
        )

    def test_reverse_order(self):
        assert (
            compute_smaller_dtype(Pair(x=torch.bfloat16, y=torch.float32))
            == torch.bfloat16
        )

    def test_same_dtype_returns_none(self):
        assert compute_smaller_dtype(Pair(x=torch.float32, y=torch.float32)) is None

    def test_unknown_pair_returns_none(self):
        assert compute_smaller_dtype(Pair(x=torch.int32, y=torch.int64)) is None


class TestPairMap:
    def test_map_basic(self):
        pair = Pair(x=[1, 2, 3], y=[4, 5, 6])
        result = pair.map(lambda lst: sum(lst))
        assert result.x == 6
        assert result.y == 15

    def test_map_type_change(self):
        pair = Pair(x=[1, 2, 3], y=[10, 20])
        result = pair.map(len)
        assert result.x == 3
        assert result.y == 2

    def test_map_returns_new_pair(self):
        pair = Pair(x="hello", y="world")
        result = pair.map(str.upper)
        assert result.x == "HELLO"
        assert result.y == "WORLD"
        assert result is not pair


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
