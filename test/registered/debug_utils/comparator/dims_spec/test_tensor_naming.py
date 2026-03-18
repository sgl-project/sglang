import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims_spec import (
    DimSpec,
    apply_dim_names,
    find_dim_index,
    parse_dims,
    resolve_dim_by_name,
    strip_dim_names,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default", nightly=True)


class TestFindDimIndex:
    def test_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "s") == 1

    def test_not_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "t") is None

    def test_first_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("t h d").dims
        assert find_dim_index(specs, "t") == 0

    def test_last_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d").dims
        assert find_dim_index(specs, "d") == 3

    def test_with_modifiers(self) -> None:
        specs: list[DimSpec] = parse_dims("b s[cp:zigzag] h[tp] d").dims
        assert find_dim_index(specs, "h") == 2

    def test_empty_list(self) -> None:
        assert find_dim_index([], "t") is None


class TestResolveDimByName:
    def test_resolve_found(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3, 4).refine_names("b", "s", "h")
        assert resolve_dim_by_name(tensor, "b") == 0
        assert resolve_dim_by_name(tensor, "s") == 1
        assert resolve_dim_by_name(tensor, "h") == 2

    def test_resolve_not_found_raises(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3).refine_names("b", "s")
        with pytest.raises(ValueError, match="not in tensor names"):
            resolve_dim_by_name(tensor, "h")

    def test_resolve_unnamed_raises(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        with pytest.raises(ValueError, match="no names"):
            resolve_dim_by_name(tensor, "b")


class TestApplyDimNames:
    def test_apply(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3, 4)
        named: torch.Tensor = apply_dim_names(tensor, ["b", "s", "h"])
        assert named.names == ("b", "s", "h")
        assert named.shape == (2, 3, 4)

    def test_apply_preserves_data(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        named: torch.Tensor = apply_dim_names(tensor, ["x", "y"])
        assert torch.equal(strip_dim_names(named), tensor)

    def test_ndim_mismatch_gives_clear_error(self) -> None:
        tensor: torch.Tensor = torch.randn(10, 1, 128)
        with pytest.raises(
            ValueError,
            match=r"dims metadata mismatch.*3 dims.*shape \[10, 1, 128\].*2 names \['t', 'num_experts'\].*fix the dims string",
        ):
            apply_dim_names(tensor, ["t", "num_experts"])


class TestStripDimNames:
    def test_strip(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3).refine_names("a", "b")
        stripped: torch.Tensor = strip_dim_names(tensor)
        assert stripped.names == (None, None)

    def test_strip_already_unnamed(self) -> None:
        tensor: torch.Tensor = torch.randn(2, 3)
        stripped: torch.Tensor = strip_dim_names(tensor)
        assert stripped.names == (None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
