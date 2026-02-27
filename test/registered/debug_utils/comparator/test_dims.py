import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims import (
    BATCH_DIM_NAME,
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    DimSpec,
    Ordering,
    ParallelAxis,
    Reduction,
    apply_dim_names,
    find_dim_index,
    parse_dim,
    parse_dim_names,
    parse_dims,
    resolve_dim_by_name,
    strip_dim_names,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestParseDim:
    def test_plain_name(self) -> None:
        assert parse_dim("b") == DimSpec(name="b")

    def test_parallel_axis(self) -> None:
        assert parse_dim("h(tp)") == DimSpec(name="h", parallel=ParallelAxis.TP)

    def test_all_parallel_axes(self) -> None:
        assert parse_dim("a(tp)").parallel == ParallelAxis.TP
        assert parse_dim("a(cp)").parallel == ParallelAxis.CP
        assert parse_dim("a(ep)").parallel == ParallelAxis.EP
        assert parse_dim("a(sp)").parallel == ParallelAxis.SP

    def test_ordering(self) -> None:
        assert parse_dim("s(cp,zigzag)").ordering == Ordering.ZIGZAG
        assert parse_dim("s(cp,natural)").ordering == Ordering.NATURAL

    def test_reduction(self) -> None:
        assert parse_dim("h(tp,partial)").reduction == Reduction.PARTIAL

    def test_all_modifiers(self) -> None:
        assert parse_dim("s(cp,zigzag,partial)") == DimSpec(
            name="s",
            parallel=ParallelAxis.CP,
            ordering=Ordering.ZIGZAG,
            reduction=Reduction.PARTIAL,
        )

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h()")
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h(tp(x))")

    def test_unknown_modifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown modifier"):
            parse_dim("h(xyz)")
        with pytest.raises(ValueError, match="Unknown modifier"):
            parse_dim("h(tp,foobar)")

    def test_multiple_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple ordering"):
            parse_dim("s(cp,zigzag,natural)")

    def test_multiple_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple reduction"):
            parse_dim("h(tp,partial,partial)")


class TestParseDims:
    def test_multi_dims(self) -> None:
        assert parse_dims("b s h d") == [
            DimSpec(name="b"),
            DimSpec(name="s"),
            DimSpec(name="h"),
            DimSpec(name="d"),
        ]

    def test_single_dim(self) -> None:
        assert parse_dims("t") == [DimSpec(name="t")]

    def test_mixed_annotated(self) -> None:
        assert parse_dims("b s(cp,zigzag) h(tp) d") == [
            DimSpec(name="b"),
            DimSpec(name="s", parallel=ParallelAxis.CP, ordering=Ordering.ZIGZAG),
            DimSpec(name="h", parallel=ParallelAxis.TP),
            DimSpec(name="d"),
        ]

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_dims("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_dims("   ")

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_dims("h h")


class TestParseDimNames:
    def test_plain(self) -> None:
        assert parse_dim_names("b s h d") == ["b", "s", "h", "d"]

    def test_strips_modifiers(self) -> None:
        assert parse_dim_names("b s(cp,zigzag) h(tp) d") == ["b", "s", "h", "d"]


class TestDimConstants:
    def test_token_dim_name(self) -> None:
        assert TOKEN_DIM_NAME == "t"

    def test_batch_dim_name(self) -> None:
        assert BATCH_DIM_NAME == "b"

    def test_seq_dim_name(self) -> None:
        assert SEQ_DIM_NAME == "s"


class TestFindDimIndex:
    def test_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d")
        assert find_dim_index(specs, "s") == 1

    def test_not_found(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d")
        assert find_dim_index(specs, "t") is None

    def test_first_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("t h d")
        assert find_dim_index(specs, "t") == 0

    def test_last_dim(self) -> None:
        specs: list[DimSpec] = parse_dims("b s h d")
        assert find_dim_index(specs, "d") == 3

    def test_with_modifiers(self) -> None:
        specs: list[DimSpec] = parse_dims("b s(cp,zigzag) h(tp) d")
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
