import sys

import pytest

from sglang.srt.debug_utils.comparator.dims_spec import (
    DimSpec,
    Ordering,
    ParallelAxis,
    ParallelModifier,
    Reduction,
    parse_dim,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default", nightly=True)


class TestParseDim:
    def test_plain_name(self) -> None:
        assert parse_dim("b") == DimSpec(name="b")

    def test_parallel_axis(self) -> None:
        assert parse_dim("h[tp]") == DimSpec(
            name="h",
            parallel_modifiers=[ParallelModifier(axis=ParallelAxis.TP)],
        )

    def test_all_parallel_axes(self) -> None:
        assert parse_dim("a[tp]").parallel_modifiers[0].axis == ParallelAxis.TP
        assert parse_dim("a[cp]").parallel_modifiers[0].axis == ParallelAxis.CP
        assert parse_dim("a[ep]").parallel_modifiers[0].axis == ParallelAxis.EP
        assert parse_dim("a[sp]").parallel_modifiers[0].axis == ParallelAxis.SP

    def test_ordering(self) -> None:
        assert (
            parse_dim("s[cp:zigzag]").parallel_modifiers[0].ordering == Ordering.ZIGZAG
        )
        assert (
            parse_dim("s[cp:natural]").parallel_modifiers[0].ordering
            == Ordering.NATURAL
        )

    def test_reduction(self) -> None:
        assert (
            parse_dim("h[tp:partial]").parallel_modifiers[0].reduction
            == Reduction.PARTIAL
        )

    def test_all_qualifiers(self) -> None:
        assert parse_dim("s[cp:zigzag+partial]") == DimSpec(
            name="s",
            parallel_modifiers=[
                ParallelModifier(
                    axis=ParallelAxis.CP,
                    ordering=Ordering.ZIGZAG,
                    reduction=Reduction.PARTIAL,
                ),
            ],
        )

    def test_multi_axis(self) -> None:
        result: DimSpec = parse_dim("t[cp:zigzag,sp]")
        assert result.name == "t"
        assert len(result.parallel_modifiers) == 2
        assert result.parallel_modifiers[0] == ParallelModifier(
            axis=ParallelAxis.CP, ordering=Ordering.ZIGZAG
        )
        assert result.parallel_modifiers[1] == ParallelModifier(axis=ParallelAxis.SP)

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h[]")
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h[tp[x]]")

    def test_unknown_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown axis"):
            parse_dim("h[xyz]")

    def test_unknown_qualifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown qualifier"):
            parse_dim("h[tp:foobar]")

    def test_multiple_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple ordering"):
            parse_dim("s[cp:zigzag+natural]")

    def test_multiple_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple reduction"):
            parse_dim("h[tp:partial+partial]")

    def test_duplicate_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate axis"):
            parse_dim("h[tp,tp]")

    def test_squeeze_dim(self) -> None:
        assert parse_dim("1") == DimSpec(name="1")

    def test_squeeze_dim_rejects_modifiers(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("1[tp]")


class TestParseFusedDim:
    def test_basic_fused(self) -> None:
        result: DimSpec = parse_dim("(num_heads*head_dim)")
        assert result.name == "num_heads*head_dim"
        assert result.parallel_modifiers == []
        assert result.is_fused
        assert result.sub_dims == ["num_heads", "head_dim"]

    def test_fused_with_modifier(self) -> None:
        result: DimSpec = parse_dim("(num_heads*head_dim)[tp]")
        assert result.name == "num_heads*head_dim"
        assert result.parallel_modifiers == [ParallelModifier(axis=ParallelAxis.TP)]
        assert result.sub_dims == ["num_heads", "head_dim"]

    def test_three_way_fused(self) -> None:
        result: DimSpec = parse_dim("(a*b*c)")
        assert result.name == "a*b*c"
        assert len(result.sub_dims) == 3
        assert result.sub_dims == ["a", "b", "c"]

    def test_three_way_fused_with_modifier(self) -> None:
        result: DimSpec = parse_dim("(a*b*c)[tp]")
        assert result.parallel_modifiers == [ParallelModifier(axis=ParallelAxis.TP)]
        assert len(result.sub_dims) == 3

    def test_fused_with_complex_modifier(self) -> None:
        result: DimSpec = parse_dim("(a*b)[cp:zigzag]")
        assert result.parallel_modifiers == [
            ParallelModifier(axis=ParallelAxis.CP, ordering=Ordering.ZIGZAG)
        ]
        assert result.sub_dims == ["a", "b"]

    def test_regular_dim_not_fused(self) -> None:
        result: DimSpec = parse_dim("h[tp]")
        assert not result.is_fused
        assert result.sub_dims == ["h"]

    def test_fused_duplicate_sub_names_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate sub-dim"):
            parse_dim("(a*a)")

    def test_fused_invalid_sub_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid sub-dim"):
            parse_dim("(a*1)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
