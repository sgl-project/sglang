import sys

import pytest

from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    Ordering,
    ParallelAxis,
    Reduction,
    parse_dims,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestParseDims:
    def test_plain_dims(self) -> None:
        result = parse_dims("b s h d")
        assert result == [
            DimSpec(name="b"),
            DimSpec(name="s"),
            DimSpec(name="h"),
            DimSpec(name="d"),
        ]

    def test_single_dim(self) -> None:
        assert parse_dims("t") == [DimSpec(name="t")]

    def test_tp_annotated(self) -> None:
        result = parse_dims("b s h(tp) d")
        assert result == [
            DimSpec(name="b"),
            DimSpec(name="s"),
            DimSpec(name="h", parallel=ParallelAxis.TP),
            DimSpec(name="d"),
        ]

    def test_cp_zigzag(self) -> None:
        result = parse_dims("b s(cp,zigzag) h(tp) d")
        assert result == [
            DimSpec(name="b"),
            DimSpec(name="s", parallel=ParallelAxis.CP, ordering=Ordering.ZIGZAG),
            DimSpec(name="h", parallel=ParallelAxis.TP),
            DimSpec(name="d"),
        ]

    def test_tp_partial(self) -> None:
        result = parse_dims("b s h(tp,partial) d")
        assert result == [
            DimSpec(name="b"),
            DimSpec(name="s"),
            DimSpec(name="h", parallel=ParallelAxis.TP, reduction=Reduction.PARTIAL),
            DimSpec(name="d"),
        ]

    def test_all_modifiers(self) -> None:
        result = parse_dims("s(cp,zigzag,partial)")
        assert result == [
            DimSpec(
                name="s",
                parallel=ParallelAxis.CP,
                ordering=Ordering.ZIGZAG,
                reduction=Reduction.PARTIAL,
            ),
        ]

    def test_ep_and_sp(self) -> None:
        assert parse_dims("t(ep)")[0].parallel == ParallelAxis.EP
        assert parse_dims("s(sp)")[0].parallel == ParallelAxis.SP

    def test_natural_ordering(self) -> None:
        result = parse_dims("s(cp,natural)")
        assert result[0].ordering == Ordering.NATURAL

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_dims("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_dims("   ")

    def test_duplicate_dim_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_dims("h h")

    def test_unknown_token_in_parens_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown modifier"):
            parse_dims("h(tp,foobar)")

    def test_empty_parens_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dims("h()")

    def test_invalid_modifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown modifier"):
            parse_dims("h(xyz)")

    def test_nested_parens_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dims("h(tp(x))")

    def test_multiple_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple ordering"):
            parse_dims("s(cp,zigzag,natural)")

    def test_multiple_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple reduction"):
            parse_dims("h(tp,partial,partial)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
