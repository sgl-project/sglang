import sys

import pytest

from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    Ordering,
    ParallelAxis,
    Reduction,
    parse_dim,
    parse_dims,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
