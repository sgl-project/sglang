import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims import (
    BATCH_DIM_NAME,
    SEQ_DIM_NAME,
    SQUEEZE_DIM_NAME,
    TOKEN_DIM_NAME,
    DimSpec,
    DimsSpec,
    Ordering,
    ParallelAxis,
    ParallelModifier,
    Reduction,
    _SingletonDimUtil,
    apply_dim_names,
    find_dim_index,
    parse_dim,
    parse_dims,
    resolve_dim_by_name,
    resolve_dim_names,
    strip_dim_names,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestParseDim:
    def test_plain_name(self) -> None:
        assert parse_dim("b") == DimSpec(name="b")

    def test_parallel_axis(self) -> None:
        assert parse_dim("h(tp)") == DimSpec(
            name="h",
            parallel_modifiers=[ParallelModifier(axis=ParallelAxis.TP)],
        )

    def test_all_parallel_axes(self) -> None:
        assert parse_dim("a(tp)").parallel_modifiers[0].axis == ParallelAxis.TP
        assert parse_dim("a(cp)").parallel_modifiers[0].axis == ParallelAxis.CP
        assert parse_dim("a(ep)").parallel_modifiers[0].axis == ParallelAxis.EP
        assert parse_dim("a(sp)").parallel_modifiers[0].axis == ParallelAxis.SP

    def test_ordering(self) -> None:
        assert (
            parse_dim("s(cp:zigzag)").parallel_modifiers[0].ordering == Ordering.ZIGZAG
        )
        assert (
            parse_dim("s(cp:natural)").parallel_modifiers[0].ordering
            == Ordering.NATURAL
        )

    def test_reduction(self) -> None:
        assert (
            parse_dim("h(tp:partial)").parallel_modifiers[0].reduction
            == Reduction.PARTIAL
        )

    def test_all_qualifiers(self) -> None:
        assert parse_dim("s(cp:zigzag+partial)") == DimSpec(
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
        result: DimSpec = parse_dim("t(cp:zigzag,sp)")
        assert result.name == "t"
        assert len(result.parallel_modifiers) == 2
        assert result.parallel_modifiers[0] == ParallelModifier(
            axis=ParallelAxis.CP, ordering=Ordering.ZIGZAG
        )
        assert result.parallel_modifiers[1] == ParallelModifier(axis=ParallelAxis.SP)

    def test_invalid_token_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h()")
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("h(tp(x))")

    def test_unknown_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown axis"):
            parse_dim("h(xyz)")

    def test_unknown_qualifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown qualifier"):
            parse_dim("h(tp:foobar)")

    def test_multiple_ordering_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple ordering"):
            parse_dim("s(cp:zigzag+natural)")

    def test_multiple_reduction_raises(self) -> None:
        with pytest.raises(ValueError, match="Multiple reduction"):
            parse_dim("h(tp:partial+partial)")

    def test_duplicate_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate axis"):
            parse_dim("h(tp,tp)")

    def test_squeeze_dim(self) -> None:
        assert parse_dim("1") == DimSpec(name="1")

    def test_squeeze_dim_rejects_modifiers(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("1(tp)")

    def test_squeeze_dim(self) -> None:
        assert parse_dim("1") == DimSpec(name="1")

    def test_squeeze_dim_rejects_modifiers(self) -> None:
        with pytest.raises(ValueError, match="Invalid dim token"):
            parse_dim("1(tp)")


class TestParseDims:
    def test_multi_dims(self) -> None:
        assert parse_dims("b s h d").dims == [
            DimSpec(name="b"),
            DimSpec(name="s"),
            DimSpec(name="h"),
            DimSpec(name="d"),
        ]

    def test_single_dim(self) -> None:
        assert parse_dims("t").dims == [DimSpec(name="t")]

    def test_mixed_annotated(self) -> None:
        assert parse_dims("b s(cp:zigzag) h(tp) d").dims == [
            DimSpec(name="b"),
            DimSpec(
                name="s",
                parallel_modifiers=[
                    ParallelModifier(axis=ParallelAxis.CP, ordering=Ordering.ZIGZAG),
                ],
            ),
            DimSpec(
                name="h",
                parallel_modifiers=[ParallelModifier(axis=ParallelAxis.TP)],
            ),
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

    def test_with_squeeze_dims(self) -> None:
        dims: list[DimSpec] = parse_dims("t 1 h").dims
        assert len(dims) == 3
        assert dims[0] == DimSpec(name="t")
        assert dims[1] == DimSpec(name="1")
        assert dims[2] == DimSpec(name="h")

    def test_multiple_squeeze_dims_no_duplicate_error(self) -> None:
        dims: list[DimSpec] = parse_dims("t 1 h 1 d").dims
        assert len(dims) == 5
        assert dims[1] == DimSpec(name="1")
        assert dims[3] == DimSpec(name="1")


class TestDimConstants:
    def test_token_dim_name(self) -> None:
        assert TOKEN_DIM_NAME == "t"

    def test_batch_dim_name(self) -> None:
        assert BATCH_DIM_NAME == "b"

    def test_seq_dim_name(self) -> None:
        assert SEQ_DIM_NAME == "s"


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
        specs: list[DimSpec] = parse_dims("b s(cp:zigzag) h(tp) d").dims
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


class TestResolveDimNames:
    def test_no_squeeze(self) -> None:
        assert resolve_dim_names("t h d") == ["t", "h", "d"]

    def test_single_squeeze(self) -> None:
        assert resolve_dim_names("t 1 h") == ["t", "singleton0", "h"]

    def test_multiple_squeeze(self) -> None:
        assert resolve_dim_names("1 t 1 h") == [
            "singleton0",
            "t",
            "singleton1",
            "h",
        ]


class TestSingletonDimUtilFilterOut:
    def test_no_squeeze(self) -> None:
        specs: list[DimSpec] = parse_dims("t h d").dims
        assert _SingletonDimUtil.filter_out(specs) == specs

    def test_with_squeeze(self) -> None:
        specs: list[DimSpec] = parse_dims("t 1 h").dims
        filtered: list[DimSpec] = _SingletonDimUtil.filter_out(specs)
        assert len(filtered) == 2
        assert filtered[0].name == "t"
        assert filtered[1].name == "h"

    def test_all_squeeze(self) -> None:
        specs: list[DimSpec] = parse_dims("1 1").dims
        assert _SingletonDimUtil.filter_out(specs) == []


class TestSingletonDimUtilIsSqueeze:
    def test_squeeze(self) -> None:
        assert _SingletonDimUtil.is_squeeze(DimSpec(name=SQUEEZE_DIM_NAME)) is True

    def test_non_squeeze(self) -> None:
        assert _SingletonDimUtil.is_squeeze(DimSpec(name="t")) is False


class TestSingletonDimUtilMakeName:
    def test_indices(self) -> None:
        assert _SingletonDimUtil.make_name(0) == "singleton0"
        assert _SingletonDimUtil.make_name(1) == "singleton1"
        assert _SingletonDimUtil.make_name(99) == "singleton99"


class TestSingletonDimUtilSanitizeNames:
    def test_no_squeeze(self) -> None:
        assert _SingletonDimUtil.sanitize_names(["t", "h", "d"]) == ["t", "h", "d"]

    def test_single_squeeze(self) -> None:
        assert _SingletonDimUtil.sanitize_names(["t", "1", "h"]) == [
            "t",
            "singleton0",
            "h",
        ]

    def test_multiple_squeeze(self) -> None:
        assert _SingletonDimUtil.sanitize_names(["1", "t", "1", "h"]) == [
            "singleton0",
            "t",
            "singleton1",
            "h",
        ]

    def test_empty(self) -> None:
        assert _SingletonDimUtil.sanitize_names([]) == []


class TestParseDimsWithHash:
    """parse_dims strips the ``#`` declaration section from dims."""

    def test_shape_dims_unchanged(self) -> None:
        assert parse_dims("b s h(tp) # dp:=moe_dp").dims == parse_dims("b s h(tp)").dims

    def test_dp_group_alias_extracted(self) -> None:
        assert parse_dims("b s h(tp) # dp:=moe_dp").dp_group_alias == "moe_dp"

    def test_no_hash_no_alias(self) -> None:
        assert parse_dims("b s h(tp)").dp_group_alias is None

    def test_whitespace_around_hash(self) -> None:
        assert parse_dims("t h #   dp:=foo  ").dims == parse_dims("t h").dims
        assert parse_dims("t h #   dp:=foo  ").dp_group_alias == "foo"

    def test_multiple_declarations_picks_dp(self) -> None:
        result: DimsSpec = parse_dims("t h(tp) # dp:=moe_dp ep:replicated")
        assert result.dims == parse_dims("t h(tp)").dims
        assert result.dp_group_alias == "moe_dp"

    def test_no_dp_alias_token(self) -> None:
        assert parse_dims("t h(tp) # ep:replicated").dp_group_alias is None


class TestDpGroupAlias:
    def test_basic(self) -> None:
        assert parse_dims("b s h(tp) # dp:=moe_dp").dp_group_alias == "moe_dp"

    def test_no_hash_returns_none(self) -> None:
        assert parse_dims("t h").dp_group_alias is None

    def test_no_dp_alias_token(self) -> None:
        assert parse_dims("t h(tp) # ep:replicated").dp_group_alias is None

    def test_multiple_tokens_picks_dp(self) -> None:
        assert (
            parse_dims("b s # ep:replicated dp:=custom_dp").dp_group_alias
            == "custom_dp"
        )


class TestResolveDimNamesWithHash:
    def test_hash_stripped(self) -> None:
        assert resolve_dim_names("t h # dp:=moe_dp") == ["t", "h"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
