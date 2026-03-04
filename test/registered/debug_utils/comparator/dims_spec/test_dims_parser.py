import sys

import pytest

from sglang.srt.debug_utils.comparator.dims_spec import (
    SQUEEZE_DIM_NAME,
    DimSpec,
    DimsSpec,
    Ordering,
    ParallelAxis,
    ParallelModifier,
    _SingletonDimUtil,
    parse_dims,
    resolve_dim_names,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default", nightly=True)


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
        assert parse_dims("b s[cp:zigzag] h[tp] d").dims == [
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


class TestParseDimsWithFused:
    def test_fused_in_dims(self) -> None:
        result: DimsSpec = parse_dims("t (num_heads*head_dim)[tp]")
        assert len(result.dims) == 2
        assert result.dims[0] == DimSpec(name="t")
        assert result.dims[1].is_fused
        assert result.dims[1].name == "num_heads*head_dim"

    def test_fused_and_regular_mixed(self) -> None:
        result: DimsSpec = parse_dims("t (num_heads*head_dim)[tp] d")
        assert len(result.dims) == 3
        assert not result.dims[0].is_fused
        assert result.dims[1].is_fused
        assert not result.dims[2].is_fused

    def test_fused_sub_name_conflicts_with_regular_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_dims("t num_heads (num_heads*head_dim)")

    def test_multiple_fused_dims(self) -> None:
        result: DimsSpec = parse_dims("(a*b) (c*d)")
        assert len(result.dims) == 2
        assert result.dims[0].is_fused
        assert result.dims[1].is_fused

    def test_cross_fused_duplicate_sub_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            parse_dims("(a*b) (c*a)")


class TestParseDimsWithHash:
    """parse_dims strips the ``#`` declaration section from dims."""

    def test_shape_dims_unchanged(self) -> None:
        assert parse_dims("b s h[tp] # dp:=moe_dp").dims == parse_dims("b s h[tp]").dims

    def test_dp_group_alias_extracted(self) -> None:
        assert parse_dims("b s h[tp] # dp:=moe_dp").dp_group_alias == "moe_dp"

    def test_no_hash_no_alias(self) -> None:
        assert parse_dims("b s h[tp]").dp_group_alias is None

    def test_whitespace_around_hash(self) -> None:
        assert parse_dims("t h #   dp:=foo  ").dims == parse_dims("t h").dims
        assert parse_dims("t h #   dp:=foo  ").dp_group_alias == "foo"

    def test_multiple_declarations_picks_dp(self) -> None:
        result: DimsSpec = parse_dims("t h[tp] # dp:=moe_dp ep:replicated")
        assert result.dims == parse_dims("t h[tp]").dims
        assert result.dp_group_alias == "moe_dp"
        assert result.replicated_axes == frozenset({ParallelAxis.EP})

    def test_no_dp_alias_token(self) -> None:
        result: DimsSpec = parse_dims("t h[tp] # ep:replicated")
        assert result.dp_group_alias is None
        assert result.replicated_axes == frozenset({ParallelAxis.EP})


class TestDpGroupAlias:
    def test_basic(self) -> None:
        assert parse_dims("b s h[tp] # dp:=moe_dp").dp_group_alias == "moe_dp"

    def test_no_hash_returns_none(self) -> None:
        assert parse_dims("t h").dp_group_alias is None

    def test_no_dp_alias_token(self) -> None:
        assert parse_dims("t h[tp] # ep:replicated").dp_group_alias is None

    def test_multiple_tokens_picks_dp(self) -> None:
        assert (
            parse_dims("b s # ep:replicated dp:=custom_dp").dp_group_alias
            == "custom_dp"
        )


class TestExplicitReplicatedAxes:
    def test_single_replicated(self) -> None:
        result: DimsSpec = parse_dims("b s h[tp] d # ep:replicated")
        assert result.replicated_axes == frozenset({ParallelAxis.EP})

    def test_explicit_sharded_equivalent(self) -> None:
        assert parse_dims("b s h[tp:sharded] d").dims == parse_dims("b s h[tp] d").dims

    def test_multiple_replicated(self) -> None:
        result: DimsSpec = parse_dims("b s h[tp] d # ep:replicated cp:replicated")
        assert result.replicated_axes == frozenset({ParallelAxis.EP, ParallelAxis.CP})

    def test_dp_alias_and_replicated_coexist(self) -> None:
        result: DimsSpec = parse_dims("b s h[tp] d # dp:=moe_dp ep:replicated")
        assert result.dp_group_alias == "moe_dp"
        assert result.replicated_axes == frozenset({ParallelAxis.EP})

    def test_no_hash_replicated_empty(self) -> None:
        result: DimsSpec = parse_dims("b s h[tp] d")
        assert result.replicated_axes == frozenset()

    def test_hash_without_replicated(self) -> None:
        result: DimsSpec = parse_dims("b s h[tp] d # dp:=moe_dp")
        assert result.replicated_axes == frozenset()

    def test_replicated_conflicts_with_sharded_raises(self) -> None:
        with pytest.raises(ValueError, match="both sharded.*and replicated"):
            parse_dims("b s h[tp] d # tp:replicated")

    def test_unknown_axis_in_replicated_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown axis"):
            parse_dims("b s h[tp] d # xyz:replicated")

    def test_duplicate_replicated_declaration_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate replicated"):
            parse_dims("b s h d # ep:replicated ep:replicated")

    def test_unrecognized_token_in_comment_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized token"):
            parse_dims("b s h[tp] d # ep:replicatd")

    def test_duplicate_dp_alias_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate dp alias"):
            parse_dims("b s h d # dp:=foo dp:=bar")


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


class TestResolveDimNamesWithFused:
    def test_fused_dim_uses_triple_underscore(self) -> None:
        assert resolve_dim_names("t (num_heads*head_dim)") == [
            "t",
            "num_heads___head_dim",
        ]

    def test_fused_with_regular_dims(self) -> None:
        assert resolve_dim_names("t (num_heads*head_dim)[tp] d") == [
            "t",
            "num_heads___head_dim",
            "d",
        ]

    def test_three_way_fused(self) -> None:
        assert resolve_dim_names("(a*b*c)") == ["a___b___c"]

    def test_fused_with_squeeze(self) -> None:
        assert resolve_dim_names("t 1 (a*b)") == ["t", "singleton0", "a___b"]


class TestResolveDimNamesWithHash:
    def test_hash_stripped(self) -> None:
        assert resolve_dim_names("t h # dp:=moe_dp") == ["t", "h"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
