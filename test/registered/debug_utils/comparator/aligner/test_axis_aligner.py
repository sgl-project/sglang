import sys
from typing import Optional

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    AxisAlignerPlan,
    compute_axis_aligner_plan,
    execute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.log_sink import log_sink
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestComputeAxisAlignerPlan:
    def test_no_dims_returns_none(self) -> None:
        assert compute_axis_aligner_plan(Pair(x=None, y=None)) is None
        assert compute_axis_aligner_plan(Pair(x="t h d", y=None)) is None
        assert compute_axis_aligner_plan(Pair(x=None, y="t h d")) is None

    def test_same_order_returns_none(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t h d", y="t h d")
        )
        assert result is None

    def test_different_order(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t h d", y="t d h")
        )
        assert result is not None
        assert result.pattern.x == "t h d -> t d h"
        assert result.pattern.y is None

    def test_name_mismatch_returns_none_with_warning(self) -> None:
        with log_sink.context() as warnings:
            result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
                Pair(x="t h d", y="t h e")
            )

        assert result is None
        assert len(warnings) == 1
        assert warnings[0].category == "axis_aligner_dim_mismatch"
        assert "dim name sets differ" in warnings[0].message

    def test_modifiers_ignored_for_name_extraction(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t h[tp] d", y="t d h[tp]")
        )
        assert result is not None
        assert result.pattern.x == "t h d -> t d h"

    def test_squeeze_only_no_swap(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t 1 h", y="t h")
        )
        assert result is not None
        assert result.pattern.x == "t 1 h -> t h"
        assert result.pattern.y is None

    def test_squeeze_both_sides(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t 1 h", y="1 t h")
        )
        assert result is not None
        assert result.pattern.x == "t 1 h -> t h"
        assert result.pattern.y == "1 t h -> t h"

    def test_squeeze_plus_swap(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t 1 h d", y="t d h")
        )
        assert result is not None
        assert result.pattern.x == "t 1 h d -> t d h"
        assert result.pattern.y is None

    def test_squeeze_y_only(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t h", y="t 1 h")
        )
        assert result is not None
        assert result.pattern.x is None
        assert result.pattern.y == "t 1 h -> t h"

    def test_multiple_squeeze_one_side(self) -> None:
        """Two squeeze dims on x, none on y."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="1 t 1 h", y="t h")
        )
        assert result is not None
        assert result.pattern.x == "1 t 1 h -> t h"
        assert result.pattern.y is None

    def test_multiple_squeeze_asymmetric(self) -> None:
        """Different numbers of squeeze dims on each side."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="1 t 1 h", y="1 t h")
        )
        assert result is not None
        assert result.pattern.x == "1 t 1 h -> t h"
        assert result.pattern.y == "1 t h -> t h"

    def test_four_dim_full_reversal(self) -> None:
        """4-dim permutation: full reversal."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="a b c d", y="d c b a")
        )
        assert result is not None
        assert result.pattern.x == "a b c d -> d c b a"
        assert result.pattern.y is None


class TestComputeAxisAlignerPlanFused:
    def test_fused_vs_separate(self) -> None:
        """x=fused 2D, y=separate 3D: y flattens to match x's fused form."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t (num_heads*head_dim)[tp]", y="t num_heads[tp] head_dim")
        )
        assert result is not None
        assert result.pattern.x is None
        assert result.pattern.y == "t num_heads head_dim -> t (num_heads head_dim)"

    def test_separate_vs_fused(self) -> None:
        """x=separate 3D, y=fused 2D: x flattens to match y's fused form."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t num_heads[tp] head_dim", y="t (num_heads*head_dim)[tp]")
        )
        assert result is not None
        assert result.pattern.x == "t num_heads head_dim -> t (num_heads head_dim)"
        assert result.pattern.y is None

    def test_both_fused_same_no_plan(self) -> None:
        """Both sides fused, same order → None (no-op)."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t (a*b)", y="t (a*b)")
        )
        assert result is None

    def test_fused_name_mismatch_returns_none(self) -> None:
        """Fused vs separate with mismatched names → None."""
        with log_sink.context() as warnings:
            result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
                Pair(x="t (a*b)", y="t c d")
            )
        assert result is None
        assert len(warnings) == 1

    def test_partial_fused_and_regular(self) -> None:
        """x has "(a*b) c", y has "a b c": y flattens a,b to match x's fused form."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="(a*b) c", y="a b c")
        )
        assert result is not None
        assert result.pattern.x is None
        assert result.pattern.y == "a b c -> (a b) c"

    def test_fused_vs_reordered_separate(self) -> None:
        """x=fused "(a*b) c", y=reordered separate "b a c": y flattens+reorders."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="(a*b) c", y="b a c")
        )
        assert result is not None
        assert result.pattern.x is None
        assert result.pattern.y == "b a c -> (a b) c"

    def test_fused_reorder_both_sides(self) -> None:
        """x=fused "c (a*b)", y=separate "a b c": x reorders fused, y flattens."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="c (a*b)", y="a b c")
        )
        assert result is not None
        assert result.pattern.x == "c a___b -> a___b c"
        assert result.pattern.y == "a b c -> (a b) c"

    def test_fused_with_squeeze(self) -> None:
        """Fused + squeeze on one side, separate on other."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t 1 (a*b)", y="t a b")
        )
        assert result is not None
        assert result.pattern.x == "t 1 a___b -> t a___b"
        assert result.pattern.y == "t a b -> t (a b)"

    def test_three_way_fused_vs_separate(self) -> None:
        """3-way fused on x, separate on y."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t (a*b*c)", y="t a b c")
        )
        assert result is not None
        assert result.pattern.x is None
        assert result.pattern.y == "t a b c -> t (a b c)"

    def test_separate_vs_three_way_fused(self) -> None:
        """Separate on x, 3-way fused on y."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t a b c", y="t (a*b*c)")
        )
        assert result is not None
        assert result.pattern.x == "t a b c -> t (a b c)"
        assert result.pattern.y is None

    def test_both_fused_different_order(self) -> None:
        """Both sides fused same group but dims in different order."""
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="c (a*b)", y="(a*b) c")
        )
        assert result is not None
        assert result.pattern.x == "c a___b -> a___b c"
        assert result.pattern.y is None

    def test_overlapping_fused_groups_returns_none(self) -> None:
        """x fuses (a*b), y fuses (b*c): incompatible overlap → None with warning."""
        with log_sink.context() as warnings:
            result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
                Pair(x="(a*b) c", y="a (b*c)")
            )
        assert result is None
        assert len(warnings) == 1
        assert warnings[0].category == "axis_aligner_fused_conflict"
        assert "overlapping fused groups" in warnings[0].message


class TestExecuteAxisAlignerPlan:
    def test_rearrange(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16).refine_names("t", "h", "d")
        plan = AxisAlignerPlan(pattern=Pair(x="t h d -> t d h", y=None))

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 16, 8)
        for i in range(4):
            assert torch.equal(result[i], tensor.rename(None)[i].T)

    def test_execute_squeeze(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8).refine_names("t", "singleton0", "h")
        plan = AxisAlignerPlan(pattern=Pair(x="t 1 h -> t h", y=None))

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 8)

    def test_execute_squeeze_then_swap(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8, 16).refine_names(
            "t", "singleton0", "h", "d"
        )
        plan = AxisAlignerPlan(pattern=Pair(x="t 1 h d -> t d h", y=None))

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 16, 8)

    def test_execute_y_side(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8).refine_names("t", "singleton0", "h")
        plan = AxisAlignerPlan(pattern=Pair(x=None, y="t 1 h -> t h"))

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="y"
        )

        assert result.shape == (4, 8)

    def test_noop_side(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16).refine_names("t", "h", "d")
        plan = AxisAlignerPlan(pattern=Pair(x="t h d -> t d h", y=None))

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="y"
        )

        assert result.shape == (4, 8, 16)

    def test_invalid_side_raises(self) -> None:
        """Invalid side value should raise ValueError."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16)
        plan = AxisAlignerPlan(pattern=Pair(x="t h d -> t d h", y=None))

        with pytest.raises(ValueError, match="side must be"):
            execute_axis_aligner_plan(tensor=tensor, plan=plan, side="z")


class TestExecuteAxisAlignerPlanFlatten:
    def test_flatten_separate_to_match_fused(self) -> None:
        """3D (t=4, nh=8, hd=16) → 2D (t=4, nh*hd=128) via einops flatten."""
        torch.manual_seed(42)
        tensor_3d: torch.Tensor = torch.randn(4, 8, 16)
        plan = AxisAlignerPlan(
            pattern=Pair(x=None, y="t nh hd -> t (nh hd)"),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor_3d, plan=plan, side="y"
        )

        assert result.shape == (4, 128)
        assert torch.equal(result, tensor_3d.reshape(4, 128))

    def test_flatten_preserves_data(self) -> None:
        """Flatten should be equivalent to reshape — verify element equality."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(2, 3, 4, 5)
        plan = AxisAlignerPlan(
            pattern=Pair(x="a b c d -> a (b c) d", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (2, 12, 5)
        assert torch.equal(result, tensor.reshape(2, 12, 5))

    def test_flatten_then_rearrange(self) -> None:
        """Flatten + reorder in a single einops pattern."""
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16, 32)
        plan = AxisAlignerPlan(
            pattern=Pair(x="t a b d -> t d (a b)", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 32, 128)


class TestEndToEndFusedAlignment:
    def test_fused_vs_separate_full_pipeline(self) -> None:
        """Full pipeline: x=fused 2D "t nh*hd", y=separate 3D "t nh hd"."""
        torch.manual_seed(42)
        num_heads: int = 8
        head_dim: int = 16

        x_tensor: torch.Tensor = torch.randn(4, num_heads * head_dim)
        y_tensor: torch.Tensor = x_tensor.reshape(4, num_heads, head_dim)

        plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t (num_heads*head_dim)", y="t num_heads head_dim")
        )
        assert plan is not None

        y_aligned: torch.Tensor = execute_axis_aligner_plan(
            tensor=y_tensor, plan=plan, side="y"
        )

        assert y_aligned.shape == x_tensor.shape
        assert torch.equal(y_aligned, x_tensor)

    def test_separate_vs_fused_full_pipeline(self) -> None:
        """Full pipeline: x=separate 3D "t nh hd", y=fused 2D "t nh*hd"."""
        torch.manual_seed(42)
        num_heads: int = 8
        head_dim: int = 16

        x_tensor: torch.Tensor = torch.randn(4, num_heads, head_dim)
        y_tensor: torch.Tensor = x_tensor.reshape(4, num_heads * head_dim)

        plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t num_heads head_dim", y="t (num_heads*head_dim)")
        )
        assert plan is not None

        x_aligned: torch.Tensor = execute_axis_aligner_plan(
            tensor=x_tensor, plan=plan, side="x"
        )

        assert x_aligned.shape == y_tensor.shape
        assert torch.equal(x_aligned, y_tensor)

    def test_fused_with_reorder(self) -> None:
        """Fused x + reordered separate y: both need alignment."""
        torch.manual_seed(42)
        a_size: int = 3
        b_size: int = 5

        # x: fused "c a*b" shape (7, 15)
        x_tensor: torch.Tensor = torch.randn(7, a_size * b_size)
        # y: separate "a b c" shape (3, 5, 7)
        y_tensor: torch.Tensor = x_tensor.reshape(7, a_size, b_size).permute(1, 2, 0)

        plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="c (a*b)", y="a b c")
        )
        assert plan is not None

        x_aligned: torch.Tensor = execute_axis_aligner_plan(
            tensor=x_tensor, plan=plan, side="x"
        )
        y_aligned: torch.Tensor = execute_axis_aligner_plan(
            tensor=y_tensor, plan=plan, side="y"
        )

        assert x_aligned.shape == y_aligned.shape
        assert torch.allclose(x_aligned, y_aligned)


class TestEndToEndThreeWayFused:
    def test_three_way_fused_vs_separate(self) -> None:
        """Full pipeline: x=3-way fused "t (a*b*c)", y=separate "t a b c"."""
        torch.manual_seed(42)
        a_size, b_size, c_size = 2, 3, 4

        x_tensor: torch.Tensor = torch.randn(5, a_size * b_size * c_size)
        y_tensor: torch.Tensor = x_tensor.reshape(5, a_size, b_size, c_size)

        plan: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t (a*b*c)", y="t a b c")
        )
        assert plan is not None

        y_aligned: torch.Tensor = execute_axis_aligner_plan(
            tensor=y_tensor, plan=plan, side="y"
        )

        assert y_aligned.shape == x_tensor.shape
        assert torch.equal(y_aligned, x_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
