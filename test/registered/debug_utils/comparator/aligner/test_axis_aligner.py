import sys
from typing import Optional

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import (
    AxisAlignerPlan,
    compute_axis_aligner_plan,
    execute_axis_aligner_plan,
)
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
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
        with warning_sink.context() as warnings:
            result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
                Pair(x="t h d", y="t h e")
            )

        assert result is None
        assert len(warnings) == 1
        assert warnings[0].category == "axis_aligner_dim_mismatch"
        assert "dim name sets differ" in warnings[0].message

    def test_modifiers_ignored_for_name_extraction(self) -> None:
        result: Optional[AxisAlignerPlan] = compute_axis_aligner_plan(
            Pair(x="t h(tp) d", y="t d h(tp)")
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


class TestExecuteAxisAlignerPlan:
    def test_rearrange(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16).refine_names("t", "h", "d")
        plan = AxisAlignerPlan(
            pattern=Pair(x="t h d -> t d h", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 16, 8)
        for i in range(4):
            assert torch.equal(
                result[i],
                tensor.rename(None)[i].T,
            )

    def test_execute_squeeze(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8).refine_names("t", "singleton0", "h")
        plan = AxisAlignerPlan(
            pattern=Pair(x="t 1 h -> t h", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 8)

    def test_execute_squeeze_then_swap(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8, 16).refine_names(
            "t", "singleton0", "h", "d"
        )
        plan = AxisAlignerPlan(
            pattern=Pair(x="t 1 h d -> t d h", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="x"
        )

        assert result.shape == (4, 16, 8)

    def test_execute_y_side(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 1, 8).refine_names("t", "singleton0", "h")
        plan = AxisAlignerPlan(
            pattern=Pair(x=None, y="t 1 h -> t h"),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="y"
        )

        assert result.shape == (4, 8)

    def test_noop_side(self) -> None:
        torch.manual_seed(42)
        tensor: torch.Tensor = torch.randn(4, 8, 16).refine_names("t", "h", "d")
        plan = AxisAlignerPlan(
            pattern=Pair(x="t h d -> t d h", y=None),
        )

        result: torch.Tensor = execute_axis_aligner_plan(
            tensor=tensor, plan=plan, side="y"
        )

        assert result.shape == (4, 8, 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
