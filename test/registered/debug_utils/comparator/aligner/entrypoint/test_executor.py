import sys
from typing import Optional

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.entrypoint.executor import (
    AlignerResult,
    _execute_step_plans,
    execute_aligner_plan,
    execute_sub_plan,
    execute_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestExecuteSubPlans:
    def test_empty_tensors_returns_none(self) -> None:
        result: Optional[torch.Tensor] = execute_sub_plans(tensors=[], plans=[])
        assert result is None

    def test_no_plans_single_tensor_passthrough(self) -> None:
        tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        result: Optional[torch.Tensor] = execute_sub_plans(tensors=[tensor], plans=[])
        assert result is not None
        assert torch.equal(result, tensor)

    def test_no_plans_multiple_tensors_returns_none(self) -> None:
        tensors: list[torch.Tensor] = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]
        result: Optional[torch.Tensor] = execute_sub_plans(tensors=tensors, plans=[])
        assert result is None

    def test_with_unsharder_plan(self) -> None:
        t0: torch.Tensor = torch.tensor([[1.0, 2.0]])
        t1: torch.Tensor = torch.tensor([[3.0, 4.0]])

        plan = UnsharderPlan(
            axis=ParallelAxis.TP,
            params=ConcatParams(dim=1),
            groups=[[0, 1]],
        )

        result: Optional[torch.Tensor] = execute_sub_plans(
            tensors=[t0, t1], plans=[plan]
        )

        assert result is not None
        expected: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert torch.equal(result, expected)


class TestExecuteSubPlan:
    def test_unknown_plan_type_raises(self) -> None:
        class _FakePlan:
            pass

        with pytest.raises(NotImplementedError, match="Unknown"):
            execute_sub_plan(tensors=[torch.tensor([1.0])], plan=_FakePlan())  # type: ignore[arg-type]


class TestExecuteStepPlans:
    def test_step_with_none_result_omitted(self) -> None:
        tensors: list[torch.Tensor] = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]

        step_plan = AlignerPerStepPlan(
            step=0,
            input_object_indices=[0, 1],
            sub_plans=[],
        )

        result: dict[int, torch.Tensor] = _execute_step_plans(
            tensors=tensors, step_plans=[step_plan]
        )

        assert result == {}

    def test_single_step_passthrough(self) -> None:
        tensor: torch.Tensor = torch.tensor([1.0, 2.0])

        step_plan = AlignerPerStepPlan(
            step=5,
            input_object_indices=[0],
            sub_plans=[],
        )

        result: dict[int, torch.Tensor] = _execute_step_plans(
            tensors=[tensor], step_plans=[step_plan]
        )

        assert 5 in result
        assert torch.equal(result[5], tensor)


class TestExecuteAlignerPlan:
    def _make_step_plan(self, *, step: int, indices: list[int]) -> AlignerPerStepPlan:
        return AlignerPerStepPlan(step=step, input_object_indices=indices, sub_plans=[])

    def test_x_side_empty_returns_failed_x(self) -> None:
        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0, 1])],
                y=[self._make_step_plan(step=0, indices=[0])],
            ),
            token_aligner_plan=None,
        )

        tensors_pair: Pair[list[torch.Tensor]] = Pair(
            x=[torch.tensor([1.0]), torch.tensor([2.0])],
            y=[torch.tensor([3.0])],
        )

        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.tensors is None
        assert result.failed_side_xy == "x"

    def test_y_side_empty_returns_failed_y(self) -> None:
        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0])],
                y=[self._make_step_plan(step=0, indices=[0, 1])],
            ),
            token_aligner_plan=None,
        )

        tensors_pair: Pair[list[torch.Tensor]] = Pair(
            x=[torch.tensor([1.0])],
            y=[torch.tensor([2.0]), torch.tensor([3.0])],
        )

        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.tensors is None
        assert result.failed_side_xy == "y"

    def test_no_token_aligner_single_step(self) -> None:
        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0])],
                y=[self._make_step_plan(step=0, indices=[0])],
            ),
            token_aligner_plan=None,
        )

        t_x: torch.Tensor = torch.tensor([1.0, 2.0])
        t_y: torch.Tensor = torch.tensor([3.0, 4.0])
        tensors_pair: Pair[list[torch.Tensor]] = Pair(x=[t_x], y=[t_y])

        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.tensors is not None
        assert result.failed_side_xy is None
        assert torch.equal(result.tensors.x, t_x)
        assert torch.equal(result.tensors.y, t_y)

    def test_success_returns_none_failed_side(self) -> None:
        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0])],
                y=[self._make_step_plan(step=0, indices=[0])],
            ),
            token_aligner_plan=None,
        )

        tensors_pair: Pair[list[torch.Tensor]] = Pair(
            x=[torch.tensor([10.0])],
            y=[torch.tensor([20.0])],
        )

        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.failed_side_xy is None
        assert result.tensors is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
