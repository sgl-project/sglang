import sys

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
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
    TokenLocator,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, TokenLayout
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


class TestExecuteSubPlans:
    def test_empty_tensors_returns_none(self) -> None:
        result, checks = execute_sub_plans(tensors=[], plans=[])
        assert result is None
        assert checks == []

    def test_no_plans_single_tensor_passthrough(self) -> None:
        tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        result, checks = execute_sub_plans(tensors=[tensor], plans=[])
        assert result is not None
        assert torch.equal(result, tensor)
        assert checks == []

    def test_no_plans_multiple_tensors_returns_none(self) -> None:
        tensors: list[torch.Tensor] = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        ]
        result, checks = execute_sub_plans(tensors=tensors, plans=[])
        assert result is None
        assert checks == []

    def test_with_unsharder_plan(self) -> None:
        t0: torch.Tensor = torch.tensor([[1.0, 2.0]]).refine_names("b", "h")
        t1: torch.Tensor = torch.tensor([[3.0, 4.0]]).refine_names("b", "h")

        plan = UnsharderPlan(
            axis=ParallelAxis.TP,
            params=ConcatParams(dim_name="h"),
            groups=[[0, 1]],
        )

        result, checks = execute_sub_plans(tensors=[t0, t1], plans=[plan])

        assert result is not None
        expected: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert torch.equal(result.rename(None), expected)
        assert checks == []


class TestExecuteSubPlan:
    def test_unknown_plan_type_raises(self) -> None:
        class _FakePlan:
            pass

        with pytest.raises(NotImplementedError, match="Unknown"):
            execute_sub_plan(tensors=[torch.tensor([1.0])], plan=_FakePlan())


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

        result, checks = _execute_step_plans(tensors=tensors, step_plans=[step_plan])

        assert result == {}
        assert checks == []

    def test_single_step_passthrough(self) -> None:
        tensor: torch.Tensor = torch.tensor([1.0, 2.0])

        step_plan = AlignerPerStepPlan(
            step=5,
            input_object_indices=[0],
            sub_plans=[],
        )

        result, checks = _execute_step_plans(tensors=[tensor], step_plans=[step_plan])

        assert 5 in result
        assert torch.equal(result[5], tensor)
        assert checks == []


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


class TestExecuteAlignerPlanWithTokenDim:
    """End-to-end tests for AlignerPlan with non-zero token_dim."""

    def _make_step_plan(self, *, step: int, indices: list[int]) -> AlignerPerStepPlan:
        return AlignerPerStepPlan(step=step, input_object_indices=indices, sub_plans=[])

    def test_token_dim_nonzero_e2e(self) -> None:
        """AlignerPlan with token at dim 1 passes through to token aligner correctly."""
        torch.manual_seed(42)

        # shape [3, 4, 8]: dim0=a, dim1=token(4 tokens), dim2=hidden
        tensor_x: torch.Tensor = torch.randn(3, 4, 8).refine_names("a", "t", "h")
        tensor_y: torch.Tensor = torch.randn(3, 4, 8).refine_names("a", "t", "h")

        locator_x = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 1, 2],
        )
        locator_y = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 1, 2],
        )
        token_plan = TokenAlignerPlan(
            locators=Pair(x=locator_x, y=locator_y),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0])],
                y=[self._make_step_plan(step=0, indices=[0])],
            ),
            token_aligner_mode="smart",
            token_aligner_plan=token_plan,
        )

        tensors_pair: Pair[list[torch.Tensor]] = Pair(x=[tensor_x], y=[tensor_y])
        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.tensors is not None
        assert result.failed_side_xy is None
        # token dim stays at dim 1 -> shape [3, 3, 8] (3 tokens selected from 4)
        assert result.tensors.x.shape == (3, 3, 8)
        assert result.tensors.y.shape == (3, 3, 8)

        plain_x: torch.Tensor = tensor_x.rename(None)
        plain_y: torch.Tensor = tensor_y.rename(None)
        for i in range(3):
            assert torch.equal(
                result.tensors.x.select(dim=1, index=i),
                plain_x.select(dim=1, index=i),
            )
            assert torch.equal(
                result.tensors.y.select(dim=1, index=i),
                plain_y.select(dim=1, index=i),
            )

    def test_bshd_cross_layout_e2e(self) -> None:
        """x=SGLang THD, y=Megatron BSHD: planner->executor full flow."""
        torch.manual_seed(42)

        # x side: THD layout, shape [6, 8] (6 tokens, hidden=8), pre-named
        tensor_x: torch.Tensor = torch.randn(6, 8).refine_names("t", "h")

        # y side: BSHD layout, shape [2, 3, 8] (B=2, S=3, H=8), pre-named
        tensor_y: torch.Tensor = torch.randn(2, 3, 8).refine_names("b", "s", "h")
        flat_y: torch.Tensor = tensor_y.rename(None).reshape(6, 8)

        locator = TokenLocator(
            steps=[0, 0, 0],
            token_index_in_step=[0, 2, 5],
        )
        token_plan = TokenAlignerPlan(
            locators=Pair(x=locator, y=locator),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.BS),
        )

        plan = AlignerPlan(
            per_step_plans=Pair(
                x=[self._make_step_plan(step=0, indices=[0])],
                y=[self._make_step_plan(step=0, indices=[0])],
            ),
            token_aligner_mode="smart",
            token_aligner_plan=token_plan,
        )

        tensors_pair: Pair[list[torch.Tensor]] = Pair(x=[tensor_x], y=[tensor_y])
        result: AlignerResult = execute_aligner_plan(
            tensors_pair=tensors_pair, plan=plan
        )

        assert result.tensors is not None
        assert result.failed_side_xy is None

        assert result.tensors.x.shape == (3, 8)
        assert result.tensors.y.shape == (3, 8)

        plain_x: torch.Tensor = tensor_x.rename(None)
        assert torch.equal(result.tensors.x[0], plain_x[0])
        assert torch.equal(result.tensors.x[1], plain_x[2])
        assert torch.equal(result.tensors.x[2], plain_x[5])

        assert torch.equal(result.tensors.y[0], flat_y[0])
        assert torch.equal(result.tensors.y[1], flat_y[2])
        assert torch.equal(result.tensors.y[2], flat_y[5])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
