from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sglang.srt.debug_utils.comparator.dims import ParallelAxis


class _PlanBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AxisInfo(_PlanBase):
    axis_rank: int
    axis_size: int


class ConcatParams(_PlanBase):
    op: Literal["concat"] = "concat"
    dim: int


# Phase 2: add ReduceSumParams, CpZigzagParams here, then change UnshardParams to:
#   UnshardParams = Annotated[
#       Union[ConcatParams, ReduceSumParams, CpZigzagParams],
#       Field(discriminator="op"),
#   ]
UnshardParams = ConcatParams


class UnshardStep(_PlanBase):
    axis: ParallelAxis
    params: UnshardParams
    world_ranks_by_axis_rank: list[int]


class UnshardPlan(_PlanBase):
    tensor_name: str
    dims_str: str
    replicated_axes: dict[str, AxisInfo] = {}
    steps: list[UnshardStep] = []
    pick_world_ranks: frozenset[int]
