from __future__ import annotations

from typing import Literal

from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class AxisInfo(_FrozenBase):
    axis_rank: int
    axis_size: int


class ConcatParams(_FrozenBase):
    op: Literal["concat"] = "concat"
    dim: int


# Phase 2: add ReduceSumParams, CpZigzagParams here, then change UnshardParams to:
#   UnshardParams = Annotated[
#       Union[ConcatParams, ReduceSumParams, CpZigzagParams],
#       Field(discriminator="op"),
#   ]
UnshardParams = ConcatParams


class UnshardStep(_FrozenBase):
    axis: ParallelAxis
    params: UnshardParams
    world_ranks_by_axis_rank: list[int]


class UnshardPlan(_FrozenBase):
    replicated_axes: dict[str, AxisInfo] = {}
    steps: list[UnshardStep] = []
    pick_world_ranks: frozenset[int]
