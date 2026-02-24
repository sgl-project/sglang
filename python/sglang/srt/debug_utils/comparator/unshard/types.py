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


class UnshardPlan(_FrozenBase):
    axis: ParallelAxis
    params: UnshardParams
    # Each inner list is one group of tensor indices to unshard together,
    # ordered by axis_rank. In a list[UnshardPlan], the first plan's indices
    # refer to positions in the original tensor list (by world_rank); each
    # subsequent plan's indices refer to the previous plan's output list.
    # Every plan's groups must cover all input indices exactly once; the
    # final plan must produce exactly one output tensor.
    groups: list[list[int]]


# Union of all plan types. Future pipeline components (e.g. reduction,
# reordering) will add their own plan types here.
Plan = UnshardPlan
