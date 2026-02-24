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
    # groups[i] = indices in the input tensor list, which will be operated (e.g. concat) into i-th output tensor.
    #
    # Multistep example (CP=2, TP=2, 4 input tensors):
    #   plan[0] (CP): groups=[[0,2],[1,3]]  — 4 tensors → 2 tensors
    #   plan[1] (TP): groups=[[0,1]]        — 2 tensors → 1 tensor
    groups: list[list[int]]


# Union of all plan types. Future pipeline components (e.g. reduction,
# reordering) will add their own plan types here.
Plan = UnshardPlan
