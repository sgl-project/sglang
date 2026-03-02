from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class AxisInfo(_FrozenBase):
    axis_rank: int
    axis_size: int

    @model_validator(mode="after")
    def _validate_bounds(self) -> AxisInfo:
        if self.axis_size <= 0:
            raise ValueError(f"axis_size must be > 0, got {self.axis_size}")
        if not (0 <= self.axis_rank < self.axis_size):
            raise ValueError(
                f"axis_rank must be in [0, {self.axis_size}), got {self.axis_rank}"
            )
        return self


class ConcatParams(_FrozenBase):
    op: Literal["concat"] = "concat"
    dim_name: str


class CpThdConcatParams(_FrozenBase):
    op: Literal["cp_thd_concat"] = "cp_thd_concat"
    dim_name: str
    seq_lens_per_rank: list[int]  # per-seq token count on each rank, e.g. [50, 32, 46]


class PickParams(_FrozenBase):
    op: Literal["pick"] = "pick"


class ReduceSumParams(_FrozenBase):
    op: Literal["reduce_sum"] = "reduce_sum"


UnsharderParams = Annotated[
    Union[ConcatParams, CpThdConcatParams, PickParams, ReduceSumParams],
    Field(discriminator="op"),
]


class UnsharderPlan(_FrozenBase):
    type: Literal["unsharder"] = "unsharder"
    axis: ParallelAxis
    params: UnsharderParams
    # groups[i] = indices in the input tensor list, which will be operated (e.g. concat) into i-th output tensor.
    #
    # Multistep example (CP=2, TP=2, 4 input tensors):
    #   plan[0] (CP): groups=[[0,2],[1,3]]  — 4 tensors → 2 tensors
    #   plan[1] (TP): groups=[[0,1]]        — 2 tensors → 1 tensor
    groups: list[list[int]]
