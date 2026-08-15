from typing import Annotated, Literal, Union

from pydantic import Field

from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class ZigzagToNaturalParams(_FrozenBase):
    op: Literal["zigzag_to_natural"] = "zigzag_to_natural"
    dim_name: str
    cp_size: int


class ZigzagToNaturalThdParams(_FrozenBase):
    op: Literal["zigzag_to_natural_thd"] = "zigzag_to_natural_thd"
    dim_name: str
    cp_size: int
    seq_lens: list[int]  # unshard-ed per-seq token counts, e.g. [100, 64, 92]


ReordererParams = Annotated[
    Union[ZigzagToNaturalParams, ZigzagToNaturalThdParams],
    Field(discriminator="op"),
]


class ReordererPlan(_FrozenBase):
    type: Literal["reorderer"] = "reorderer"
    params: ReordererParams
