from typing import Literal

from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class ZigzagToNaturalParams(_FrozenBase):
    op: Literal["zigzag_to_natural"] = "zigzag_to_natural"
    dim: int
    cp_size: int


ReordererParams = ZigzagToNaturalParams


class ReordererPlan(_FrozenBase):
    params: ReordererParams
