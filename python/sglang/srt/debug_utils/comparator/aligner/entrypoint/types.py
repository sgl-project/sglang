from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import Discriminator

from sglang.srt.debug_utils.comparator.aligner.axis_aligner import AxisAlignerPlan
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase

AlignerPerStepSubPlan = Annotated[
    Union[UnsharderPlan, ReordererPlan],
    Discriminator("type"),
]


class AlignerPerStepPlan(_FrozenBase):
    step: int
    input_object_indices: list[int]
    sub_plans: list[AlignerPerStepSubPlan]


class AlignerPlan(_FrozenBase):
    per_step_plans: Pair[list[AlignerPerStepPlan]]
    token_aligner_mode: Optional[str] = None  # "concat_steps" | "smart" | None
    token_aligner_plan: Optional[TokenAlignerPlan] = None
    axis_aligner_plan: Optional[AxisAlignerPlan] = None
