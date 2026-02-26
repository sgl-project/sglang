from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
    TokenAlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.utils import Pair

AlignerPerStepSubPlan = Union[UnsharderPlan, ReordererPlan]


@dataclass(frozen=True)
class AlignerPerStepPlan:
    step: int
    input_object_indices: list[int]
    sub_plans: list[AlignerPerStepSubPlan]


@dataclass(frozen=True)
class AlignerPlan:
    per_step_plans: Pair[list[AlignerPerStepPlan]]
    token_aligner_plan: Optional[TokenAlignerPlan]
