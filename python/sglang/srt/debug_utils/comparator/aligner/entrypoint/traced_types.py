"""Traced wrapper types that embed execution traces (ShapeSnapshots) into plan nodes.

These types are created *after* execution, pairing each sub-plan with its
observed shape snapshot so that downstream formatters never need to manually
zip plan + trace by index.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepSubPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.output_types import ShapeSnapshot
from sglang.srt.debug_utils.comparator.utils import Pair, _StrictBase


class TracedSubPlan(_StrictBase):
    plan: AlignerPerStepSubPlan
    snapshot: Optional[ShapeSnapshot] = None


class TracedStepPlan(_StrictBase):
    step: int
    input_object_indices: list[int]
    sub_plans: list[TracedSubPlan]


class TracedSidePlan(_StrictBase):
    step_plans: list[TracedStepPlan]


class TracedAlignerPlan(_StrictBase):
    plan: AlignerPlan
    per_side: Pair[TracedSidePlan]
