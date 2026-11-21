"""Opt-in, request-specific adjustments to the scheduler's decode reservation.

These are admission heuristics, not generation limits or physical allocations.
The global new-token ratio still supplies decay, idle reset and retraction
feedback. Histories belong to one scheduler/model and are bounded independently
of request volume; LoRA and weight versions do not share observations.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.srt.managers.req_types import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


EFFORT_MULTIPLIERS = {
    "none": 0.5,
    "minimal": 0.625,
    "low": 0.75,
    "medium": 1.0,
    "high": 1.25,
    "xhigh": 1.375,
    "ultra": 1.5,
    "max": 1.5,
}
HISTORY_SIZE = 256
MIN_SAMPLES = 8
MAX_HISTORY_GROUPS = 128


def normalize_effort_hint(value: str | int | float | None) -> str | None:
    """Canonicalize explicit tiers and DeepSeek's numeric effort ranges.

    Unknown/model-default effort is deliberately not guessed. A native caller
    validates unknown hints; chat can fall back to the unchanged admission policy.
    """
    if isinstance(value, str):
        value = "none" if value == "no_think" else value
        return value if value in EFFORT_MULTIPLIERS else None
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not 0 <= value <= 0.99:
            return None
        value = max(1, round(value * 100))
    if not isinstance(value, int) or not 1 <= value <= 100:
        return None
    return "low" if value <= 50 else "high" if value <= 75 else "max"


def validate_effort_hint(value: str | int | float | None) -> str | None:
    """Validate before Pydantic can coerce JSON true into the integer 1."""
    if value is None:
        return None
    effort = normalize_effort_hint(value)
    if effort is None:
        raise ValueError("Invalid reasoning_effort_hint")
    return effort


@dataclass
class _OutputHistory:
    lengths: deque[int] = field(default_factory=deque)
    total: int = 0

    def add(self, length: int) -> None:
        if len(self.lengths) == HISTORY_SIZE:
            self.total -= self.lengths.popleft()
        self.lengths.append(length)
        self.total += length


class ReasoningEffortBudget:
    def __init__(
        self, mode: str, *, initial: float, minimum: float, estimation_cap: int = 4096
    ):
        if mode not in ("off", "scaled", "adaptive"):
            raise ValueError(f"Unknown reasoning effort KV budget mode: {mode}")
        self.mode = mode
        self.initial = initial
        self.minimum = minimum
        self.estimation_cap = estimation_cap
        self.histories: OrderedDict[
            tuple[str | None, str | None, str], _OutputHistory
        ] = OrderedDict()

    @staticmethod
    def _key(req: Req) -> tuple[str | None, str | None, str]:
        return (
            req.reasoning_budget_weight_version,
            req.lora_id,
            req.reasoning_effort_hint,
        )

    def ratio_for(self, req: Req, base_ratio: float) -> float:
        if req.sampling_params.ignore_eos:
            return 1.0
        effort = req.reasoning_effort_hint
        if self.mode == "off" or effort not in EFFORT_MULTIPLIERS:
            return base_ratio

        multiplier = EFFORT_MULTIPLIERS[effort]
        # Scaling r alone also scales its decrement. Instead scale the initial
        # reservation and divide the accumulated decrement by the same factor:
        # higher effort both reserves more and decays more slowly.
        anchor = min(1.0, self.initial * multiplier)
        decay_scale = 1.0 / multiplier
        if self.mode == "adaptive":
            history = self.histories.get(self._key(req))
            if history is not None and len(history.lengths) >= MIN_SAMPLES:
                maximum = req.sampling_params.max_new_tokens
                generated = len(req.output_ids)
                mean = min(maximum, history.total / len(history.lengths))
                remaining = mean - generated
                if remaining <= 0:
                    # Surviving past the mean is evidence against that estimate.
                    # Do not let a negative difference erase live decode demand.
                    return 1.0
                anchor = self.initial
                # Normalize against the same clipped demand that admission
                # charges. Huge user limits must not collapse every tier onto
                # the fastest decay even when their observed means differ.
                demand = min(maximum - generated, self.estimation_cap)
                decay_scale = min(
                    4.0,
                    max(0.25, self.initial * demand / remaining),
                )

        ratio = anchor - (self.initial - base_ratio) * decay_scale
        return min(1.0, max(self.minimum, ratio))

    def observe(self, req: Req) -> None:
        if (
            self.mode != "adaptive"
            or req.reasoning_budget_observed
            or req.reasoning_effort_hint not in EFFORT_MULTIPLIERS
            or req.finished_reason is None
            or isinstance(req.finished_reason, FINISH_ABORT)
            or req.sampling_params.ignore_eos
            or not req.output_ids
        ):
            return
        req.reasoning_budget_observed = True
        key = self._key(req)
        history = self.histories.get(key)
        if history is None:
            if len(self.histories) == MAX_HISTORY_GROUPS:
                self.histories.popitem(last=False)
            history = self.histories[key] = _OutputHistory()
        self.histories.move_to_end(key)
        # Count actual generated tokens, including natural length-limit finishes.
        # Aborted and forced-length benchmarking requests are excluded above.
        history.add(len(req.output_ids))
