"""Policies that map hardware telemetry to scheduler knob overrides."""

from __future__ import annotations

import enum
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import msgspec

from sglang.srt.managers.hw_governor.knobs import KnobOverrides
from sglang.srt.managers.hw_governor.registry import resolve_class
from sglang.srt.managers.hw_governor.telemetry import ClockEventReason, DeviceTelemetry


class GovernorDecision(msgspec.Struct, frozen=True, kw_only=True):
    """A throttle level and the knob overrides it applies; level 0 means none."""

    level: int = 0
    overrides: KnobOverrides = KnobOverrides()


class GovernorPolicy(ABC):
    """Turns telemetry samples into a ``GovernorDecision``.

    Subclass to write a custom policy and select it with
    ``--hw-governor-policy package.module:ClassName``. The constructor receives
    the parsed ``--hw-governor-config`` JSON (``None`` when unset) and should
    reject an invalid config by raising ``ValueError``. ``update`` runs on the
    governor's background thread once per telemetry interval.
    """

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def update(
        self, samples: Sequence[DeviceTelemetry], *, now: float
    ) -> GovernorDecision: ...


# ---------------------------------------------------------------------------
# Stepwise policy
# ---------------------------------------------------------------------------

_THRESHOLD_METRICS: Dict[str, Callable[[DeviceTelemetry], Optional[float]]] = {
    "gpu_temp_c": lambda t: t.gpu_temp_c,
    "temp_margin_c": lambda t: t.temp_margin_c,
    "power_w": lambda t: t.power_w,
    "power_ratio": lambda t: t.power_ratio,
    "sm_clock_ratio": lambda t: t.sm_clock_ratio,
}
_CLOCK_EVENT_METRIC = "clock_event_reasons"


class TripRule(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One watched metric. ``trip > clear`` trips on high values, ``trip < clear`` on low ones.

    ``clock_event_reasons`` takes ``any_of`` (lowercase ``ClockEventReason``
    names) instead, and trips while any listed reason is active.
    """

    metric: str
    trip: Optional[float] = None
    clear: Optional[float] = None
    any_of: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.metric == _CLOCK_EVENT_METRIC:
            if not self.any_of or self.trip is not None or self.clear is not None:
                raise ValueError(
                    f"rule on {_CLOCK_EVENT_METRIC} takes a non-empty any_of "
                    "and no trip/clear"
                )
            unknown = [
                n for n in self.any_of if n.upper() not in ClockEventReason.__members__
            ]
            if unknown:
                raise ValueError(
                    f"unknown clock event reasons {unknown}; known: "
                    f"{sorted(n.lower() for n in ClockEventReason.__members__)}"
                )
            return
        if self.metric not in _THRESHOLD_METRICS:
            raise ValueError(
                f"unknown metric {self.metric!r}; known: "
                f"{sorted([*_THRESHOLD_METRICS, _CLOCK_EVENT_METRIC])}"
            )
        if self.trip is None or self.clear is None or self.any_of is not None:
            raise ValueError(f"rule on {self.metric} takes trip and clear, not any_of")
        if self.trip == self.clear:
            raise ValueError(
                f"rule on {self.metric} needs trip != clear; the gap is the hysteresis band"
            )

    @property
    def clock_event_mask(self) -> int:
        return functools.reduce(
            lambda mask, name: mask | ClockEventReason[name.upper()], self.any_of, 0
        )


# Arbitrary starting points, not measured: tune per deployment.
_DEFAULT_RULES = (
    TripRule(
        metric=_CLOCK_EVENT_METRIC,
        any_of=(
            "sw_thermal_slowdown",
            "hw_thermal_slowdown",
            "hw_slowdown",
            "hw_power_brake_slowdown",
        ),
    ),
    TripRule(metric="temp_margin_c", trip=5.0, clear=10.0),
)
_DEFAULT_LEVELS = (
    KnobOverrides(chunked_prefill_size=0.5, max_prefill_tokens=0.5),
    KnobOverrides(
        chunked_prefill_size=0.25, max_prefill_tokens=0.25, prefill_decode_interval=2
    ),
)


class StepwiseConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """``--hw-governor-config`` schema of the ``stepwise`` policy; omitted keys take defaults."""

    rules: Tuple[TripRule, ...] = _DEFAULT_RULES
    # levels[i] is the overrides of throttle level i + 1.
    levels: Tuple[KnobOverrides, ...] = _DEFAULT_LEVELS
    escalate_after_s: float = 3.0
    recover_after_s: float = 15.0

    def __post_init__(self) -> None:
        if not self.rules or not self.levels:
            raise ValueError("stepwise policy needs at least one rule and one level")
        if self.escalate_after_s < 0 or self.recover_after_s < 0:
            raise ValueError("escalate_after_s and recover_after_s must be >= 0")


class Verdict(enum.IntEnum):
    """Ordered so that aggregating rules and devices is ``max``."""

    COOL = 0
    HOLD = 1
    HOT = 2


class StepwiseState(msgspec.Struct, frozen=True, kw_only=True):
    level: int = 0
    # +1 while a hot streak runs, -1 while a cool one runs, 0 inside the band.
    streak: int = 0
    streak_start: float = 0.0


def rule_verdict(rule: TripRule, sample: DeviceTelemetry) -> Verdict:
    if rule.metric == _CLOCK_EVENT_METRIC:
        return (
            Verdict.HOT
            if sample.clock_event_reasons & rule.clock_event_mask
            else Verdict.COOL
        )
    value = _THRESHOLD_METRICS[rule.metric](sample)
    # A metric the device does not report never holds the throttle.
    if value is None:
        return Verdict.COOL
    sign = 1.0 if rule.trip > rule.clear else -1.0
    if sign * value >= sign * rule.trip:
        return Verdict.HOT
    if sign * value <= sign * rule.clear:
        return Verdict.COOL
    return Verdict.HOLD


def combined_verdict(
    rules: Sequence[TripRule], samples: Sequence[DeviceTelemetry]
) -> Verdict:
    # No samples (a failed read) counts as cool, so lost telemetry releases the throttle.
    return max(
        (rule_verdict(rule, sample) for rule in rules for sample in samples),
        default=Verdict.COOL,
    )


def next_stepwise_state(
    state: StepwiseState, verdict: Verdict, *, now: float, config: StepwiseConfig
) -> StepwiseState:
    """Move one level per sustained streak: hot for escalate_after_s, cool for recover_after_s."""
    if verdict == Verdict.HOLD:
        return StepwiseState(level=state.level)
    direction = 1 if verdict == Verdict.HOT else -1
    start = state.streak_start if state.streak == direction else now
    hold_s = config.escalate_after_s if direction > 0 else config.recover_after_s
    level = state.level
    target = min(max(level + direction, 0), len(config.levels))
    if target != level and now - start >= hold_s:
        level, start = target, now
    return StepwiseState(level=level, streak=direction, streak_start=start)


class StepwisePolicy(GovernorPolicy):
    """Trip rules select one of a ladder of knob overrides, one level at a time.

    Modeled on the Linux thermal framework's step_wise governor: a hot streak
    of ``escalate_after_s`` raises the level, a cool streak of
    ``recover_after_s`` lowers it, and readings between a rule's ``trip`` and
    ``clear`` hold the current level.
    """

    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        super().__init__(config)
        try:
            self.config = msgspec.convert(config or {}, StepwiseConfig)
        except msgspec.ValidationError as e:
            raise ValueError(f"Invalid --hw-governor-config: {e}") from e
        self._state = StepwiseState()

    def update(
        self, samples: Sequence[DeviceTelemetry], *, now: float
    ) -> GovernorDecision:
        verdict = combined_verdict(self.config.rules, samples)
        self._state = next_stepwise_state(
            self._state, verdict, now=now, config=self.config
        )
        level = self._state.level
        if level == 0:
            return GovernorDecision()
        return GovernorDecision(level=level, overrides=self.config.levels[level - 1])


_BUILTIN_POLICIES = {
    "stepwise": "sglang.srt.managers.hw_governor.policy:StepwisePolicy",
}


def resolve_policy_cls(spec: str) -> Type[GovernorPolicy]:
    return resolve_class(
        spec, builtins=_BUILTIN_POLICIES, base=GovernorPolicy, kind="governor policy"
    )
