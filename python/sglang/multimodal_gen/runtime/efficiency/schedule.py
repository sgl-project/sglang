# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Schedule[T] -- a time-varying value over the denoising trajectory.
#
# Every efficiency-technique parameter is a *function of (step, stage)* rather
# than a scalar, so policies like "first two steps high precision, the rest
# low" or "dense for the warmup steps then sparse" are expressed declaratively
# as first-class values. Schedules are the time sub-DSL of the efficiency
# framework (see technique.py / compose.py).

from __future__ import annotations

from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


def parse_steps(spec: str | Iterable[int] | None) -> set[int]:
    """Parse a step spec into a set of indices.

    Accepts a set/list/range of ints, or a string like "1-2,5,7-9". An empty
    or ``None`` spec yields the empty set (matches no step).
    """
    if spec is None:
        return set()
    if isinstance(spec, str):
        out: set[int] = set()
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                lo, _, hi = tok.partition("-")
                try:
                    lo_i, hi_i = int(lo), int(hi)
                except ValueError:
                    continue
                if hi_i < lo_i:
                    lo_i, hi_i = hi_i, lo_i
                out.update(range(lo_i, hi_i + 1))
            else:
                try:
                    out.add(int(tok))
                except ValueError:
                    continue
        return out
    return {int(x) for x in spec}


class Schedule(Generic[T]):
    """A value resolved per (step, stage). Build via the constructors below."""

    def __init__(self, fn: Callable[[int, str], T], desc: str = "schedule"):
        self._fn = fn
        self._desc = desc

    def at(self, step: int, stage: str = "") -> T:
        return self._fn(step, stage)

    # the set of steps over [0, horizon) where this schedule is truthy; used by
    # compose() to decide whether two techniques can be co-active.
    def truthy_steps(self, horizon: int = 64) -> set[int]:
        return {s for s in range(horizon) if bool(self.at(s))}

    def __repr__(self) -> str:
        return f"Schedule({self._desc})"


def const(value: T) -> Schedule[T]:
    return Schedule(lambda step, stage="": value, f"const={value!r}")


def at_steps(steps: str | Iterable[int], value: T, default: T) -> Schedule[T]:
    """``value`` on the listed steps, ``default`` elsewhere."""
    sset = parse_steps(steps)
    return Schedule(
        lambda step, stage="": value if step in sset else default,
        f"at_steps({sorted(sset)}->{value!r} else {default!r})",
    )


def before(n: int, value: T, then: T) -> Schedule[T]:
    """``value`` for steps < n (e.g. warmup / high-precision prefix), ``then`` after."""
    return Schedule(
        lambda step, stage="": value if step < n else then,
        f"before({n}:{value!r} then {then!r})",
    )


def predicate(fn: Callable[[int, str], bool], value: T, default: T) -> Schedule[T]:
    return Schedule(
        lambda step, stage="": value if fn(step, stage) else default, "predicate"
    )


def by_stage(mapping: dict[str, "Schedule[T] | T"], default: T) -> Schedule[T]:
    """Pick a per-stage schedule/constant (e.g. different policy in stage1 vs stage2)."""

    def _resolve(step: int, stage: str = "") -> T:
        v = mapping.get(stage, default)
        return v.at(step, stage) if isinstance(v, Schedule) else v

    return Schedule(_resolve, f"by_stage({list(mapping)})")


def as_schedule(value: "Schedule[T] | T") -> Schedule[T]:
    """Coerce a bare value into a const schedule (so callers may pass either)."""
    return value if isinstance(value, Schedule) else const(value)
