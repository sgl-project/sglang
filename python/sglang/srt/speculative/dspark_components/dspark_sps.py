from __future__ import annotations

import bisect
from typing import Optional

import msgspec


def floor_probe_index(edges: list[int], batch_tokens: int) -> int:
    idx = bisect.bisect_right(edges, batch_tokens) - 1
    return max(0, min(idx, len(edges) - 1))


class SpsCostTable(msgspec.Struct, frozen=True):
    sample_batch_tokens: list[int]
    sample_steps_per_sec: list[float]
    max_batch_tokens: int

    def __post_init__(self) -> None:
        if not self.sample_batch_tokens:
            raise ValueError("SpsCostTable requires at least one probe.")
        if self.sample_batch_tokens != sorted(set(self.sample_batch_tokens)):
            raise ValueError(
                "sample_batch_tokens must be strictly increasing (monotone-sorted "
                f"invariant), got {self.sample_batch_tokens}."
            )
        if len(self.sample_batch_tokens) != len(self.sample_steps_per_sec):
            raise ValueError(
                "sample_batch_tokens and sample_steps_per_sec must have equal length, "
                f"got {len(self.sample_batch_tokens)} vs {len(self.sample_steps_per_sec)}."
            )
        if self.max_batch_tokens < self.sample_batch_tokens[-1]:
            raise ValueError(
                "max_batch_tokens must be >= the largest probe, got "
                f"{self.max_batch_tokens} < {self.sample_batch_tokens[-1]}."
            )

    def lookup(self, batch_tokens: int) -> float:
        return self.sample_steps_per_sec[
            floor_probe_index(self.sample_batch_tokens, batch_tokens)
        ]

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def _interp_clamped(xs: list[int], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_right(xs, x)
    lo = hi - 1
    frac = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + frac * (ys[hi] - ys[lo])


class SpsAdditiveCostTable(msgspec.Struct, frozen=True):

    bias_seconds: float
    bs_probes: list[int]
    alpha_seconds: list[float]
    m_probes: list[int]
    theta_seconds: list[float]

    def __post_init__(self) -> None:
        for name, probes, values in (
            ("bs", self.bs_probes, self.alpha_seconds),
            ("m", self.m_probes, self.theta_seconds),
        ):
            if not probes:
                raise ValueError(f"SpsAdditiveCostTable requires {name}_probes.")
            if probes != sorted(set(probes)):
                raise ValueError(
                    f"{name}_probes must be strictly increasing, got {probes}."
                )
            if len(probes) != len(values):
                raise ValueError(
                    f"{name}_probes and its values must have equal length, got "
                    f"{len(probes)} vs {len(values)}."
                )
        if self.bias_seconds <= 0:
            raise ValueError(f"bias_seconds must be > 0, got {self.bias_seconds}.")

    def step_time(self, *, num_reqs: int, budget: int) -> float:
        return (
            self.bias_seconds
            + _interp_clamped(self.bs_probes, self.alpha_seconds, float(num_reqs))
            + _interp_clamped(
                self.m_probes, self.theta_seconds, float(num_reqs + budget)
            )
        )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> SpsAdditiveCostTable:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def profile_sps_table(
    *,
    probes: list[tuple[int, float]],
    max_batch_tokens: Optional[int] = None,
) -> SpsCostTable:
    if not probes:
        raise ValueError("profile_sps_table requires at least one probe.")

    sorted_probes = sorted(probes, key=lambda probe: probe[0])

    sample_batch_tokens: list[int] = []
    sample_steps_per_sec: list[float] = []
    for batch_tokens, steps_per_sec in sorted_probes:
        batch_tokens = int(batch_tokens)
        if batch_tokens < 1:
            raise ValueError(
                f"profile_sps_table requires batch_tokens >= 1, got {batch_tokens}."
            )
        if sample_batch_tokens and batch_tokens == sample_batch_tokens[-1]:
            raise ValueError(
                "profile_sps_table requires unique batch_tokens per probe; "
                f"batch_tokens={batch_tokens} appears more than once. Median the "
                "repeated samples per batch_tokens before calling the assembler."
            )
        sample_batch_tokens.append(batch_tokens)
        sample_steps_per_sec.append(float(steps_per_sec))

    resolved_max = (
        int(max_batch_tokens)
        if max_batch_tokens is not None
        else sample_batch_tokens[-1]
    )
    return SpsCostTable(
        sample_batch_tokens=sample_batch_tokens,
        sample_steps_per_sec=sample_steps_per_sec,
        max_batch_tokens=resolved_max,
    )


def load_sps_table_from_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if '"bias_seconds"' in data:
        return SpsAdditiveCostTable.from_json(data)
    return SpsCostTable.from_json(data)


MIN_DERIVED_PROBES = 2
MIN_DERIVED_DYNAMIC_RANGE = 1.5
# SpsAdditiveCostTable requires a positive bias. The two terms that dominate a step -- the draft
# forward and the target verify -- are both measured below, and what remains is host-side work no
# captured graph covers. Rather than invent a value for it, the bias carries a floor only.
DERIVED_BIAS_SECONDS = 1e-9


def _monotone_non_decreasing(values: list[float]) -> list[float]:
    """Pool adjacent violators until the sequence never falls.

    Neither term of the cost model can get cheaper as its axis grows: a verify step does not speed
    up as it gets wider, and a draft step does not speed up as requests are added. Replay timings
    are noisy enough to violate that locally, and a curve that dips tells the planner a bigger step
    is faster -- the one lie that can make it choose a worse budget than verify-all.
    """
    pooled: list[float] = []
    weights: list[float] = []
    for value in values:
        pooled.append(value)
        weights.append(1.0)
        while len(pooled) > 1 and pooled[-2] > pooled[-1]:
            total = weights[-2] + weights[-1]
            merged = (pooled[-2] * weights[-2] + pooled[-1] * weights[-1]) / total
            pooled[-2:] = [merged]
            weights[-2:] = [total]
    out: list[float] = []
    for value, weight in zip(pooled, weights):
        out.extend([value] * int(weight))
    return out


def _pool_within_resolution(seconds: list[float], spreads: list[float]) -> list[float]:
    """Flatten runs of neighbours whose cost difference the measurement cannot resolve.

    A cost model must not encode a difference smaller than its own uncertainty. Over the low-token
    tiers the real verify cost is nearly flat, so without this the planner reads replay noise as a
    reason to trim -- and trimming where there is nothing to win is exactly the case that costs
    someone throughput for no gain. Pooling makes those tiers equal-cost, under which the budget
    objective is strictly increasing again and the planner declines to trim, which is the behaviour
    the flat region should produce.
    """
    pooled = list(seconds)
    start = 0
    for i in range(1, len(pooled) + 1):
        resolvable = i < len(pooled) and abs(pooled[i] - pooled[start]) > max(
            spreads[i], spreads[start]
        )
        if i == len(pooled) or resolvable:
            if i - start > 1:
                mean = sum(pooled[start:i]) / (i - start)
                pooled[start:i] = [mean] * (i - start)
            start = i
    return pooled


def _clean_cost_axis(
    *, probes: list[tuple[int, float, float]]
) -> Optional[tuple[list[int], list[float]]]:
    """Order one axis of the cost model and strip what the measurement cannot resolve."""
    unique: dict[int, tuple[float, float]] = {}
    for point, seconds, spread_seconds in probes:
        if point >= 1 and seconds > 0.0:
            unique[int(point)] = (float(seconds), abs(float(spread_seconds)))
    if not unique:
        return None
    points = sorted(unique)
    resolved = _pool_within_resolution(
        [unique[point][0] for point in points],
        [unique[point][1] for point in points],
    )
    return points, _monotone_non_decreasing(resolved)


def build_capture_derived_sps_table(
    *,
    verify_probes: list[tuple[int, float, float]],
    draft_probes: list[tuple[int, float, float]],
    bias_seconds: float = DERIVED_BIAS_SECONDS,
) -> Optional[SpsAdditiveCostTable]:
    """Build an additive cost model from measured (shape, seconds, spread_seconds) triples.

    `verify_probes` are keyed by total verify tokens and `draft_probes` by request count -- the two
    axes SpsAdditiveCostTable already separates, and the two graph ladders capture already provides.

    Keeping them apart is what makes the model correct rather than merely populated. The planner
    varies the budget at a *fixed* request count, and the draft forward has already run, at full
    width, for every one of those requests before a budget is chosen: it costs the same whichever
    budget wins. Folding it into a single token-indexed curve prices it as though trimming removed
    requests, which understates what trimming costs and biases the argmax towards it. A sunk
    constant is not neutral here, because the objective is a ratio.

    Returns None when the measurement cannot support a usable curve. That is a first-class outcome:
    the caller keeps the uninitialized table and the engine behaves exactly as it does today.
    """
    verify_axis = _clean_cost_axis(probes=verify_probes)
    if verify_axis is None:
        return None
    m_probes, theta_seconds = verify_axis
    if len(m_probes) < MIN_DERIVED_PROBES:
        return None
    if max(theta_seconds) / min(theta_seconds) < MIN_DERIVED_DYNAMIC_RANGE:
        # Too flat to carry a scheduling signal; a near-constant curve is the degenerate case this
        # replaces, so shipping it would add risk for no gain.
        return None

    # An unmeasurable draft ladder is not fatal: a zero per-request term is exactly the verify-only
    # model, which is a worse cost model but still a real curve, and still beats the uninitialized flat table.
    draft_axis = _clean_cost_axis(probes=draft_probes)
    bs_probes, alpha_seconds = draft_axis if draft_axis is not None else ([1], [0.0])

    return SpsAdditiveCostTable(
        bias_seconds=max(bias_seconds, DERIVED_BIAS_SECONDS),
        bs_probes=bs_probes,
        alpha_seconds=alpha_seconds,
        m_probes=m_probes,
        theta_seconds=theta_seconds,
    )


def build_uninitialized_sps_table(*, max_batch_tokens: int) -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1],
        sample_steps_per_sec=[1.0],
        max_batch_tokens=max_batch_tokens,
    )


def is_uninitialized_sps_table(table: SpsCostTable | SpsAdditiveCostTable) -> bool:
    if isinstance(table, SpsAdditiveCostTable):
        return False
    return len(table.sample_batch_tokens) <= 1
