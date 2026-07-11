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
