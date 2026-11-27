# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import msgspec
import torch


def prepare_sol_beta(
    *,
    states: dict[torch.device, SolBetaState],
    step: int,
    total_steps: int,
    is_warmup: bool,
    device: torch.device,
    config: dict,
) -> torch.Tensor | None:
    spec = config.get("beta_schedule")
    if spec is None:
        return None
    spec = str(spec)
    if is_warmup:
        schedule = BetaSchedule.from_spec(spec)
        if schedule.values:
            total_steps = len(schedule.values)
    state = states.get(device)
    if state is None:
        state = SolBetaState.create(spec, total_steps, device)
        states[device] = state
    elif state.spec != spec or state.total_steps != total_steps:
        updated = SolBetaState.create(spec, total_steps, device)
        state.spec = updated.spec
        state.total_steps = updated.total_steps
        state.table = updated.table
    return state.update(step)


class BetaSchedule(msgspec.Struct, frozen=True):
    """One routing threshold per denoising step."""

    endpoints: tuple[float, float] | None = None
    values: tuple[float, ...] = ()

    @classmethod
    def from_spec(cls, spec: str) -> BetaSchedule:
        text = spec.strip()
        if ":" in text:
            first, last = text.split(":", 1)
            return cls(endpoints=(float(first), float(last)))
        values = tuple(float(part) for part in text.split(",") if part.strip())
        if not values:
            raise ValueError(
                "beta_schedule must be 'first:last' for a linear ramp or one "
                f"comma-separated value per step, got {spec!r}."
            )
        if len(values) == 1:
            return cls(endpoints=(values[0], values[0]))
        return cls(values=values)

    def beta_at(self, step: int, total_steps: int) -> float:
        if self.values:
            return self.values[min(step, len(self.values) - 1)]
        first, last = self.endpoints
        if total_steps <= 1:
            return first
        position = min(step, total_steps - 1) / (total_steps - 1)
        return first + (last - first) * position


class SolBetaState(msgspec.Struct):
    """Stage-owned beta storage whose address remains valid for captured graphs."""

    spec: str
    total_steps: int
    table: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def create(cls, spec: str, total_steps: int, device: torch.device) -> SolBetaState:
        schedule = BetaSchedule.from_spec(spec)
        if total_steps <= 0:
            raise ValueError("beta_schedule requires a positive denoising step count.")
        if schedule.values and len(schedule.values) != total_steps:
            raise ValueError(
                f"beta_schedule lists {len(schedule.values)} values for "
                f"{total_steps} denoising steps."
            )
        table = torch.tensor(
            [schedule.beta_at(step, total_steps) for step in range(total_steps)],
            dtype=torch.float32,
            device=device,
        )
        return cls(spec, total_steps, table, torch.empty_like(table[0]))

    def update(self, step: int) -> torch.Tensor:
        self.beta.copy_(self.table[step % self.total_steps])
        return self.beta
