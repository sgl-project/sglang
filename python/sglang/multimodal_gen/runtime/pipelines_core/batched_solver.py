# SPDX-License-Identifier: Apache-2.0
"""Vectorized FlowMatch-Euler solver for packed continuous batches.

Advances all packed rows in one fused update and keeps per-request scheduler
state consistent so requests can leave the batch at any step.
"""

from __future__ import annotations

from typing import Any

import torch

# First-order FlowMatch-Euler update: prev = sample + (sigma_next - sigma) * model_output.
_FLOW_MATCH_EULER_CLASS_NAMES = frozenset({"FlowMatchEulerDiscreteScheduler"})


def _scheduler_is_plain_flow_match_euler(scheduler: Any) -> bool:
    if type(scheduler).__name__ not in _FLOW_MATCH_EULER_CLASS_NAMES:
        return False
    if getattr(scheduler, "order", 1) != 1:
        return False
    config = getattr(scheduler, "config", None)
    if config is not None and getattr(config, "stochastic_sampling", False):
        return False
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        return False
    return True


class BatchedFlowMatchEulerSolver:
    """Fused FlowMatch-Euler step across packed rows."""

    def __init__(
        self,
        *,
        sigma_table: torch.Tensor,
        start_indices: torch.Tensor,
        row_index: torch.Tensor | None,
        num_rows: int,
    ) -> None:
        # [B, T_max + 1] padded with each row's final sigma for safe gathers.
        self._sigma_table = sigma_table
        # [B] step indices at build time.
        self._start_indices = start_indices
        # Maps packed rows to member index; None for 1 row per member.
        self._row_index = row_index
        self._num_rows = num_rows
        self._steps_taken = 0

    @classmethod
    def try_build(
        cls,
        states: list[Any],
        device: torch.device,
    ) -> BatchedFlowMatchEulerSolver | None:
        """Build a vectorized solver for member states, or return None."""
        schedulers = [state.denoising_context.scheduler for state in states]
        if not all(_scheduler_is_plain_flow_match_euler(s) for s in schedulers):
            return None

        sigma_rows: list[torch.Tensor] = []
        max_len = 0
        for scheduler in schedulers:
            sigmas = scheduler.sigmas
            max_len = max(max_len, int(sigmas.shape[0]))
        # +1 so sigma_{i+1} at the final step stays in range.
        table_len = max_len + 1
        for scheduler in schedulers:
            sigmas = scheduler.sigmas.to(dtype=torch.float32)
            pad = table_len - int(sigmas.shape[0])
            if pad > 0:
                sigmas = torch.cat([sigmas, sigmas[-1:].expand(pad)])
            sigma_rows.append(sigmas)
        sigma_table = torch.stack(sigma_rows).to(device=device, non_blocking=True)

        start_indices = torch.tensor(
            [int(state.step_index) for state in states],
            dtype=torch.long,
            device=device,
        )

        row_counts = [int(state.denoising_context.latents.shape[0]) for state in states]
        num_rows = sum(row_counts)
        if all(count == 1 for count in row_counts):
            row_index = None
        else:
            row_index = torch.repeat_interleave(
                torch.arange(len(states), device=device),
                torch.tensor(row_counts, device=device),
            )
        return cls(
            sigma_table=sigma_table,
            start_indices=start_indices,
            row_index=row_index,
            num_rows=num_rows,
        )

    def scale_model_input(self, packed_sample: torch.Tensor) -> torch.Tensor:
        """FlowMatch-Euler does not scale the sample."""
        return packed_sample

    def step_rows(
        self,
        packed_model_output: torch.Tensor,
        packed_sample: torch.Tensor,
        states: list[Any],
    ) -> torch.Tensor:
        """Advance every packed row by one solver step."""
        indices = self._start_indices + self._steps_taken
        sigma = self._sigma_table.gather(1, indices.unsqueeze(1)).squeeze(1)
        sigma_next = self._sigma_table.gather(1, (indices + 1).unsqueeze(1)).squeeze(1)
        delta = sigma_next - sigma
        if self._row_index is not None:
            delta = delta.index_select(0, self._row_index)
        delta = delta.view(-1, *([1] * (packed_sample.ndim - 1)))

        prev_sample = packed_sample.to(torch.float32) + delta * packed_model_output
        prev_sample = prev_sample.to(packed_model_output.dtype)

        self._steps_taken += 1
        for state in states:
            # Mirror the completed step index into the scheduler counter.
            state.denoising_context.scheduler._step_index = int(state.step_index) + 1
        return prev_sample


def build_batched_solver(
    states: list[Any],
    device: torch.device,
) -> BatchedFlowMatchEulerSolver | None:
    """Build a batched solver, falling back to None on any error."""
    if not states:
        return None
    try:
        return BatchedFlowMatchEulerSolver.try_build(states, device)
    except Exception:
        return None
