# SPDX-License-Identifier: Apache-2.0
"""Vectorized FlowMatch-Euler solver for packed continuous batches.

Advances all packed rows in one fused update and keeps per-request scheduler
state consistent so requests can leave the batch at any step.

Packed execution has no per-request ``scheduler.step()`` fallback: only
schedulers accepted by :func:`scheduler_is_batchable` may join a packed
group, and a packed group that cannot build its solver is a hard error.
Requests with unsupported schedulers simply run unpacked; the reason is
surfaced through rate-limited diagnostics instead of a silent degradation.

A fused CUDA kernel for the update was intentionally not added: the packed
update is a couple of elementwise kernels on latents and profiling on the
target GPUs should justify a custom kernel before one is introduced.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.fallback_diagnostics import (
    fallback_diagnostics,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# First-order FlowMatch-Euler update: prev = sample + (sigma_next - sigma) * model_output.
_FLOW_MATCH_EULER_CLASS_NAMES = frozenset({"FlowMatchEulerDiscreteScheduler"})


class SolverRejection(Exception):
    """The batched solver cannot handle these schedulers (with the reason)."""


def _scheduler_rejection_reason(scheduler: Any) -> str | None:
    """Why this scheduler cannot use the vectorized solver, or None if it can."""
    if type(scheduler).__name__ not in _FLOW_MATCH_EULER_CLASS_NAMES:
        return f"unsupported scheduler {type(scheduler).__name__}"
    if getattr(scheduler, "order", 1) != 1:
        return "scheduler order != 1"
    config = getattr(scheduler, "config", None)
    if config is not None and getattr(config, "stochastic_sampling", False):
        return "stochastic_sampling is not vectorized"
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        return "scheduler sigmas are not a 1-D tensor"
    return None


def scheduler_is_batchable(scheduler: Any, *, diagnose: bool = False) -> bool:
    """Whether this scheduler may join a packed (vectorized) step group.

    With ``diagnose=True`` a rejection is recorded in the rate-limited
    fallback diagnostics so operators can see why requests run unpacked.
    """
    reason = _scheduler_rejection_reason(scheduler)
    if reason is None:
        return True
    if diagnose:
        fallback_diagnostics.record("batched-solver", type(scheduler).__name__, reason)
    return False


class _SigmaTableCache:
    """LRU cache of per-schedule sigma rows already resident on device.

    Keyed by the schedule content (hash of the fp32 sigmas) and target device
    so repeated group rebuilds with the same schedules skip the host-to-device
    copies. Tables are tiny ([num_steps + 1] fp32) so a small LRU suffices.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _fingerprint(sigmas: torch.Tensor) -> str:
        cpu = sigmas.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return hashlib.sha1(cpu.numpy().tobytes()).hexdigest()

    def get(self, sigmas: torch.Tensor, device: torch.device) -> torch.Tensor:
        key = (self._fingerprint(sigmas), str(device))
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return cached
            self.misses += 1
        row = sigmas.detach().to(device=device, dtype=torch.float32)
        with self._lock:
            self._entries[key] = row
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return row

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0


sigma_table_cache = _SigmaTableCache()


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
    ) -> BatchedFlowMatchEulerSolver:
        """Build a vectorized solver for member states or raise SolverRejection."""
        schedulers = [state.denoising_context.scheduler for state in states]
        for scheduler in schedulers:
            reason = _scheduler_rejection_reason(scheduler)
            if reason is not None:
                raise SolverRejection(reason)

        rows = [
            sigma_table_cache.get(scheduler.sigmas, device) for scheduler in schedulers
        ]
        max_len = max(int(row.shape[0]) for row in rows)
        # +1 so sigma_{i+1} at the final step stays in range.
        table_len = max_len + 1
        padded_rows = []
        for row in rows:
            pad = table_len - int(row.shape[0])
            if pad > 0:
                row = torch.cat([row, row[-1:].expand(pad)])
            padded_rows.append(row)
        sigma_table = torch.stack(padded_rows)

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
) -> BatchedFlowMatchEulerSolver:
    """Build the batched solver for a packed group.

    Raises :class:`SolverRejection` when any member's scheduler is not
    vectorizable. Packed groups must not form around such schedulers, so a
    rejection here means the packability pre-screen was bypassed.
    """
    if not states:
        raise SolverRejection("no member states")
    return BatchedFlowMatchEulerSolver.try_build(states, device)
