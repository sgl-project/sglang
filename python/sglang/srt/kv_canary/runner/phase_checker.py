from __future__ import annotations

from enum import IntEnum

import torch


class CanaryPerForwardPhase(IntEnum):
    """Lifecycle phases of one canary per-forward-pass.

    Enforced order::

        IDLE
          -> AFTER_BEFORE_FORWARD      (PerForwardOrchestrator.before_forward)
          -> AFTER_HEAD_KERNELS        (PerForwardOrchestrator.launch_head_kernels)
          -> AFTER_TAIL_KERNELS        (PerForwardOrchestrator.launch_tail_kernels)
          -> IDLE                      (PerForwardOrchestrator.end_of_step)

    Any other transition (skipped step, re-entry, out-of-order) is a bug and
    must surface at the next sync point.
    """

    IDLE = 0
    AFTER_BEFORE_FORWARD = 1
    AFTER_HEAD_KERNELS = 2
    AFTER_TAIL_KERNELS = 3


class CanaryPerForwardPhaseChecker:
    """GPU-side state machine for the per-forward-pass lifecycle.

    The phase value lives in a 1-element int32 tensor on GPU. Every transition
    issues two async device ops:

    1. ``torch._assert_async(phase == expect_phase)`` — async assert; the error
       surfaces at the next host sync (e.g. the d2h drain at ``_end_of_step``),
       not immediately. This is the same pattern spec_utils.maybe_detect_nan/oob
       use so the check is cuda-graph-safe.
    2. ``phase.fill_(next_phase)`` — single async kernel writing the scalar to
       GPU memory. Captured into the graph and replayed verbatim.

    Both ops are safe under ``torch.cuda.graph(...)`` capture and replay, so the
    same checker runs unmodified inside and outside cuda graphs (EAGLE draft
    cuda graph included)."""

    def __init__(self, *, device: torch.device) -> None:
        self._phase = torch.tensor(
            CanaryPerForwardPhase.IDLE.value, dtype=torch.int32, device=device
        )

    def update(
        self,
        *,
        expect_phase: CanaryPerForwardPhase,
        next_phase: CanaryPerForwardPhase,
    ) -> None:
        torch._assert_async(
            self._phase == expect_phase.value,
            f"kv-canary: per-forward phase mismatch — expected {expect_phase.name} "
            f"before transitioning to {next_phase.name}",
        )
        self._phase.fill_(next_phase.value)
