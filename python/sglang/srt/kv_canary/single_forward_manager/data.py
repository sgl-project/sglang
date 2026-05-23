from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class PostOpsInsideGraphOutputSnapshot:
    """Per-SingleForwardManager cloned view of the in-graph signals produced by phases 2-3.

    The snapshot is written by phase 3 (``post_ops_maybe_inside_graph``)
    and read by phase 4 (``post_ops_outside_graph``). It captures the
    in-graph signals (verify-plan enable flag, kernel/slot counters,
    violation write index, swa verify totals) whose live device-state
    might be mutated by later steps in the same cycle. ForwardBatch
    fields are NOT cloned here — perturb / divergence consumers in
    phase 4 read the live (possibly inaccurate) ``ForwardBatch`` instead.
    """

    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor
    swa_verify_total_count: torch.Tensor | None
