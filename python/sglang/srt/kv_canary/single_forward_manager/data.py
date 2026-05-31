from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class PostOpsInsideGraphOutputBuffer:
    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        num_kernel_tags: int,
        num_slot_tags: int,
        device: torch.device,
    ) -> "PostOpsInsideGraphOutputBuffer":
        return cls(
            verify_plan_enable=torch.zeros(1, dtype=torch.int32, device=device),
            kernel_run_counters=torch.zeros(
                num_kernel_tags, dtype=torch.int64, device=device
            ),
            slot_run_counters=torch.zeros(
                num_slot_tags, dtype=torch.int64, device=device
            ),
            violation_write_index=torch.zeros(1, dtype=torch.int32, device=device),
        )

    def copy_from(
        self,
        *,
        verify_plan_enable: torch.Tensor,
        kernel_run_counters: torch.Tensor,
        slot_run_counters: torch.Tensor,
        violation_write_index: torch.Tensor,
    ) -> None:
        self.verify_plan_enable.copy_(verify_plan_enable)
        self.kernel_run_counters.copy_(kernel_run_counters)
        self.slot_run_counters.copy_(slot_run_counters)
        self.violation_write_index.copy_(violation_write_index)
