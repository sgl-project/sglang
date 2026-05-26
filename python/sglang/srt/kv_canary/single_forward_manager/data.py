from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class PostOpsInsideGraphOutputBuffer:
    verify_plan_enable: torch.Tensor
    kernel_run_counters: torch.Tensor
    slot_run_counters: torch.Tensor
    violation_write_index: torch.Tensor
    swa_verify_total_count: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        *,
        num_kernel_tags: int,
        num_slot_tags: int,
        swa_verify_total_count_shape: tuple[int, ...] | None,
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
            swa_verify_total_count=(
                None
                if swa_verify_total_count_shape is None
                else torch.zeros(
                    swa_verify_total_count_shape, dtype=torch.int32, device=device
                )
            ),
        )

    def copy_from(
        self,
        *,
        verify_plan_enable: torch.Tensor,
        kernel_run_counters: torch.Tensor,
        slot_run_counters: torch.Tensor,
        violation_write_index: torch.Tensor,
        swa_verify_total_count: torch.Tensor | None,
    ) -> None:
        self.verify_plan_enable.copy_(verify_plan_enable)
        self.kernel_run_counters.copy_(kernel_run_counters)
        self.slot_run_counters.copy_(slot_run_counters)
        self.violation_write_index.copy_(violation_write_index)
        assert (self.swa_verify_total_count is not None) == (
            swa_verify_total_count is not None
        )
        if self.swa_verify_total_count is not None:
            self.swa_verify_total_count.copy_(swa_verify_total_count)
