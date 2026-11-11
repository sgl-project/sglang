"""Execution contracts shared by the DSpark planner and target executor.

The default policy preserves the scheduler's existing DP tier coordination.
Platform adapters may coordinate packed layouts at the target boundary instead;
all ranks, including idle ranks, must enter that boundary once per decode step.
"""

from typing import Optional

from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class DSparkExecutionPolicy:
    dynamic_verify_tiers_with_dp: bool = False
    coordinates_dp_verify: bool = False

    def sync_verify_budget(self, budget: Optional[int]) -> Optional[int]:
        """Agree within the TP group owning the same requests, if required."""
        return budget

    def coordinate_verify_layout(
        self, *, batch, layout: Optional[RaggedVerifyLayout]
    ) -> Optional[RaggedVerifyLayout]:
        """Agree on target shapes after drafting, including idle participants."""
        return layout

    def prepare_verify_forward_batch(self, forward_batch) -> None:
        """Publish the coordinated target token counts before model execution."""
