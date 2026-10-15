"""Which LoRA adapters a step can take: the drainer's fairness gate, then the
adapter slots (or the overlap loader)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.runtime_context import get_lora

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.bundles import Workers


class LoraAdmission:
    """Admits a LoRA request into a step; prefill admission and PD decode's
    preallocation both ask it."""

    def __init__(self, *, workers: Workers):
        self.tp_worker = workers.tp_worker
        if get_lora().lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                get_lora().max_loras_per_batch,
                get_lora().lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

        if get_lora().enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if get_lora().enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )
