from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.kv_canary.runner.future_tensor import FutureTensors

logger = logging.getLogger(__name__)


class _CanaryEnableWarner:
    """Double-buffered host mirror of ``VerifyPlan.enable``. Each call to :meth:`tick` drains the
    previous step's async d2h (warn-logging when that step's plan kernel set enable=0 due to
    capacity overflow) and enqueues a new copy for the current step. Drain happens right before
    the slot is overwritten so the d2h gets a full forward pass of pipelining headroom.
    """

    def __init__(
        self, *, verify_capacity: int, d2h_stream: Optional[torch.cuda.Stream]
    ) -> None:
        self._verify_capacity = verify_capacity
        self._d2h_stream = d2h_stream
        self._pending_future: Optional[FutureTensors] = None
        self._overflow_count_total: int = 0

    def tick(self, enable_device: torch.Tensor) -> None:
        self._drain_previous()
        self._pending_future = FutureTensors.device_to_host(
            src_device=enable_device,
            stream=self._d2h_stream,
        )

    def _drain_previous(self) -> None:
        previous = self._pending_future
        if previous is None:
            return
        enable_value = int(previous.wait().item())
        if enable_value == 0:
            self._overflow_count_total += 1
            logger.warning(
                "kv-canary: per-forward verify skipped this step due to overflow "
                "(total=%d, capacity=%d); check ServerArgs / pool sizing",
                self._overflow_count_total,
                self._verify_capacity,
            )
