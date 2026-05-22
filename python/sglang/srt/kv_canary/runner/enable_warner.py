from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler

logger = logging.getLogger(__name__)


class _CanaryEnableWarner:
    """Double-buffered host mirror of ``VerifyPlan.enable``. Each call to :meth:`tick` stages
    a fresh async d2h and (on the next tick) drains the previous step's host snapshot,
    warning when that step's plan kernel set enable=0 due to capacity overflow.
    """

    def __init__(
        self, *, verify_capacity: int, d2h_stream: Optional[torch.cuda.Stream]
    ) -> None:
        self._verify_capacity = verify_capacity
        self._overflow_count_total: int = 0
        self._current_enable_device: Optional[torch.Tensor] = None
        self._handler = DelayedDeviceHostHandler(
            compute_on_device=self._compute_on_device,
            postprocess_on_host=self._postprocess_on_host,
            d2h_stream=d2h_stream,
        )

    def tick(self, enable_device: torch.Tensor) -> None:
        self._current_enable_device = enable_device
        self._handler.step()

    def _compute_on_device(self) -> Optional[torch.Tensor]:
        return self._current_enable_device

    def _postprocess_on_host(self, host_tensor: torch.Tensor) -> None:
        if int(host_tensor.item()) == 0:
            self._overflow_count_total += 1
            logger.warning(
                "kv-canary: per-forward verify skipped this step due to overflow "
                "(total=%d, capacity=%d); check ServerArgs / pool sizing",
                self._overflow_count_total,
                self._verify_capacity,
            )
