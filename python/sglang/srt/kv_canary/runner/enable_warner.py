from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler

logger = logging.getLogger(__name__)


class CanaryEnableWarner:
    def __init__(
        self, *, verify_capacity: int, d2h_stream: Optional[torch.cuda.Stream]
    ) -> None:
        self._verify_capacity = verify_capacity
        self._overflow_count_total: int = 0
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def tick(self, enable_device: torch.Tensor) -> None:
        self._handler.step(
            compute_on_device=lambda: enable_device,
            postprocess_on_host=self._postprocess_on_host,
        )

    def _postprocess_on_host(self, host_tensor: torch.Tensor) -> None:
        if int(host_tensor.item()) == 0:
            self._overflow_count_total += 1
            logger.warning(
                "kv-canary: per-forward verify skipped this step due to overflow "
                "(total=%d, capacity=%d); check ServerArgs / pool sizing",
                self._overflow_count_total,
                self._verify_capacity,
            )
