from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import DelayedDeviceHostHandler
from sglang.srt.kv_canary.runner.violation_reporter import ViolationReporter
from sglang.srt.kv_canary.state import CanaryDeviceState


class ViolationManager:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        d2h_stream: torch.cuda.Stream,
        step_counter_getter: Callable[[], int],
    ) -> None:
        self._device_state = device_state
        self._step_counter_getter = step_counter_getter
        self._violation_reporter = ViolationReporter(
            config=config, device_state=device_state
        )
        self._last_drained_errored: bool = False
        self._handler = DelayedDeviceHostHandler(
            compute_on_device=self._compute_on_device,
            postprocess_on_host=self._postprocess_on_host,
            d2h_stream=d2h_stream,
        )

    def step(self) -> None:
        self._last_drained_errored = False
        self._handler.step()
        if self._last_drained_errored and not self._violation_reporter.is_raised:
            self._violation_reporter.log_or_raise_violation(
                step_counter=self._step_counter_getter()
            )

    def _compute_on_device(self) -> torch.Tensor:
        violation_log = self._device_state.violation_log
        return (violation_log.violation_write_index > 0).to(torch.uint8).view(-1)[:1]

    def _postprocess_on_host(self, host_tensor: torch.Tensor) -> None:
        self._last_drained_errored = bool(int(host_tensor.item()))
