from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
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
        self._d2h_stream = d2h_stream
        self._step_counter_getter = step_counter_getter
        self._violation_reporter = ViolationReporter(
            config=config, device_state=device_state
        )
        self._previous_pump_future: Optional[FutureTensor] = None

    def step(self) -> None:
        any_rank_errored = self._pump()
        if any_rank_errored and not self._violation_reporter.is_raised:
            self._violation_reporter.log_or_raise_violation(
                step_counter=self._step_counter_getter()
            )

    def _pump(self) -> bool:
        violation_log = self._device_state.violation_log
        signal = (violation_log.violation_write_index > 0).to(torch.uint8)

        local_errored = False
        if self._previous_pump_future is not None:
            local_errored = bool(int(self._previous_pump_future.wait().item()))
        self._previous_pump_future = FutureTensor.device_to_host(
            src_device=signal.view(-1)[:1], stream=self._d2h_stream
        )

        return local_errored
