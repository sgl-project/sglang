from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor
from sglang.srt.kv_canary.state import CanaryDeviceState


class ViolationSignalPump:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        d2h_stream: torch.cuda.Stream,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._d2h_stream = d2h_stream
        self._step_counter: int = 0
        self._previous_pump_future: Optional[FutureTensor] = None

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def pump_and_drain(self) -> bool:
        violation_log = self._device_state.violation_log
        signal = (violation_log.violation_write_index > 0).to(torch.uint8)

        local_errored = False
        if self._previous_pump_future is not None:
            local_errored = bool(int(self._previous_pump_future.wait().item()))
        self._previous_pump_future = FutureTensor.device_to_host(
            src_device=signal.view(-1)[:1], stream=self._d2h_stream
        )

        self._step_counter += 1

        return local_errored
