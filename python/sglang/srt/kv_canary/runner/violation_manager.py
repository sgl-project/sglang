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
        outer_step_counter_getter: Callable[[], int],
    ) -> None:
        self._device_state = device_state
        self._outer_step_counter_getter = outer_step_counter_getter
        self._violation_reporter = ViolationReporter(
            config=config, device_state=device_state
        )
        self._handler = DelayedDeviceHostHandler(d2h_stream=d2h_stream)

    def step(self) -> None:
        drain_result: dict[str, bool] = {"errored": False}
        self._handler.step(
            compute_on_device=self._compute_on_device,
            postprocess_on_host=lambda host: drain_result.update(
                errored=bool(int(host.item()))
            ),
        )
        if drain_result["errored"] and not self._violation_reporter.is_raised:
            self._violation_reporter.log_or_raise_violation(
                outer_step_counter=self._outer_step_counter_getter()
            )

    def _compute_on_device(self) -> torch.Tensor:
        violation_log = self._device_state.violation_log
        return (violation_log.violation_write_index > 0).to(torch.uint8).view(-1)[:1]
