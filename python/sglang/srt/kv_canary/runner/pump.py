from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.future_tensor import FutureTensor, stage_d2h_future
from sglang.srt.kv_canary.state import CanaryDeviceState

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class PumpAndAllreduce:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        tp_group: Optional["GroupCoordinator"],
        d2h_stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._config = config
        self._device = device
        self._device_state = device_state
        self._tp_group = tp_group
        self._d2h_stream = d2h_stream
        self._step_counter: int = 0
        self._previous_pump_future: Optional[FutureTensor] = None
        self._previous_allreduce_future: Optional[FutureTensor] = None

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def pump_and_drain(self) -> bool:
        violation_log = self._device_state.violation_log
        signal = (violation_log.violation_write_index > 0).to(torch.uint8)

        local_errored = False
        if self._previous_pump_future is not None:
            local_errored = bool(int(self._previous_pump_future.wait().item()))
        self._previous_pump_future = stage_d2h_future(
            src_device=signal.view(-1)[:1], stream=self._d2h_stream
        )

        self._step_counter += 1

        any_rank_errored = local_errored
        allreduce_buf = self._device_state.allreduce_buf
        if (
            self._config.allreduce_violation_signal
            and allreduce_buf is not None
            and self._tp_group is not None
            and dist.is_initialized()
        ):
            allreduce_buf.fill_(int(local_errored))
            dist.all_reduce(
                allreduce_buf,
                op=dist.ReduceOp.MAX,
                group=self._tp_group.device_group,
            )
            if self._previous_allreduce_future is not None:
                any_rank_errored = bool(
                    int(self._previous_allreduce_future.wait().item())
                )
            else:
                any_rank_errored = local_errored
            self._previous_allreduce_future = stage_d2h_future(
                src_device=allreduce_buf, stream=self._d2h_stream
            )

        return any_rank_errored
