from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.d2h_slot import DelayedD2HReadSlot
from sglang.srt.kv_canary.state import CanaryDeviceState, CanaryHostState

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class PumpAndAllreduce:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device: torch.device,
        device_state: CanaryDeviceState,
        host_state: CanaryHostState,
        tp_group: Optional["GroupCoordinator"],
        d2h_stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._config = config
        self._device = device
        self._device_state = device_state
        self._host_state = host_state
        self._tp_group = tp_group
        self._step_counter: int = 0
        self._pump_slot: DelayedD2HReadSlot = DelayedD2HReadSlot(
            host=host_state.violation_signal_host,
            stream=d2h_stream,
        )
        allreduce_signal_host = host_state.allreduce_signal_host
        self._allreduce_slot: Optional[DelayedD2HReadSlot] = (
            DelayedD2HReadSlot(host=allreduce_signal_host, stream=d2h_stream)
            if allreduce_signal_host is not None
            else None
        )

    @property
    def step_counter(self) -> int:
        return self._step_counter

    def pump_and_drain(self) -> bool:
        violation_log = self._device_state.violation_log
        signal = (violation_log.violation_write_index > 0).to(torch.uint8)

        prev_pump = self._pump_slot.pop()
        self._pump_slot.stage(src_device=signal.view(-1)[:1])

        self._step_counter += 1

        local_errored = bool(int(prev_pump.item())) if prev_pump is not None else False

        any_rank_errored = local_errored
        allreduce_buf = self._device_state.allreduce_buf
        if (
            self._config.allreduce_violation_signal
            and allreduce_buf is not None
            and self._allreduce_slot is not None
            and self._tp_group is not None
            and dist.is_initialized()
        ):
            allreduce_buf.fill_(int(local_errored))
            dist.all_reduce(
                allreduce_buf,
                op=dist.ReduceOp.MAX,
                group=self._tp_group.device_group,
            )
            prev_allreduce = self._allreduce_slot.pop()
            self._allreduce_slot.stage(src_device=allreduce_buf)
            if prev_allreduce is not None:
                any_rank_errored = bool(int(prev_allreduce.item()))
            else:
                any_rank_errored = local_errored

        return any_rank_errored
