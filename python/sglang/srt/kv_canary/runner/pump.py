from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.kv_canary.runner.d2h_pipeline import CanaryD2HPipeline

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner


class PumpAndAllreduce:
    def __init__(self, *, owner: "CanaryRunner") -> None:
        self._owner = owner
        self._d2h: CanaryD2HPipeline = CanaryD2HPipeline(device=owner._device)
        self._previous_pump_event: Optional[torch.cuda.Event] = None
        self._previous_allreduce_event: Optional[torch.cuda.Event] = None

    def pump_and_drain(self) -> bool:
        owner = self._owner

        violation_log = owner._device_state.violation_log
        signal = (violation_log.violation_write_index > 0).to(torch.uint8)
        pump_event = self._d2h.stage(
            dst_host=owner._device_state.violation_signal_host,
            src_device=signal.view(-1)[:1],
        )

        owner._step_counter += 1

        if self._previous_pump_event is not None:
            CanaryD2HPipeline.wait(self._previous_pump_event)
            local_errored = bool(int(owner._device_state.violation_signal_host.item()))
        else:
            local_errored = False
        self._previous_pump_event = pump_event

        any_rank_errored = local_errored
        allreduce_buf = owner._device_state.allreduce_buf
        allreduce_signal_host = owner._device_state.allreduce_signal_host
        if (
            owner.config.allreduce_violation_signal
            and allreduce_buf is not None
            and allreduce_signal_host is not None
            and owner._tp_group is not None
            and dist.is_initialized()
        ):
            allreduce_buf.fill_(int(local_errored))
            dist.all_reduce(
                allreduce_buf,
                op=dist.ReduceOp.MAX,
                group=owner._tp_group.device_group,
            )
            allreduce_event = self._d2h.stage(
                dst_host=allreduce_signal_host,
                src_device=allreduce_buf,
            )
            if self._previous_allreduce_event is not None:
                CanaryD2HPipeline.wait(self._previous_allreduce_event)
                any_rank_errored = bool(int(allreduce_signal_host.item()))
            else:
                any_rank_errored = local_errored
            self._previous_allreduce_event = allreduce_event

        return any_rank_errored
