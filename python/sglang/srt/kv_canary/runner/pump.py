from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner


class PumpAndAllreduce:
    def __init__(self, *, owner: "CanaryRunner") -> None:
        self._owner = owner
        device = owner._device
        use_cuda_pump = device.type == "cuda" and torch.cuda.is_available()
        self._pump_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if use_cuda_pump else None
        )
        self._pump_event: Optional[torch.cuda.Event] = (
            torch.cuda.Event() if use_cuda_pump else None
        )
        self._previous_pump_event: Optional[torch.cuda.Event] = None

    def pump_and_drain(self) -> bool:
        owner = self._owner

        if self._pump_stream is not None and self._pump_event is not None:
            violation_log = owner._device_state.violation_log
            default_stream = torch.cuda.current_stream(owner._device)
            self._pump_stream.wait_stream(default_stream)
            with torch.cuda.stream(self._pump_stream):
                signal = (violation_log.violation_write_index > 0).to(torch.uint8)
                owner._device_state.violation_signal_host.copy_(
                    signal.view(-1)[:1], non_blocking=True
                )
                self._pump_event.record()

        owner._step_counter += 1

        if self._previous_pump_event is not None:
            self._previous_pump_event.synchronize()
            local_errored = bool(int(owner._device_state.violation_signal_host.item()))
        else:
            local_errored = False
        self._previous_pump_event = self._pump_event
        if owner._device.type == "cuda" and torch.cuda.is_available():
            self._pump_event = torch.cuda.Event()

        any_rank_errored = local_errored
        allreduce_buf = owner._device_state.allreduce_buf
        if (
            owner.config.allreduce_violation_signal
            and allreduce_buf is not None
            and owner._tp_group is not None
            and dist.is_initialized()
        ):
            allreduce_buf.fill_(int(local_errored))
            dist.all_reduce(
                allreduce_buf,
                op=dist.ReduceOp.MAX,
                group=owner._tp_group.device_group,
            )
            any_rank_errored = bool(int(allreduce_buf.item()))

        return any_rank_errored
