from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureTensor:
    _tensor: torch.Tensor
    _event: torch.cuda.Event

    @classmethod
    def create(
        cls, *, src_device: torch.Tensor, stream: torch.cuda.Stream
    ) -> "FutureTensor":
        host = torch.empty(
            src_device.shape,
            dtype=src_device.dtype,
            pin_memory=True,
        )
        stream.wait_stream(torch.cuda.current_stream(src_device.device))
        with torch.cuda.stream(stream):
            host.copy_(src_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return cls(_tensor=host, _event=event)

    def wait(self) -> torch.Tensor:
        self._event.synchronize()
        return self._tensor
