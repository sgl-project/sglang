from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureTensor:
    _tensor: torch.Tensor
    _event: Optional[torch.cuda.Event]

    @classmethod
    def create(
        cls, *, src_device: torch.Tensor, stream: Optional[torch.cuda.Stream]
    ) -> "FutureTensor":
        host = torch.empty(
            src_device.shape,
            dtype=src_device.dtype,
            pin_memory=(stream is not None),
        )
        if stream is None:
            host.copy_(src_device)
            return cls(_tensor=host, _event=None)
        stream.wait_stream(torch.cuda.current_stream(src_device.device))
        with torch.cuda.stream(stream):
            host.copy_(src_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return cls(_tensor=host, _event=event)

    def wait(self) -> torch.Tensor:
        if self._event is not None:
            self._event.synchronize()
        return self._tensor
