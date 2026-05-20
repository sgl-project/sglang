from __future__ import annotations

from typing import Optional

import torch


class DelayedD2HReadSlot:
    """One staged D2H copy with read-one-step-later semantics.

    Step N: stage(src_device) enqueues an async copy on the injected alt stream.
    Step N+1: pop() waits on the previous step's event and returns the host tensor.
    pop() returns None on the first call (no previous stage). When stream is None
    (no cuda available) stage() performs a sync copy and pop() returns the host
    tensor immediately after the first stage.
    """

    def __init__(
        self,
        *,
        host: torch.Tensor,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        self._host: torch.Tensor = host
        self._stream: Optional[torch.cuda.Stream] = stream
        self._previous_event: Optional[torch.cuda.Event] = None
        self._has_pending: bool = False

    def stage(self, *, src_device: torch.Tensor) -> None:
        if self._stream is None:
            self._host.copy_(src_device)
            self._previous_event = None
            self._has_pending = True
            return
        device = src_device.device
        default_stream = torch.cuda.current_stream(device)
        self._stream.wait_stream(default_stream)
        with torch.cuda.stream(self._stream):
            self._host.copy_(src_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        self._previous_event = event
        self._has_pending = True

    def pop(self) -> Optional[torch.Tensor]:
        if not self._has_pending:
            return None
        if self._previous_event is not None:
            self._previous_event.synchronize()
        self._has_pending = False
        self._previous_event = None
        return self._host
