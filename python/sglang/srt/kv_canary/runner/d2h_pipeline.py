from __future__ import annotations

from typing import Optional

import torch


class CanaryD2HPipeline:
    """One alt cuda stream + event pair used to issue all kv-canary D2H copies off the main
    stream. The caller submits a copy via stage(), and reads the result via wait() one step later
    (or whenever the host actually needs it).
    """

    def __init__(self, *, device: torch.device) -> None:
        self._device: torch.device = device
        self._enabled: bool = device.type == "cuda" and torch.cuda.is_available()
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if self._enabled else None
        )

    def stage(
        self,
        *,
        dst_host: torch.Tensor,
        src_device: torch.Tensor,
    ) -> Optional[torch.cuda.Event]:
        """Enqueue ``dst_host.copy_(src_device, non_blocking=True)`` on the alt stream.
        Returns the recorded event the caller stores; pass it back to wait().
        Returns None when no cuda device is available (caller falls back to sync read).
        """
        if not self._enabled or self._stream is None:
            dst_host.copy_(src_device)
            return None
        default_stream = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(default_stream)
        with torch.cuda.stream(self._stream):
            dst_host.copy_(src_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return event

    @staticmethod
    def wait(event: Optional[torch.cuda.Event]) -> None:
        """Block the calling thread until the staged D2H is observable on host."""
        if event is not None:
            event.synchronize()
