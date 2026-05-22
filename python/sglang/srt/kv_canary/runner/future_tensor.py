from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

_TensorOrDict = Union[torch.Tensor, dict[str, torch.Tensor]]


@dataclass(slots=True, kw_only=True)
class FutureTensors:
    _tensors: Optional[_TensorOrDict]
    _event: Optional[torch.cuda.Event]

    @classmethod
    def device_to_host(
        cls, src_device: _TensorOrDict, *, stream: torch.cuda.Stream
    ) -> "FutureTensors":
        if isinstance(src_device, dict):
            host: _TensorOrDict = {
                key: torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                for key, t in src_device.items()
            }
            ref_device = next(iter(src_device.values())).device
        else:
            host = torch.empty(
                src_device.shape, dtype=src_device.dtype, pin_memory=True
            )
            ref_device = src_device.device

        stream.wait_stream(torch.cuda.current_stream(ref_device))
        with torch.cuda.stream(stream):
            if isinstance(src_device, dict):
                for key, t in src_device.items():
                    host[key].copy_(t, non_blocking=True)
            else:
                host.copy_(src_device, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        return cls(_tensors=host, _event=event)

    def wait(self) -> _TensorOrDict:
        tensors = self._tensors
        event = self._event
        if tensors is None or event is None:
            raise RuntimeError("FutureTensors.wait() was called more than once")

        event.synchronize()
        self._tensors = None
        self._event = None
        return tensors


@dataclass(slots=True, kw_only=True)
class DelayedDeviceHostHandler:
    """Stage device-side compute at step T, drain + postprocess host copy at step T+1.

    Each call to :meth:`step` first drains the previous step's staged future (running
    ``postprocess_on_host`` against the host snapshot) and then asks ``compute_on_device``
    for fresh device data to stage. ``compute_on_device`` may return ``None`` to skip
    staging (e.g. period gating)."""

    compute_on_device: Callable[[], Optional[_TensorOrDict]]
    postprocess_on_host: Callable[[_TensorOrDict], None]
    d2h_stream: torch.cuda.Stream
    _future: Optional[FutureTensors] = field(default=None)

    def step(self) -> None:
        if (pending := self._future) is not None:
            self.postprocess_on_host(pending.wait())

        with torch.cuda.stream(self.d2h_stream):
            device_data = self.compute_on_device()

        if device_data is None:
            self._future = None
        else:
            self._future = FutureTensors.device_to_host(
                device_data, stream=self.d2h_stream
            )
