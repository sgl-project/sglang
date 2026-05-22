from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch

_PayloadDict = dict[str, Any]
_TensorOrDict = Union[torch.Tensor, _PayloadDict]


@dataclass(slots=True, kw_only=True)
class FutureTensors:
    _tensors: Optional[_TensorOrDict]
    _event: Optional[torch.cuda.Event]

    @classmethod
    def device_to_host(
        cls, src_device: _TensorOrDict, *, stream: torch.cuda.Stream
    ) -> "FutureTensors":
        if isinstance(src_device, dict):
            host: _PayloadDict = {}
            ref_device: Optional[torch.device] = None
            for key, value in src_device.items():
                if isinstance(value, torch.Tensor):
                    host[key] = torch.empty(
                        value.shape, dtype=value.dtype, pin_memory=True
                    )
                    if ref_device is None:
                        ref_device = value.device
                else:
                    # Non-tensor payload (ints, dicts, etc.) is pass-through so callers
                    # can bundle host metadata (e.g. the step at which the snapshot was
                    # staged) alongside the device tensors and recover that context in
                    # postprocess without reaching back into the producer.
                    host[key] = value
            if ref_device is None:
                raise ValueError(
                    "FutureTensors.device_to_host requires at least one torch.Tensor in "
                    "the source dict to anchor the d2h stream sync"
                )
        else:
            host = torch.empty(
                src_device.shape, dtype=src_device.dtype, pin_memory=True
            )
            ref_device = src_device.device

        stream.wait_stream(torch.cuda.current_stream(ref_device))
        with torch.cuda.stream(stream):
            if isinstance(src_device, dict):
                for key, value in src_device.items():
                    if isinstance(value, torch.Tensor):
                        host[key].copy_(value, non_blocking=True)
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
    staging (e.g. period gating). Both callables are passed per-call so the caller can
    capture step-local state in a closure instead of stashing it on ``self``."""

    d2h_stream: torch.cuda.Stream
    _future: Optional[FutureTensors] = field(default=None)

    def step(
        self,
        *,
        compute_on_device: Callable[[], Optional[_TensorOrDict]],
        postprocess_on_host: Callable[[_TensorOrDict], None],
    ) -> None:
        if (pending := self._future) is not None:
            postprocess_on_host(pending.wait())

        with torch.cuda.stream(self.d2h_stream):
            device_data = compute_on_device()

        if device_data is None:
            self._future = None
        else:
            self._future = FutureTensors.device_to_host(
                device_data, stream=self.d2h_stream
            )
