from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch

_PayloadDict = dict[str, Any]
_TensorOrDict = Union[torch.Tensor, _PayloadDict]

_DUMMY_DICT_KEY = "__dummy_key__"


@dataclass(slots=True, kw_only=True)
class FutureTensors:
    _tensors: Optional[_PayloadDict]
    _event: Optional[torch.cuda.Event]

    @classmethod
    def device_to_host(
        cls, src_device: _TensorOrDict, *, stream: torch.cuda.Stream
    ) -> "FutureTensors":
        if not isinstance(src_device, dict):
            src_device = {_DUMMY_DICT_KEY: src_device}

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

        stream.wait_stream(torch.cuda.current_stream(ref_device))
        with torch.cuda.stream(stream):
            for key, value in src_device.items():
                if isinstance(value, torch.Tensor):
                    _clone_and_copy_to_host(x_device=value, x_host=host)
            event = torch.cuda.Event()
            event.record()

        return cls(_tensors=host, _event=event)

    def wait(self) -> _TensorOrDict:
        tensors = self._tensors
        event = self._event
        self._tensors = None
        self._event = None

        if tensors is None or event is None:
            raise RuntimeError("FutureTensors.wait() was called more than once")

        event.synchronize()

        if _DUMMY_DICT_KEY in tensors:
            tensors = tensors[_DUMMY_DICT_KEY]

        return tensors


def _clone_and_copy_to_host(x_device: torch.Tensor, x_host: torch.Tensor) -> torch.Tensor:
    x_device_cloned = x_device.detach().clone()
    x_host.copy_(x_device_cloned, non_blocking=True)


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
