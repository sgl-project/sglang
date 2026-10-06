from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch

from sglang.srt.utils import is_device_stream_capturing

_PayloadDict = dict[str, Any]
_TensorOrDict = Union[torch.Tensor, _PayloadDict]

_DUMMY_DICT_KEY = "__dummy_key__"


@dataclass(slots=True, kw_only=True)
class FutureTensors:
    _data: Optional[_PayloadDict]
    _event: Optional[torch.Event]
    # Device-source clones must outlive the async d2h copy.
    _retained_device_clones: Optional[dict[str, torch.Tensor]] = None

    @classmethod
    def device_to_host(
        cls, xs_device: _TensorOrDict, *, d2h_stream: torch.Stream
    ) -> FutureTensors:
        if not isinstance(xs_device, dict):
            xs_device = {_DUMMY_DICT_KEY: xs_device}

        first_tensor = next(
            (x for x in xs_device.values() if isinstance(x, torch.Tensor)), None
        )
        if first_tensor is None:
            raise ValueError(
                f"FutureTensors.device_to_host requires at least one tensor entry; "
                f"got dict with keys={list(xs_device)} containing no Tensor"
            )
        device = first_tensor.device
        del first_tensor

        assert not is_device_stream_capturing(device), (
            "FutureTensors.device_to_host must not be called during cuda-graph "
            "capture: the d2h side-stream copy + pinned-host alloc cannot be "
            "captured. Upper-layer callers are responsible for placing the d2h "
            "staging OUTSIDE the cuda graph (not inside it)."
        )

        tensors_device = {
            k: v for k, v in xs_device.items() if isinstance(v, torch.Tensor)
        }
        non_tensors_device = {
            k: v for k, v in xs_device.items() if not isinstance(v, torch.Tensor)
        }
        del xs_device

        # Must happen in current stream, not d2h stream
        tensors_device_cloned = {
            key: x.detach().clone() for key, x in tensors_device.items()
        }

        tensors_host = {
            key: torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
            for key, x in tensors_device.items()
        }

        device_module = torch.get_device_module(device)
        d2h_stream.wait_stream(device_module.current_stream(device))
        with device_module.stream(d2h_stream):
            for key in tensors_device_cloned:
                tensors_host[key].copy_(tensors_device_cloned[key], non_blocking=True)
            event = device_module.Event()
            event.record()

        return cls(
            _data=tensors_host | non_tensors_device,
            _event=event,
            _retained_device_clones=tensors_device_cloned,
        )

    def wait(self) -> _TensorOrDict:
        data = self._data
        event = self._event
        retained_device_clones = self._retained_device_clones
        self._data = None
        self._event = None
        self._retained_device_clones = None

        if data is None or event is None:
            raise RuntimeError("FutureTensors.wait() was called more than once")

        # Releasing clones AFTER event.synchronize() so the d2h copy
        # finishes reading from them before they become free-able.
        event.synchronize()
        del retained_device_clones

        if _DUMMY_DICT_KEY in data:
            data = data[_DUMMY_DICT_KEY]

        return data


@dataclass(slots=True, kw_only=True)
class DelayedDeviceHostHandler:
    """Stage device-side compute at step T, drain + postprocess host copy at step T+1."""

    d2h_stream: torch.Stream
    _future: Optional[FutureTensors] = field(default=None)

    def step(
        self,
        *,
        compute_on_device: Callable[[], Optional[_TensorOrDict]],
        postprocess_on_host: Callable[[_TensorOrDict], None],
    ) -> None:
        if (pending := self._future) is not None:
            postprocess_on_host(pending.wait())
            self._future = None

        # Must run on current stream, not d2h stream
        device_data = compute_on_device()

        if device_data is None:
            self._future = None
        else:
            self._future = FutureTensors.device_to_host(
                device_data, d2h_stream=self.d2h_stream
            )
