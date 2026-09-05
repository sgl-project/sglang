from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import torch

_T = TypeVar("_T")


class MoeLoraWorkspace:

    def __init__(self) -> None:
        self._graph_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor
        ] = {}
        self._eager_buffers: dict[
            tuple[str, torch.dtype, torch.device], torch.Tensor
        ] = {}
        self._iota: dict[torch.device, torch.Tensor] = {}
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._events: dict[tuple[torch.device, str], torch.cuda.Event] = {}
        self._graph_mode = False

    def begin_forward(self, *, graph_mode: bool) -> None:
        self._graph_mode = bool(graph_mode)

    @staticmethod
    def _capturing(device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def tensor(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        zero_on_first_allocation: bool = False,
    ) -> torch.Tensor:
        # Per-forward clears belong to the caller so graphs replay them.
        resolved_device = torch.device(device)
        resolved_shape = tuple(int(dim) for dim in shape)
        if self._graph_mode:
            key = (name, resolved_shape, dtype, resolved_device)
            tensor = self._graph_buffers.get(key)
            if tensor is None:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "MoE LoRA workspace was not warmed before CUDA capture: "
                        f"missing {name!r} {resolved_shape} {dtype} on "
                        f"{resolved_device}"
                    )
                factory = torch.zeros if zero_on_first_allocation else torch.empty
                tensor = factory(resolved_shape, dtype=dtype, device=resolved_device)
                self._graph_buffers[key] = tensor
        else:
            key = (name, dtype, resolved_device)
            elements = 1
            for dimension in resolved_shape:
                elements *= dimension
            storage = self._eager_buffers.get(key)
            if storage is None or storage.numel() < elements:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "an eager MoE LoRA workspace cannot grow inside " "CUDA capture"
                    )
                factory = torch.zeros if zero_on_first_allocation else torch.empty
                storage = factory((elements,), dtype=dtype, device=resolved_device)
                self._eager_buffers[key] = storage
            tensor = storage[:elements].view(resolved_shape)
        return tensor

    def iota(self, n: int, device: torch.device | str) -> torch.Tensor:
        """Return an int32 identity map [0..n), filled outside capture.

        Graph mode keys the map by length so a captured pointer is never
        freed by later growth; eager mode grows one shared buffer.
        """
        resolved_device = torch.device(device)
        if self._graph_mode:
            key = ("iota", (n,), torch.int32, resolved_device)
            tensor = self._graph_buffers.get(key)
            if tensor is None:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "the MoE LoRA iota buffer was not warmed before CUDA capture"
                    )
                tensor = torch.arange(n, dtype=torch.int32, device=resolved_device)
                self._graph_buffers[key] = tensor
            return tensor
        buffer = self._iota.get(resolved_device)
        if buffer is None or buffer.numel() < n:
            if self._capturing(resolved_device):
                raise RuntimeError(
                    "the MoE LoRA iota buffer cannot grow inside CUDA capture"
                )
            capacity = max(n, 2 * buffer.numel() if buffer is not None else n)
            buffer = torch.arange(capacity, dtype=torch.int32, device=resolved_device)
            self._iota[resolved_device] = buffer
        return buffer[:n]

    def side_stream(self, device: torch.device | str) -> torch.cuda.Stream:
        # Calls on one device must fork from the same stream.
        resolved_device = torch.device(device)
        stream = self._streams.get(resolved_device)
        if stream is None:
            if self._capturing(resolved_device):
                raise RuntimeError(
                    "MoE LoRA side stream was not created before CUDA capture"
                )
            stream = torch.cuda.Stream(device=resolved_device)
            self._streams[resolved_device] = stream
        return stream

    def event(self, device: torch.device | str, name: str) -> torch.cuda.Event:
        resolved_device = torch.device(device)
        key = (resolved_device, name)
        event = self._events.get(key)
        if event is None:
            if self._capturing(resolved_device):
                raise RuntimeError(
                    f"MoE LoRA event was not created before CUDA capture: {name}"
                )
            event = torch.cuda.Event()
            self._events[key] = event
        return event

    def run_parallel(
        self,
        *,
        name: str,
        device: torch.device,
        compute: Callable[[], _T],
        side: Callable[[], object],
    ) -> _T:
        if device.type != "cuda":
            side()
            return compute()

        current = torch.cuda.current_stream(device)
        side_stream = self.side_stream(device)
        ready = self.event(device, f"{name}:ready")
        done = self.event(device, f"{name}:done")

        ready.record(current)
        side_stream.wait_event(ready)
        with torch.cuda.stream(side_stream):
            side()
            done.record(side_stream)
        result = compute()
        current.wait_event(done)
        return result
