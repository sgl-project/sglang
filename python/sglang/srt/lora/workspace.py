from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import torch

if TYPE_CHECKING:
    from sglang.kernels.ops.lora.common.route_view import RouteView

_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


_T = TypeVar("_T")


class LoraWorkspace:
    """Reusable, stream-local scratch for LoRA runners.

    Warm up on the streams used during capture. Graph phases use separate
    buffers; outgrown graph storage stays alive for existing captures.
    """

    def __init__(self) -> None:
        self._graph_storage: dict[
            tuple[str, torch.dtype, torch.device, int, bool], torch.Tensor
        ] = {}
        self._retired: list[torch.Tensor] = []
        self._eager_buffers: dict[
            tuple[str, torch.dtype, torch.device, int], torch.Tensor
        ] = {}
        self._streams: dict[tuple[torch.device, int], torch.cuda.Stream] = {}
        self._events: dict[tuple[torch.device, int, str], torch.cuda.Event] = {}
        self._graph_mode = False
        self._is_prefill_graph = False
        self.routes: dict[tuple, dict[tuple, RouteView]] = {}

    def route(
        self, mapping: torch.Tensor, key: tuple, build: Callable[[], RouteView]
    ) -> RouteView:
        stream = self._caller(mapping.device) if mapping.is_cuda else 0
        routes = self.routes.setdefault((mapping.device, stream), {})
        key = (
            self._graph_mode,
            self._is_prefill_graph if self._graph_mode else False,
            mapping.dtype,
            mapping.data_ptr(),
            mapping.numel(),
            mapping.stride(),
            *key,
        )
        route = routes.get(key)
        if route is None:
            route = build()
            routes[key] = route
        return route

    def begin_forward(
        self, *, graph_mode: bool, is_prefill_graph: bool = False
    ) -> None:
        self._graph_mode = bool(graph_mode)
        self._is_prefill_graph = bool(is_prefill_graph)

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
        elements = 1
        for dimension in resolved_shape:
            elements *= dimension
        stream = self._caller(resolved_device) if resolved_device.type == "cuda" else 0
        key = (name, dtype, resolved_device, stream)
        if self._graph_mode:
            key = (*key, self._is_prefill_graph)
            storage = self._graph_storage.get(key)
            if storage is None or storage.numel() < elements:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "MoE LoRA workspace was not warmed before CUDA capture: "
                        f"missing {name!r} {resolved_shape} {dtype} on "
                        f"{resolved_device}"
                    )
                if storage is not None:
                    self._retired.append(storage)
                factory = torch.zeros if zero_on_first_allocation else torch.empty
                storage = factory((elements,), dtype=dtype, device=resolved_device)
                self._graph_storage[key] = storage
            tensor = storage[:elements].view(resolved_shape)
        else:
            storage = self._eager_buffers.get(key)
            if storage is None or storage.numel() < elements:
                if self._capturing(resolved_device):
                    raise RuntimeError(
                        "an eager MoE LoRA workspace cannot grow inside CUDA capture"
                    )
                factory = torch.zeros if zero_on_first_allocation else torch.empty
                storage = factory((elements,), dtype=dtype, device=resolved_device)
                self._eager_buffers[key] = storage
            tensor = storage[:elements].view(resolved_shape)
        return tensor

    @staticmethod
    def _caller(device: torch.device) -> int:
        # Read the raw handle without constructing a Python Stream wrapper.
        if _raw_stream is not None and device.type == "cuda":
            index = device.index
            return _raw_stream(
                index if index is not None else torch.cuda.current_device()
            )
        return torch.cuda.current_stream(device).cuda_stream

    def side_stream(self, device: torch.device | str) -> torch.cuda.Stream:
        resolved_device = torch.device(device)
        caller = self._caller(resolved_device)
        key = (resolved_device, caller)
        stream = self._streams.get(key)
        if stream is None:
            # The stream pool can return the caller's stream, preventing overlap.
            stream = torch.cuda.Stream(device=resolved_device)
            while stream.cuda_stream == caller:
                stream = torch.cuda.Stream(device=resolved_device)
            self._streams[key] = stream
        return stream

    def event(self, device: torch.device | str, name: str) -> torch.cuda.Event:
        resolved_device = torch.device(device)
        key = (resolved_device, self._caller(resolved_device), name)
        event = self._events.get(key)
        if event is None:
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
        caller_routes = self.routes.setdefault(
            (current.device, current.cuda_stream), {}
        )
        side_routes = self.routes.setdefault(
            (side_stream.device, side_stream.cuda_stream), {}
        )
        # Routes become visible across streams at the existing event waits.
        side_routes.update(caller_routes)
        with torch.cuda.stream(side_stream):
            side()
            done.record(side_stream)
        completed_routes = side_routes.copy()
        result = compute()
        current.wait_event(done)
        caller_routes.update(completed_routes)
        return result
