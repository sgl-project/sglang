"""Live HiCache dependencies for KDA state reads inside breakable CUDA graphs.

A captured pool accessor cannot issue the next batch's Python layer wait. Capture
external EventWait nodes instead, then bind each executable to the current load
slot before replay. The raw graphs and ready events must outlive the executables.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
    rt,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

_capture: ContextVar[HiCacheGraphEvents | None] = ContextVar(
    "hicache_graph_events", default=None
)


@dataclass
class LayerEventWait:
    node: Any
    layer: int
    event_handle: int


def capture_layer_wait(layer: int) -> None:
    """Insert the dependency before any captured Conv/SSM state access."""
    owner = _capture.get()
    if owner is None or not torch.cuda.is_current_stream_capturing():
        return
    owner.ready_events[layer].wait()


def wait_for_layer_loads(pool: HybridReqToTokenPool) -> None:
    """Conservative fallback when graph nodes cannot use live CUDA events."""
    counter = pool.layer_transfer_counter
    if counter is None or counter.consumer_index < 0:
        return
    # CPU futures may resolve independently and run a callback at each wait.
    for layer in range(counter.num_layers):
        counter.wait_until(layer)


class HiCacheGraphEvents:
    def __init__(self, pool: HybridReqToTokenPool) -> None:
        self.pool = pool
        self.ready_events = {
            layer: torch.cuda.Event(external=True) for layer in sorted(pool.mamba_map)
        }
        for event in self.ready_events.values():
            event.record()
        # Startup only. These completed events also represent a no-load replay.
        torch.cuda.synchronize()
        self._layer_by_event = {
            int(event.cuda_event): layer for layer, event in self.ready_events.items()
        }

    @contextmanager
    def capture_scope(self):
        token = _capture.set(self)
        try:
            yield
        finally:
            _capture.reset(token)

    def find_wait_nodes(self, graph) -> list[LayerEventWait]:
        raw = graph.raw_cuda_graph()
        _, count = checkCudaErrors(rt.cudaGraphGetNodes(raw, 0))
        if count == 0:
            return []
        nodes, _ = checkCudaErrors(rt.cudaGraphGetNodes(raw, count))
        waits = []
        for node in nodes:
            node_type = checkCudaErrors(rt.cudaGraphNodeGetType(node))
            if node_type != rt.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent:
                continue
            event = checkCudaErrors(rt.cudaGraphEventWaitNodeGetEvent(node))
            handle = int(event)
            layer = self._layer_by_event.get(handle)
            if layer is not None:
                waits.append(LayerEventWait(node, layer, handle))
        return waits

    def bind(self, segments, bindings: list[list[LayerEventWait]]) -> None:
        if not any(bindings):
            return
        events = self.ready_events
        counter = self.pool.layer_transfer_counter
        if counter is not None and counter.consumer_index >= 0:
            if hasattr(counter, "events"):
                load_events = counter.events[counter.consumer_index].load_events
                events = {
                    layer: load_events[layer - self.pool.start_layer]
                    for layer in self.ready_events
                }
            else:
                # CPU futures can complete independently and run layer callbacks.
                wait_for_layer_loads(self.pool)
        for graph, waits in zip(segments, bindings, strict=True):
            for wait in waits:
                handle = int(events[wait.layer].cuda_event)
                if handle != wait.event_handle:
                    checkCudaErrors(
                        rt.cudaGraphExecEventWaitNodeSetEvent(
                            graph.raw_cuda_graph_exec(), wait.node, handle
                        )
                    )
                    wait.event_handle = handle
