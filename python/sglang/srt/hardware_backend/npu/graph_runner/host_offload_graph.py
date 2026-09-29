"""NPU graph segments separated by explicit host-table lookups.

Like BreakableCUDAGraph, replay alternates device graphs and host work. The
segment boundary follows vllm-ascend's BreakableACLGraphWrapper, which runs
vLLM's breakable capture against NPUGraph (it aliases torch.cuda.CUDAGraph to
torch.npu.NPUGraph) and drives capture_begin/capture_end directly. Going
through torch.npu.graph instead would synchronize and empty_cache at every
boundary. graph_dispatch_mode is entered by hand because, unlike vllm-ascend,
the attention update of NPUCudaGraphBackend needs auto_dispatch_capture.

Only PLE lookups register breaks.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

_active_capture: ContextVar[NPUHostOffloadGraph | None] = ContextVar(
    "npu_host_offload_graph", default=None
)


def get_host_offload_graph() -> NPUHostOffloadGraph | None:
    return _active_capture.get()


class NPUHostOffloadGraph:
    """Keep graph addresses alive and rerun host lookups on every replay.

    Host callbacks retain their input/output tensors strongly. This deliberately
    pins the small ID and row buffers between segments; the host table is never
    copied in full. Capture records callbacks without reading uninitialized
    graph-produced IDs. Warmup and replay execute the real lookup.

    Breaks must be on the capture stream, with any model side streams already
    joined. PLE disables its prefetch side stream during model graph capture.
    """

    def __init__(self, device_module):
        self._device_module = device_module
        self._segments = []
        self._host_calls: list[Callable[[], object]] = []
        self._current_graph = None
        self._captured = False

    @contextmanager
    def capture(self, *, pool=None, stream=None):
        if get_host_offload_graph() is not None:
            raise RuntimeError("Nested NPU host-offload graph capture is unsupported")
        if self._segments or self._captured:
            raise RuntimeError("Use a new NPUHostOffloadGraph for each capture")
        self._pool = (
            pool if pool is not None else self._device_module.graph_pool_handle()
        )
        self._stream = stream or self._device_module.current_stream()
        # Once for the whole capture; torch.npu.graph would do both per segment.
        self._device_module.synchronize()
        self._device_module.empty_cache()
        with self._device_module.stream(self._stream):
            token = _active_capture.set(self)
            try:
                self._begin_segment()
                try:
                    yield self
                except BaseException as exc:
                    self._end_segment(type(exc), exc, exc.__traceback__)
                    raise
                else:
                    self._end_segment()
                    self._captured = True
            finally:
                _active_capture.reset(token)

    def _begin_segment(self):
        graph = self._device_module.NPUGraph()
        # Each NPUGraph owns its dispatch mode, so records stay per segment.
        graph.auto_dispatch_capture = True
        graph.graph_dispatch_mode.__enter__()
        graph.capture_begin(pool=self._pool)
        self._current_graph = graph

    def _end_segment(self, exc_type=None, exc=None, traceback=None):
        graph = self._current_graph
        if graph is None:
            return
        self._current_graph = None
        graph.capture_end()
        graph.graph_dispatch_mode.__exit__(exc_type, exc, traceback)
        self._segments.append(graph)

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    @property
    def num_host_lookups(self) -> int:
        return len(self._host_calls)

    def add_host_call(self, callback: Callable[[], object]) -> None:
        if get_host_offload_graph() is not self or self._current_graph is None:
            raise RuntimeError("Host lookups must be registered during capture")
        if self._device_module.current_stream() != self._stream:
            raise RuntimeError("NPU PLE graph breaks must run on the capture stream")
        self._end_segment()
        self._host_calls.append(callback)
        self._begin_segment()

    def replay(self):
        if not self._captured or get_host_offload_graph() is not None:
            raise RuntimeError("NPU host-offload graph is not ready for replay")
        for index, segment in enumerate(self._segments):
            segment.replay()
            if index < len(self._host_calls):
                # Blocking D2H in the callback waits for the producing segment.
                # Its H2D and the next segment use the same current stream.
                self._host_calls[index]()

    def update(self, cpu_update_input):
        if len(cpu_update_input) == 1:
            # NPUGraph broadcasts a single update across its dispatch records,
            # including the empty record list on segments without attention.
            for segment in self._segments:
                segment.update(cpu_update_input=cpu_update_input)
            return

        # Preserve the unsplit graph's ordered, per-op update convention (used
        # by multi-step draft runners); do not broadcast the entire list to
        # every segment or silently bind a later step to the first step's data.
        counts = [
            len(segment.graph_dispatch_mode.graph_dispatch_records)
            for segment in self._segments
        ]
        if sum(counts) != len(cpu_update_input):
            raise ValueError(
                "NPU graph update count does not match captured dispatch records: "
                f"{len(cpu_update_input)} != {sum(counts)}"
            )
        offset = 0
        for segment, count in zip(self._segments, counts):
            segment.update(cpu_update_input=cpu_update_input[offset : offset + count])
            offset += count
