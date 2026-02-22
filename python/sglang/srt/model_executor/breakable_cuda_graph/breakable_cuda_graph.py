# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
from contextvars import ContextVar
from typing import Any, Callable, NamedTuple

import torch

try:
    from cuda.bindings import runtime as rt
except ImportError:
    rt = None

from sglang.srt.model_executor.breakable_cuda_graph.cuda_utils import checkCudaErrors

logger = logging.getLogger(__name__)

__all__ = [
    "non_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphContext",
    "break_graph",
]


class GraphBreakInfo(NamedTuple):
    # python function breaking the graph
    func: Callable
    # output of the function (must be a tensor so we keep them)
    output: Any
    # raw handle after capture or raw exec handle after instantiate
    graph_handle: Any


_captured_graphs_var = ContextVar[list[GraphBreakInfo]]("captured_graphs", default=[])
_current_stream_var = ContextVar[torch.cuda.Stream | None](
    "current_stream", default=None
)
_forked_streams_var = ContextVar[set[torch.cuda.Stream] | None](
    "forked_streams", default=None
)


def get_current_stream(device: torch.device | None = None) -> torch.cuda.Stream:
    stream = _current_stream_var.get()
    if stream is None:
        return torch.cuda.current_stream(device)
    return stream


def _capture_status(stream_ptr: int) -> "rt.cudaStreamCaptureStatus":
    status, *_ = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))
    return status


def _is_capturing(stream_ptr: int) -> bool:
    return (
        _capture_status(stream_ptr)
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    )


# hook wait_stream to track forks/joins during breakable capture.
_original_wait_stream: Callable | None = None


def _hooked_wait_stream(self: torch.cuda.Stream, other: torch.cuda.Stream):
    assert _original_wait_stream is not None
    forked = _forked_streams_var.get()
    if forked is None:
        _original_wait_stream(self, other)
        return
    capturing = _current_stream_var.get()
    if capturing is None:
        _original_wait_stream(self, other)
        return

    cap_ptr = capturing.cuda_stream
    is_self_cap = self is capturing or self.cuda_stream == cap_ptr
    is_other_cap = other is capturing or other.cuda_stream == cap_ptr

    if is_self_cap and not is_other_cap:
        # Join: capturing_stream.wait_stream(other).
        # other might not be part of the capture because we join it in the last segment
        # skip the wait to avoid cuda error
        if (
            _capture_status(other.cuda_stream)
            != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
        ):
            return
        _original_wait_stream(self, other)
        forked.discard(other)
    elif is_other_cap and not is_self_cap:
        # Fork: other.wait_stream(capturing_stream).
        _original_wait_stream(self, other)
        forked.add(self)
    else:
        _original_wait_stream(self, other)


def _install_wait_stream_hook():
    global _original_wait_stream
    assert _original_wait_stream is None, "wait_stream hook already installed"
    _original_wait_stream = torch.cuda.Stream.wait_stream
    torch.cuda.Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]


def _uninstall_wait_stream_hook():
    global _original_wait_stream
    assert _original_wait_stream is not None, "wait_stream hook not installed"
    torch.cuda.Stream.wait_stream = _original_wait_stream  # type: ignore[assignment]
    _original_wait_stream = None


def _end_capture_segment(stream: torch.cuda.Stream):
    """End a capture segment, auto-joining any forked streams first."""
    # Join forked streams that are still part of this capture.
    forked = _forked_streams_var.get()
    if forked:
        assert _original_wait_stream is not None
        for s in forked:
            if _is_capturing(s.cuda_stream):
                _original_wait_stream(stream, s)
        forked.clear()

    graph = checkCudaErrors(rt.cudaStreamEndCapture(stream.cuda_stream))
    assert graph is not None
    return graph


def _begin_capture_segment(stream: torch.cuda.Stream):
    checkCudaErrors(
        rt.cudaStreamBeginCapture(
            stream.cuda_stream,
            rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal,
        )
    )


def _instantiate_graph(graph_ptr: int) -> int:
    graph_exec = checkCudaErrors(
        rt.cudaGraphInstantiateWithFlags(
            graph_ptr,
            rt.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagAutoFreeOnLaunch,
        )
    )
    assert graph_exec is not None
    checkCudaErrors(rt.cudaGraphDestroy(graph_ptr))
    return graph_exec


def _replay_graph(graph_exec_ptr: int, stream_ptr: int) -> None:
    checkCudaErrors(rt.cudaGraphLaunch(graph_exec_ptr, stream_ptr))


def non_graph(enable: bool):
    def decorator(inner: Callable):
        if not enable:
            return inner

        def wrapper(*args, **kwargs):
            stream = get_current_stream()
            if not _is_capturing(stream.cuda_stream):
                return inner(*args, **kwargs)
            last_graph = _end_capture_segment(stream)
            logger.debug(f"Break graph due to function: {inner.__name__}")
            # run the function once to allocate the output tensor captured by later graphs
            output = inner(*args, **kwargs)  # TODO: use weak ref tensor to save memory

            def f():
                new_out = inner(*args, **kwargs)
                if torch.is_tensor(output) and torch.is_tensor(new_out):
                    output.copy_(new_out)
                    return output
                return new_out

            _captured_graphs_var.get().append(GraphBreakInfo(f, output, last_graph))
            _begin_capture_segment(stream)
            return output

        return wrapper

    return decorator


class BreakableCUDAGraph(torch.cuda.CUDAGraph):

    def __new__(cls, keep_graph: bool = True) -> "BreakableCUDAGraph":
        keep_graph = True  # force keep_graph to True
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool=None, capture_error_mode: str = "global") -> None:
        super().capture_begin(pool, capture_error_mode)
        stream = get_current_stream()
        # torch graph will not record any operation but only for compatibility
        _end_capture_segment(stream)
        _begin_capture_segment(stream)

    def capture_end(self):
        stream = get_current_stream()
        self.last_graph = _end_capture_segment(stream)
        self.last_graph_exec = _instantiate_graph(self.last_graph)
        breaks = _captured_graphs_var.get()
        # TODO: need to free graph exec
        self._exec = []
        for f, output, handle in breaks:
            graph_exec = _instantiate_graph(handle)
            self._exec.append(GraphBreakInfo(f, output, graph_exec))

        # start a dummy capture so torch's capture_end() can finalize
        _begin_capture_segment(stream)
        super().capture_end()

    def replay(self):
        stream = torch.cuda.current_stream()
        token = _current_stream_var.set(stream)
        try:
            if not hasattr(self, "_exec"):
                _replay_graph(self.last_graph, stream.cuda_stream)
                return
            for func, _, handle in self._exec:
                _replay_graph(handle, stream.cuda_stream)
                func()
            _replay_graph(self.last_graph_exec, stream.cuda_stream)
        finally:
            _current_stream_var.reset(token)


class BreakableCUDAGraphContext(torch.cuda.graph):
    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ):
        super().__init__(
            cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode
        )
        self._stream = stream
        assert isinstance(
            cuda_graph, BreakableCUDAGraph
        ), "cuda_graph must be a BreakableCUDAGraph"

    def __enter__(self):
        _install_wait_stream_hook()
        self.breaks = _captured_graphs_var.set([])
        self._stream_token = _current_stream_var.set(self._stream)
        self._forked_streams_token = _forked_streams_var.set(set())
        return super().__enter__()

    def __exit__(self, *args: object):
        super().__exit__(*args)
        _current_stream_var.reset(self._stream_token)
        _captured_graphs_var.reset(self.breaks)
        _forked_streams_var.reset(self._forked_streams_token)
        _uninstall_wait_stream_hook()


@non_graph(True)
def break_graph():
    """Helper function to break the cuda graph"""
    pass
