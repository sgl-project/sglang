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
import threading
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
    "eager_on_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
]


def _check_cuda_bindings():
    if rt is None:
        raise ImportError(
            "Breakable CUDA graph requires the 'cuda-python' package. "
            "Install it with: pip install cuda-python"
        )


class GraphBreakInfo(NamedTuple):
    # python function breaking the graph
    func: Callable
    # output of the function (must be a tensor so we keep them)
    output: Any
    # raw handle after capture or raw exec handle after instantiate
    graph_handle: Any


_captured_graphs_var: ContextVar[list[GraphBreakInfo] | None] = ContextVar(
    "captured_graphs", default=None
)
_current_stream_var: ContextVar[torch.cuda.Stream | None] = ContextVar(
    "current_stream", default=None
)
_forked_streams_var: ContextVar[set[torch.cuda.Stream] | None] = ContextVar(
    "forked_streams", default=None
)


def get_current_stream(device: torch.device | None = None) -> torch.cuda.Stream:
    stream = _current_stream_var.get()
    if stream is None:
        return torch.cuda.current_stream(device)
    return stream


def _capture_status(stream_ptr: int) -> "rt.cudaStreamCaptureStatus":
    _check_cuda_bindings()
    status, *_ = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))
    return status


def _is_capturing(stream_ptr: int) -> bool:
    _check_cuda_bindings()
    return (
        _capture_status(stream_ptr)
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    )


# hook wait_stream to track forks/joins during breakable capture.
_original_wait_stream: Callable | None = None
_hook_lock = threading.Lock()
_hook_refcount = 0


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
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        if _hook_refcount == 0:
            _original_wait_stream = torch.cuda.Stream.wait_stream
            torch.cuda.Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]
        _hook_refcount += 1


def _uninstall_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        _hook_refcount -= 1
        if _hook_refcount == 0:
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


def _destroy_graph_exec(graph_exec_ptr: int) -> None:
    checkCudaErrors(rt.cudaGraphExecDestroy(graph_exec_ptr))


def _replay_graph(graph_exec_ptr: int, stream_ptr: int) -> None:
    checkCudaErrors(rt.cudaGraphLaunch(graph_exec_ptr, stream_ptr))


def _copy_output(dst: Any, src: Any) -> Any:
    """Copy src output into dst in-place where possible.

    Handles plain tensors, dataclass/object with tensor attributes,
    and dicts of tensors. Returns dst if in-place copy succeeded,
    otherwise returns src.
    """
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return dst

    # Handle objects with __dict__ (dataclasses, regular objects)
    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                setattr(dst, key, src_val)
        return dst

    # Handle dicts of tensors
    if isinstance(dst, dict) and isinstance(src, dict):
        for key, src_val in src.items():
            dst_val = dst.get(key)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                dst[key] = src_val
        return dst

    return src


def eager_on_graph(enable: bool):
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
            output = inner(*args, **kwargs)

            # Store the callable and its arguments so replay can re-invoke with
            # the same argument *references* (which point to CUDA graph input
            # buffers whose contents are updated before replay).
            captured_inner = inner
            captured_args = args
            captured_kwargs = kwargs
            captured_output = output

            def replay_fn():
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            captured_graphs = _captured_graphs_var.get()
            assert (
                captured_graphs is not None
            ), "eager_on_graph wrapper called outside of BreakableCUDAGraphCapture"
            captured_graphs.append(GraphBreakInfo(replay_fn, output, last_graph))
            _begin_capture_segment(stream)
            return output

        return wrapper

    return decorator


class BreakableCUDAGraph(torch.cuda.CUDAGraph):

    def __new__(cls) -> "BreakableCUDAGraph":
        return super().__new__(cls, True)

    def capture_begin(self, pool=None, capture_error_mode: str = "global") -> None:
        _check_cuda_bindings()
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
        self._exec = []
        if breaks:
            for replay_fn, output, handle in breaks:
                graph_exec = _instantiate_graph(handle)
                self._exec.append(GraphBreakInfo(replay_fn, output, graph_exec))

        # start a dummy capture so torch's capture_end() can finalize
        _begin_capture_segment(stream)
        super().capture_end()

    def replay(self):
        stream = torch.cuda.current_stream()
        token = _current_stream_var.set(stream)
        try:
            if not self._exec:
                _replay_graph(self.last_graph_exec, stream.cuda_stream)
                return
            for func, _, handle in self._exec:
                _replay_graph(handle, stream.cuda_stream)
                func()
            _replay_graph(self.last_graph_exec, stream.cuda_stream)
        finally:
            _current_stream_var.reset(token)

    def __del__(self):
        try:
            if hasattr(self, "_exec"):
                for _, _, handle in self._exec:
                    _destroy_graph_exec(handle)
            if hasattr(self, "last_graph_exec"):
                _destroy_graph_exec(self.last_graph_exec)
        except Exception:
            pass


class BreakableCUDAGraphCapture(torch.cuda.graph):
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
        self._breaks_token = _captured_graphs_var.set([])
        self._stream_token = _current_stream_var.set(self._stream)
        self._forked_streams_token = _forked_streams_var.set(set())
        return super().__enter__()

    def __exit__(self, *args: object):
        super().__exit__(*args)
        _current_stream_var.reset(self._stream_token)
        _captured_graphs_var.reset(self._breaks_token)
        _forked_streams_var.reset(self._forked_streams_token)
        _uninstall_wait_stream_hook()


@eager_on_graph(True)
def break_graph():
    """Insert a graph break. The @eager_on_graph decorator does the actual
    segment split; this function body intentionally does nothing."""
    pass
