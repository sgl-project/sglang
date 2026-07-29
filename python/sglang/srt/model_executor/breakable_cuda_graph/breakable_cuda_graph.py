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
"""Breakable CUDA Graph: capture a region as a sequence of
``torch.cuda.CUDAGraph`` segments separated by eager break points.

Each segment is a real ``torch.cuda.CUDAGraph``. Its destructor calls
``releasePool`` on the shared mempool, so the pool's ``use_count`` tracks how
many segments are alive; the pool stays pinned as long as any segment graph
is alive. This lets ``weak_ref_tensor`` views of intermediate pool-allocated
tensors remain valid across replays — we don't need Python-managed bridge
buffers to keep break-point tensors at stable addresses.
"""

import logging
import threading
from contextvars import ContextVar
from typing import Any, Callable

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


# Active BreakableCUDAGraphCapture context for the currently-capturing thread.
# eager_on_graph's wrapper uses this to split the current torch.cuda.CUDAGraph
# at break points.
_current_capture_var: ContextVar["BreakableCUDAGraphCapture | None"] = ContextVar(
    "current_capture", default=None
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


# Hook torch.cuda.Stream.wait_stream to track side-stream forks/joins that happen
# during breakable capture. We need this because capture_end() on a torch
# CUDAGraph fails if there are still side streams participating in the capture
# — so before ending each segment we auto-join any forked-but-not-rejoined streams.
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
        if (
            _capture_status(other.cuda_stream)
            != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
        ):
            return
        _original_wait_stream(self, other)
        forked.discard(other)
    elif is_other_cap and not is_self_cap:
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


def _weak_ref_if_tensor(x):
    """Return a weak-ref tensor view (shared storage, no refcount) for tensors;
    pass-through for non-tensors. Weak-ref'ing captured args lets the shared
    mempool reclaim per-layer intermediates between segments — storage stays
    alive for each segment CUDAGraph's lifetime via its pool use_count.

    ``weak_ref_tensors`` is imported lazily: the module hard-raises on
    non-CUDA/NPU platforms, and we only reach this code during an active
    BCG capture (which can't happen on CPU-only runners anyway)."""
    if torch.is_tensor(x):
        from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors

        return weak_ref_tensors(x)
    return x


def _copy_output(dst: Any, src: Any) -> Any:
    """Copy src output into dst in-place where possible.

    Handles plain tensors, dataclass/object with tensor attributes,
    and dicts of tensors. Returns dst if in-place copy succeeded,
    otherwise returns src.
    """
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)
        return dst

    if hasattr(dst, "__dict__") and hasattr(src, "__dict__"):
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val) and torch.is_tensor(src_val):
                dst_val.copy_(src_val)
            else:
                setattr(dst, key, src_val)
        return dst

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
            capture = _current_capture_var.get()
            if capture is None:
                return inner(*args, **kwargs)

            logger.debug("Break graph due to function: %s", inner.__name__)

            # End the segment that captured up to this break point.
            capture._end_current_segment()

            # Run the eager function once so it allocates its outputs and
            # writes real data into them.
            output = inner(*args, **kwargs)

            # Weak-ref the closure state. Storage lives with the segment
            # CUDAGraphs' mempool pin; Python refs don't need to prevent
            # pool reuse across layers.
            captured_inner = inner
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)
            captured_kwargs = {k: _weak_ref_if_tensor(v) for k, v in kwargs.items()}
            captured_output = _weak_ref_if_tensor(output)

            def replay_fn():
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            capture.cuda_graph._break_fns.append(replay_fn)

            # Start a fresh CUDAGraph segment for the remainder of the forward.
            capture._begin_new_segment()
            return output

        return wrapper

    return decorator


class BreakableCUDAGraph:
    """Container holding one ``torch.cuda.CUDAGraph`` per segment plus an
    eager break function between consecutive segments."""

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph] = []
        self._break_fns: list[Callable[[], Any]] = []

    def replay(self) -> None:
        stream = torch.cuda.current_stream()
        token = _current_stream_var.set(stream)
        try:
            for i, seg in enumerate(self._segments):
                seg.replay()
                if i < len(self._break_fns):
                    self._break_fns[i]()
        finally:
            _current_stream_var.reset(token)


class BreakableCUDAGraphCapture:
    """Context manager that captures the enclosed code as one or more
    ``torch.cuda.CUDAGraph`` segments separated by eager break points.

    Each segment shares the supplied ``pool`` (``MempoolId_t`` tuple) so
    pool-allocated intermediates can be reused across segments. While any
    segment is alive, its ``beginAllocateToPool`` call keeps the mempool's
    ``use_count`` > 0, which makes ``weak_ref_tensor`` of segment-allocated
    tensors safe across subsequent replays.
    """

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ):
        assert isinstance(
            cuda_graph, BreakableCUDAGraph
        ), "cuda_graph must be a BreakableCUDAGraph"
        self.cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._stream_ctx = None
        self._capture_token = None
        self._stream_token = None
        self._forked_token = None

    def __enter__(self):
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        self._capture_token = _current_capture_var.set(self)
        self._stream_token = _current_stream_var.set(
            self._stream or torch.cuda.current_stream()
        )
        self._forked_token = _forked_streams_var.set(set())
        self._begin_new_segment()
        return self

    def __exit__(self, *args: object):
        try:
            self._end_current_segment()
        finally:
            _forked_streams_var.reset(self._forked_token)
            _current_stream_var.reset(self._stream_token)
            _current_capture_var.reset(self._capture_token)
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)
                self._stream_ctx = None
            _uninstall_wait_stream_hook()
        return False

    def _begin_new_segment(self) -> None:
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(
            pool=self._pool, capture_error_mode=self._capture_error_mode
        )
        self.cuda_graph._segments.append(graph)

    def _end_current_segment(self) -> None:
        # Auto-join any side streams forked during this segment but not joined.
        main_stream = get_current_stream()
        forked = _forked_streams_var.get()
        if forked:
            assert _original_wait_stream is not None
            for side in list(forked):
                if _is_capturing(side.cuda_stream):
                    _original_wait_stream(main_stream, side)
            forked.clear()
        self.cuda_graph._segments[-1].capture_end()


@eager_on_graph(True)
def break_graph() -> None:
    """Insert a graph break. The @eager_on_graph decorator does the actual
    segment split; this function body intentionally does nothing."""
    pass
