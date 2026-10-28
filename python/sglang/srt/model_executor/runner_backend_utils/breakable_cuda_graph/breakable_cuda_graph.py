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
torch.cuda.CUDAGraph segments separated by eager break points.

Each segment is a real torch.cuda.CUDAGraph. Its destructor calls
releasePool on the shared mempool, so the pool's use_count tracks how
many segments are alive; the pool stays pinned as long as any segment graph
is alive. This lets weak_ref_tensor views of intermediate pool-allocated
tensors remain valid across replays, so a break's output allocated in the
captured region needs no other owner.
"""

import functools
import threading
import warnings
from contextvars import ContextVar
from typing import Any, Callable, Optional

import torch

try:
    from cuda.bindings import runtime as rt
except ImportError:
    rt = None

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)
from sglang.srt.utils import get_device_module, is_hip, is_xpu

_is_xpu = is_xpu()

__all__ = [
    "eager_on_graph",
    "BreakableCUDAGraph",
    "BreakableCUDAGraphCapture",
    "break_graph",
]


def _check_cuda_bindings():
    if rt is None:
        raise ImportError(
            "Breakable CUDA graph on NVIDIA requires the 'cuda-python' package. "
            "Install it with: pip install cuda-python"
        )


# Active BreakableCUDAGraphCapture context for the currently-capturing thread.
# eager_on_graph's wrapper uses this to split the current torch.cuda.CUDAGraph
# at break points.
_current_capture_var: ContextVar["BreakableCUDAGraphCapture | None"] = ContextVar(
    "current_capture", default=None
)
_current_stream_var: ContextVar[torch.Stream | None] = ContextVar(
    "current_stream", default=None
)


_forked_streams_var: ContextVar[set[torch.Stream] | None] = ContextVar(
    "forked_streams", default=None
)


def get_current_stream(device: torch.device | None = None) -> torch.Stream:
    stream = _current_stream_var.get()
    if stream is None:
        return get_device_module().current_stream(device)
    return stream


def _capture_status(stream_ptr: int) -> "rt.cudaStreamCaptureStatus":
    _check_cuda_bindings()
    status, *_ = checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))
    return status


def _is_stream_capturing(stream: torch.Stream) -> bool:
    # On ROCm/HIP and XPU, cuda-python is unavailable, so use the portable torch
    # API (which maps to the HIP / XPU runtime). On NVIDIA, keep querying the
    # CUDA runtime directly via cuda-python: torch.cuda.is_current_stream_capturing()
    # has proven unreliable there, so we preserve the original behavior.
    if is_hip() or _is_xpu:
        with get_device_module().stream(stream):
            return get_device_module().is_current_stream_capturing()
    return (
        _capture_status(stream.cuda_stream)
        == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    )


# Hook torch.cuda.Stream.wait_stream to track side-stream forks/joins that happen
# during breakable capture. We need this because capture_end() on a torch
# CUDAGraph fails if there are still side streams participating in the capture
# — so before ending each segment we auto-join any forked-but-not-rejoined streams.
_original_wait_stream: Callable | None = None
_hook_lock = threading.Lock()
_hook_refcount = 0


def _hooked_wait_stream(self: torch.Stream, other: torch.Stream):
    assert _original_wait_stream is not None
    forked = _forked_streams_var.get()
    if forked is None:
        _original_wait_stream(self, other)
        return
    capturing = _current_stream_var.get()
    if capturing is None:
        _original_wait_stream(self, other)
        return

    cap_id = capturing.stream_id
    is_self_cap = self is capturing or self.stream_id == cap_id
    is_other_cap = other is capturing or other.stream_id == cap_id

    if is_self_cap and not is_other_cap:
        if not _is_stream_capturing(other):
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
            _original_wait_stream = get_device_module().Stream.wait_stream
            get_device_module().Stream.wait_stream = _hooked_wait_stream  # type: ignore[assignment]
        _hook_refcount += 1


def _uninstall_wait_stream_hook():
    global _original_wait_stream, _hook_refcount
    with _hook_lock:
        _hook_refcount -= 1
        if _hook_refcount == 0:
            assert _original_wait_stream is not None, "wait_stream hook not installed"
            get_device_module().Stream.wait_stream = _original_wait_stream  # type: ignore[assignment]
            _original_wait_stream = None


def _weak_ref_if_tensor(x):
    """Return a weak-ref view for nonempty accelerator tensors; recurse into
    tuples/lists and keep CPU, empty, and non-tensor values unchanged.
    Weak-ref'ing captured args lets the shared mempool reclaim per-layer
    intermediates between segments — storage stays alive for each segment
    CUDAGraph's lifetime via its pool use_count.

    weak_ref_tensors is imported lazily because it hard-raises on
    platforms without a CUDA/HIP/NPU backend; we only reach this code during
    an active Breakable capture, which runs only on those backends."""
    if torch.is_tensor(x):
        if x.numel() == 0 or x.device.type == "cpu":
            return x
        from sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor import (
            weak_ref_tensors,
        )

        return weak_ref_tensors(x)
    if isinstance(x, tuple):
        return tuple(_weak_ref_if_tensor(e) for e in x)
    if isinstance(x, list):
        return [_weak_ref_if_tensor(e) for e in x]
    return x


class _EagerOnGraphCallable:
    """Bind both graph-aware and explicit eager entry points to the same instance."""

    def __init__(self, graph_call: Callable, eager_call: Callable):
        self._graph_call = graph_call
        self._eager_call = eager_call

    def __call__(self, *args, **kwargs):
        return self._graph_call(*args, **kwargs)

    def run_eager(self, *args, **kwargs):
        return self._eager_call(*args, **kwargs)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        bound = _EagerOnGraphCallable(
            functools.partial(self._graph_call, instance),
            functools.partial(self._eager_call, instance),
        )
        return bound


def _copy_into_output(dst, src):
    """Validate the fixed output layout before writing graph-owned storage."""
    if dst is None and src is None:
        return
    if isinstance(dst, (tuple, list)):
        assert type(dst) is type(src) and len(dst) == len(src), (
            "BCG output structure changed"
        )
        for target, value in zip(dst, src):
            _copy_into_output(target, value)
        return
    assert (
        torch.is_tensor(dst)
        and torch.is_tensor(src)
        and dst.shape == src.shape
        and dst.dtype == src.dtype
        and dst.device == src.device
    ), "BCG output layout changed"
    if dst is not src:
        dst.copy_(src)


def eager_on_graph(
    fn: Optional[Callable] = None,
    on_capture: Optional[Callable] = None,
    *,
    enable: Optional[bool] = None,
    output_factory: Optional[Callable] = None,
    pass_output: bool = False,
):
    """Record an eager call between captured segments.

    The capture pass records the call without running it; ``on_capture``, if
    given, runs at that point for host setup a later captured segment reads.
    Tensor arguments keep their capture-time identity and are held weakly, so a
    buffer allocated outside the captured region needs an owner that outlives
    the graph. Request-specific state is read inside the body when it runs.

    Use ``fn.run_eager(...)`` to call the body inside an existing eager region.
    This explicit entry bypasses graph segmentation and ``on_capture``; methods
    bind their instance automatically.

    ``output_factory(*args, **kwargs)`` optionally allocates fixed-layout tensor
    outputs (or tuples/lists with None leaves) before ending the graph segment.
    Their storage can then be reused by the graph pool after their consumers.
    With ``pass_output=True``, the inner function also receives these buffers
    as ``out`` and can return them directly to avoid a copy. Outside capture,
    neither the factory nor the extra argument nor the output copies are used.
    """

    # Transitional support while model callers migrate to the bare decorator.
    if isinstance(fn, bool):
        enable, fn = fn, None
    if enable is False:
        return lambda inner: inner

    def decorator(inner: Callable):
        def run(*args, **kwargs) -> None:
            out = inner(*args, **kwargs)
            assert out is None, (
                f"{inner.__qualname__} is a graph break and must return None; "
                "write the results into buffers the caller allocated"
            )

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            capture = _current_capture_var.get()
            if capture is None:
                if output_factory is not None:
                    return inner(*args, **kwargs)
                run(*args, **kwargs)
                return

            # allocate while still capturing so output storage belongs to the pool
            output = (
                output_factory(*args, **kwargs) if output_factory is not None else None
            )
            if pass_output:
                kwargs = dict(kwargs, out=output)

            # End the segment that captured up to this break point.
            capture._end_current_segment()

            if on_capture is not None:
                on_capture(*args, **kwargs)

            # Weak-ref captured inputs produced by graph segments. Their storage
            # is pinned by the segment CUDAGraphs' mempool use-count, so Python
            # refs do not need to keep every intermediate alive.
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)
            captured_kwargs = {k: _weak_ref_if_tensor(v) for k, v in kwargs.items()}
            if output_factory is None:
                capture.cuda_graph._break_fns.append(
                    functools.partial(run, *captured_args, **captured_kwargs)
                )
            else:
                captured_output = _weak_ref_if_tensor(output)
                if pass_output:
                    captured_kwargs["out"] = captured_output

                def replay_fn():
                    new_out = inner(*captured_args, **captured_kwargs)
                    _copy_into_output(captured_output, new_out)

                capture.cuda_graph._break_fns.append(replay_fn)

            # Start a fresh CUDAGraph segment for the remainder of the forward.
            capture._begin_new_segment()
            return output

        return functools.update_wrapper(_EagerOnGraphCallable(wrapper, inner), inner)

    return decorator(fn) if fn is not None else decorator


class BreakableCUDAGraph:
    """Container holding one torch.cuda.CUDAGraph per segment plus an
    eager break function between consecutive segments."""

    def __init__(self, deduped_cuda_graph=None) -> None:
        self._segments: list[Any] = []
        self._break_fns: list[Callable[[], Any]] = []
        self._deduped_cuda_graph = deduped_cuda_graph

    def replay(self) -> None:
        stream = get_device_module().current_stream()
        token = _current_stream_var.set(stream)
        try:
            for i, seg in enumerate(self._segments):
                seg.replay()
                if i < len(self._break_fns):
                    self._break_fns[i]()
        finally:
            _current_stream_var.reset(token)

    def _append_segment(self, graph, needs_instantiate: bool) -> None:
        if self._deduped_cuda_graph is not None:
            self._segments.append(self._deduped_cuda_graph.register(graph))
            return
        if needs_instantiate:
            graph.instantiate()
        self._segments.append(graph)


class BreakableCUDAGraphCapture:
    """Context manager that captures the enclosed code as one or more
    torch.cuda.CUDAGraph segments separated by eager break points.

    Each segment shares the supplied pool (MempoolId_t tuple) so
    pool-allocated intermediates can be reused across segments. While any
    segment is alive, its beginAllocateToPool call keeps the mempool's
    use_count > 0, which makes weak_ref_tensor of segment-allocated
    tensors safe across subsequent replays.
    """

    def __init__(
        self,
        cuda_graph: BreakableCUDAGraph,
        pool=None,
        stream: torch.Stream | None = None,
        capture_error_mode: str = "global",
    ):
        assert isinstance(cuda_graph, BreakableCUDAGraph), (
            "cuda_graph must be a BreakableCUDAGraph"
        )
        self.cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._stream_ctx = None
        self._capture_token = None
        self._stream_token = None
        self._forked_token = None
        self._current_graph = None
        self._current_graph_needs_instantiate = False

    def __enter__(self):
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = get_device_module().stream(self._stream)
            self._stream_ctx.__enter__()
        self._capture_token = _current_capture_var.set(self)
        self._stream_token = _current_stream_var.set(
            self._stream or get_device_module().current_stream()
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
        graph_cls = torch.xpu.XPUGraph if _is_xpu else torch.cuda.CUDAGraph
        # keep_graph retains the raw graph for dedup; skip it on the plain path.
        # Dedup is CUDA-only (it introspects the raw graph via cuda-python), so
        # XPU always takes the plain path below.
        if self.cuda_graph._deduped_cuda_graph is not None:
            try:
                graph = graph_cls(keep_graph=True)
                self._current_graph_needs_instantiate = True
            except TypeError:
                graph = graph_cls()
                self._current_graph_needs_instantiate = False
        else:
            graph = graph_cls()
            self._current_graph_needs_instantiate = False
        if _is_xpu:
            # torch.xpu.XPUGraph.capture_begin takes only an optional pool.
            graph.capture_begin(pool=self._pool)
        else:
            graph.capture_begin(
                pool=self._pool, capture_error_mode=self._capture_error_mode
            )
        self._current_graph = graph

    def _end_current_segment(self) -> None:
        # Auto-join any side streams forked during this segment but not joined.
        main_stream = get_current_stream()
        forked = _forked_streams_var.get()
        if forked:
            assert _original_wait_stream is not None
            for side in list(forked):
                if _is_stream_capturing(side):
                    _original_wait_stream(main_stream, side)
            forked.clear()
        graph = self._current_graph
        assert graph is not None
        # A segment that enqueued no kernels (back-to-back breaks, or a segment
        # whose ops all ran eagerly) captures an empty graph, which replays as a
        # no-op. Torch warns about it on every such capture_end; expected here.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
            graph.capture_end()
        self.cuda_graph._append_segment(graph, self._current_graph_needs_instantiate)
        self._current_graph = None
        self._current_graph_needs_instantiate = False


@eager_on_graph
def break_graph() -> None:
    """Insert a graph break. The @eager_on_graph decorator does the actual
    segment split; this function body intentionally does nothing."""
