# SPDX-License-Identifier: Apache-2.0
"""Record CUDA graph replays as child-graph nodes of an enclosing capture.

While a ``ChildGraphCapture`` is active on the current thread, a graph backend
hands each graph it would replay to :meth:`ChildGraphCapture.add_graph`. In
splice mode the graph (captured with ``keep_graph=True``) is added as a
child-graph node at the capturing stream's current dependencies and becomes
the stream's only dependency, so work captured afterwards is ordered after it
(the ``cudaStreamGetCaptureInfo`` / ``cudaStreamUpdateCaptureDependencies``
pattern). CUDA clones the child when it is added, so the original graph stays
valid for regular replays. In count mode the graph is launched as usual and
recorded, together with how far it advanced the CUDA generator, which lets a
caller check what a region replays before capturing it.

The context also carries named capture overrides: a function decorated with
:func:`capture_override` calls the handler that the active capture registered
under that name, and runs unchanged otherwise.
"""

from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch

try:
    from cuda.bindings import runtime as rt
except ImportError:
    rt = None

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)
from sglang.srt.model_executor.runner_utils.pool import graph_pool_replay_scope

__all__ = [
    "ChildGraphCapture",
    "capture_override",
    "current_child_graph_capture",
    "splice_graph_into_capture",
]

_F = TypeVar("_F", bound=Callable[..., Any])

_current_capture_var: ContextVar[Optional[ChildGraphCapture]] = ContextVar(
    "child_graph_capture", default=None
)


def current_child_graph_capture() -> Optional[ChildGraphCapture]:
    return _current_capture_var.get()


def splice_graph_into_capture(graph: torch.cuda.CUDAGraph, stream: Any) -> None:
    """Add ``graph`` (captured with ``keep_graph=True``) as a child-graph node
    of the capture that is active on ``stream``."""
    if rt is None:
        raise RuntimeError(
            "Splicing CUDA graphs requires the 'cuda-python' package. "
            "Install it with: pip install cuda-python"
        )
    stream_ptr = stream.cuda_stream
    status, _, capturing_graph, deps, _, num_deps = checkCudaErrors(
        rt.cudaStreamGetCaptureInfo(stream_ptr)
    )
    if status != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        raise RuntimeError("The current stream is not capturing a CUDA graph")
    # Raises unless the graph was created with keep_graph=True.
    child = rt.cudaGraph_t(init_value=graph.raw_cuda_graph())
    deps = list(deps)[:num_deps]
    node = checkCudaErrors(
        rt.cudaGraphAddChildGraphNode(capturing_graph, deps, len(deps), child)
    )
    checkCudaErrors(
        rt.cudaStreamUpdateCaptureDependencies(
            stream_ptr,
            [node],
            None,
            1,
            rt.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
        )
    )


class ChildGraphCapture:
    """Active-capture context consulted by graph backends and by functions
    decorated with :func:`capture_override`.

    Args:
        splice: True to splice graphs into the capture on the current stream;
            False to launch them as usual and record them.
        overrides: handlers by override name. A handler is called as
            ``handler(original_fn, *args, **kwargs)``.
    """

    def __init__(
        self,
        *,
        splice: bool,
        overrides: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        self.splice = splice
        self.overrides: Dict[str, Callable[..., Any]] = dict(overrides or {})
        # (owner, owner.graph_epoch) per graph handed over, in call order.
        self.graph_sources: List[Tuple[Any, Optional[int]]] = []
        # Count mode: CUDA generator offset advanced by each launch.
        self.generator_offset_deltas: List[int] = []
        self._token = None

    @property
    def num_graphs(self) -> int:
        return len(self.graph_sources)

    def add_graph(self, graph: torch.cuda.CUDAGraph, owner: Any = None) -> None:
        """Take over one replay of ``graph``: splice it into the capture, or
        launch it and record it. ``owner`` (e.g. the graph backend) may carry a
        ``graph_epoch`` that is recorded with it."""
        self.graph_sources.append((owner, getattr(owner, "graph_epoch", None)))
        if self.splice:
            splice_graph_into_capture(graph, torch.cuda.current_stream())
            return
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
        offset = generator.get_offset()
        with graph_pool_replay_scope():
            graph.replay()
        self.generator_offset_deltas.append(generator.get_offset() - offset)

    def __enter__(self) -> ChildGraphCapture:
        if _current_capture_var.get() is not None:
            raise RuntimeError("ChildGraphCapture contexts cannot be nested")
        self._token = _current_capture_var.set(self)
        return self

    def __exit__(self, *args: object) -> bool:
        _current_capture_var.reset(self._token)
        self._token = None
        return False


def capture_override(name: str) -> Callable[[_F], _F]:
    """Let an active :class:`ChildGraphCapture` replace the decorated function.

    Outside such a capture, or when it registered no handler under ``name``,
    the function runs unchanged.
    """

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            capture = _current_capture_var.get()
            if capture is not None:
                handler = capture.overrides.get(name)
                if handler is not None:
                    return handler(fn, *args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
