"""Breakable primitives — segmented CUDA graph capture with eager break points.

Public API (also reachable via the deeper module paths):
  - ``BreakableCUDAGraph``, ``BreakableCUDAGraphCapture`` — capture/replay
  - ``eager_on_graph`` — decorator that marks a callable as a graph break
  - ``enable_breakable_cuda_graph`` — context that flips the Breakable runtime flag
  - ``is_in_breakable_cuda_graph`` — runtime flag getter

The legacy ``model_executor/breakable_cuda_graph/`` package is a
backwards-compat shim that re-exports from here.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.model_executor.cuda_graph_backend_utils.breakable_cuda_graph.context import (  # noqa: F401
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
