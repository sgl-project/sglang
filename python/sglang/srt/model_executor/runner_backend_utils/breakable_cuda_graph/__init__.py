"""Breakable primitives — segmented CUDA graph capture with eager break points.

Public API (also reachable via the deeper module paths):
  - BreakableCUDAGraph, BreakableCUDAGraphCapture — capture/replay
  - eager_on_graph — decorator that marks a callable as a graph break
  - break_graph — helper that inserts a bare graph break
  - enable_breakable_cuda_graph — context that flips the Breakable runtime flag
  - is_in_breakable_cuda_graph — runtime flag getter
  - BaseBreakableCudaGraphRunner — capture/replay eager-runner base class

"""

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
    get_current_replay_token,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (  # noqa: F401
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.runner import (  # noqa: F401
    BaseBreakableCudaGraphRunner,
)
