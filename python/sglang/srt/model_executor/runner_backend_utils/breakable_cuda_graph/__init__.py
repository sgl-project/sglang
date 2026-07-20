"""Breakable CUDA graph runtime state.

The segmented-capture engine now lives in the PyTorch upstream
``piecewise_cuda_graphs`` package (CUDAGraphSequence / piecewise_graph /
no_graph / force_no_graph); this package only carries sglang's breakable
backend-routing flag:
  - enable_breakable_cuda_graph — context that flips the Breakable runtime flag
  - is_in_breakable_cuda_graph — runtime flag getter

``cuda_utils`` (a generic CUDA-error helper) also lives here and is imported
directly from its module.
"""

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (  # noqa: F401
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
