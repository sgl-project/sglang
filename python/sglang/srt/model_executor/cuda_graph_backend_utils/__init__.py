"""Low-level primitives used by the CUDA graph backends.

Subpackages:
  - ``breakable_cuda_graph``: ``BreakableCUDAGraph`` + capture context,
    ``eager_on_graph`` decorator, ``is_in_breakable_cuda_graph`` flag.
  - ``piecewise_cuda_graph``: shared piecewise context manager
    (``set_forward_context``, ``is_in_tc_piecewise_cuda_graph``).

Backends in ``cuda_graph_backend/`` import from here. Runners do not.
"""

# Generic failure-message hint for non-piecewise CUDA graph capture
# paths (Full backend used by decode + EAGLE draft runners). The
# piecewise-specific variant lives in
# ``piecewise_cuda_graph.context_manager`` and points users at
# ``--disable-piecewise-cuda-graph``, which doesn't apply here.
CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "CUDA graph capture failed.\n"
    "To work around this error, add --disable-cuda-graph to your launch command\n"
    "(or use --decode-disable-cuda-graph to disable only the decode phase).\n"
    "Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose"
)
