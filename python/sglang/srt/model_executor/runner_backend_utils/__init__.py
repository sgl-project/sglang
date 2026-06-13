"""Low-level primitives used by the CUDA graph backends.

Subpackages:
  - breakable_cuda_graph: BreakableCUDAGraph + capture context,
    eager_on_graph decorator, is_in_breakable_cuda_graph flag.
  - piecewise_cuda_graph: shared piecewise context manager
    (set_tc_piecewise_forward_context, is_in_tc_piecewise_cuda_graph).

Backends in cuda_graph_backend/ import from here. Runners do not.
"""

# Generic failure-message hint for non-piecewise CUDA graph capture
# paths (Full backend used by decode + EAGLE draft runners). The
# piecewise-specific variant lives in
# piecewise_cuda_graph.context_manager and points users at
# --disable-piecewise-cuda-graph, which doesn't apply here.
CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs-decode to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --cuda-graph-backend-decode=disabled. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)
