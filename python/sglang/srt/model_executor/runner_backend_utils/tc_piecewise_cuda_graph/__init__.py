"""Piecewise CUDA graph utilities — shared between Breakable and tc_piecewise backends.

Public API:
  - is_in_tc_piecewise_cuda_graph() — true while inside any piecewise capture.
  - enable_tc_piecewise_cuda_graph() — context manager that toggles the flag.
  - TcPiecewiseForwardContext + set_tc_piecewise_forward_context + get_tc_piecewise_forward_context.
  - TCPCG_FAILURE_HINT — backend-switch suggestion plugged into
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG by the prefill runner.

The torch.compile-warmup flag (is_in_torch_compile_warmup) lives in
sglang.srt.compilation.compile_phase — it is torch.compile-internal,
not piecewise-shared.
"""

from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (  # noqa: F401
    TCPCG_FAILURE_HINT,
    TcPiecewiseForwardContext,
    enable_tc_piecewise_cuda_graph,
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
