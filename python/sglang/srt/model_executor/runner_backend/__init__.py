"""Capture-mechanism backends for CUDA graphs.

A backend owns *how* a captured artifact is produced and replayed for
one shape; it is phase-agnostic. Runners (cuda_graph_runner/) own
*what* data flows in and out.

Public API:
  - BaseCudaGraphBackend — abstract interface.
  - FullCudaGraphBackend — single torch.cuda.CUDAGraph per shape.
  - BreakableCudaGraphBackend — segmented capture with eager break
    markers; no torch.compile.
  - TcPiecewiseCudaGraphBackend — torch.compile-driven piecewise
    capture; FX-splits the model at attention layers.
"""

from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (  # noqa: F401
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (  # noqa: F401
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (  # noqa: F401
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (  # noqa: F401
    TcPiecewiseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.utils import (  # noqa: F401
    resolve_decode_backend,
    resolve_prefill_backend,
)
