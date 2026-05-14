"""Capture-mechanism backends for CUDA graphs.

A backend owns *how* a captured artifact is produced and replayed for
one shape; it is phase-agnostic. Runners (``cuda_graph_runner/``) own
*what* data flows in and out.

Public API:
  - ``BaseCudaGraphBackend`` — abstract interface.
  - ``FullCudaGraphBackend`` — single ``torch.cuda.CUDAGraph`` per shape.
  - ``BreakableCudaGraphBackend`` — segmented capture with eager break
    markers; no torch.compile.
  - ``TcPiecewiseCudaGraphBackend`` — torch.compile-driven piecewise
    capture; FX-splits the model at attention layers.
"""

from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (  # noqa: F401
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.breakable_cudagraph_backend import (  # noqa: F401
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.factory import (  # noqa: F401
    resolve_decode_backend,
    resolve_prefill_backend,
)
from sglang.srt.model_executor.cuda_graph_backend.full_cudagraph_backend import (  # noqa: F401
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.tc_piecewise_cudagraph_backend import (  # noqa: F401
    TcPiecewiseCudaGraphBackend,
)
