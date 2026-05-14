"""Phase-aware CUDA graph runners.

One concrete runner per phase. Each runner owns its phase-specific
shape semantics (decode → batch size; prefill → token count) and
delegates capture/replay mechanics to a pluggable
``BaseCudaGraphBackend`` chosen via ``cuda_graph_mode``.

Public API:
  - ``BaseCudaGraphRunner`` — abstract base; shared init + bucket
    padding + capture-loop scaffolding.
  - ``DecodeCudaGraphRunner`` — concrete decode-phase runner.
  - ``PrefillCudaGraphRunner`` — concrete prefill-phase runner.
  - Buffer dataclasses, capture-mode flags, the global memory pool,
    and the DeepEP adapter live in
    ``sglang.srt.model_executor.cuda_graph_runner_utils``; they are
    re-exported here for the EAGLE / multi-step draft cuda graph
    runners that were authored against the legacy public surface.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.tc_piecewise_cuda_graph import (  # noqa: F401
    TC_PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.cuda_graph_runner.base_runner import (  # noqa: F401
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.cuda_graph_runner.decode_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner.decode_runner import (  # noqa: F401
    _make_graph_key as _default_make_graph_key,
)
from sglang.srt.model_executor.cuda_graph_runner.prefill_runner import (  # noqa: F401
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner_utils import (  # noqa: F401
    DecodeInputBuffers,
    DeepEPCudaGraphRunnerAdapter,
    PrefillInputBuffers,
    _grouped_foreach_copy_,
    _set_capture_lora_variant,
    get_capture_lora_variant,
    get_global_graph_memory_pool,
    get_is_capture_mode,
    model_capture_mode,
    set_global_graph_memory_pool,
)
