"""Low-level utilities used by the CUDA graph runners.

Mirror of ``cuda_graph_backend_utils/`` for runner-side state — buffer
dataclasses, process-global capture flags, the speculative-shared
graph memory pool, and the DeepEP capture/replay adapter. Runners in
``cuda_graph_runner/`` import from here; nothing here should import
back into ``cuda_graph_runner/``.
"""

from sglang.srt.model_executor.cuda_graph_runner_utils.buffers import (  # noqa: F401
    DecodeInputBuffers,
    PrefillInputBuffers,
    _grouped_foreach_copy_,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.capture_mode import (  # noqa: F401
    _set_capture_lora_variant,
    get_capture_lora_variant,
    get_is_capture_mode,
    model_capture_mode,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.deepep_adapter import (  # noqa: F401
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.model_executor.cuda_graph_runner_utils.pool import (  # noqa: F401
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
