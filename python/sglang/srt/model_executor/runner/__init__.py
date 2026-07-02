"""Phase-aware CUDA graph runners.

Keep this package initializer light: many model modules import small runner
helpers during normal eager execution. CUDA-graph and DeepEP-specific classes
are loaded lazily so disabled backends do not pull optional kernels at import
time.
"""

from importlib import import_module

from sglang.srt.model_executor.runner.base_runner import BaseRunner  # noqa: F401
from sglang.srt.model_executor.runner.eager_runner import EagerRunner  # noqa: F401
from sglang.srt.model_executor.runner.shape_key import ShapeKey  # noqa: F401
from sglang.srt.model_executor.runner_utils import (  # noqa: F401
    DecodeInputBuffers,
    PrefillInputBuffers,
    _grouped_foreach_copy_,
    _set_capture_lora_variant,
    compile_in_capture_mode,
    get_capture_lora_variant,
    get_global_graph_memory_pool,
    get_is_capture_mode,
    model_capture_mode,
    set_global_graph_memory_pool,
)


def __getattr__(name: str):
    if name in {
        "BaseCudaGraphRunner",
        "freeze_gc",
        "get_batch_sizes_to_capture",
    }:
        base_cuda_graph_runner = import_module(
            "sglang.srt.model_executor.runner.base_cuda_graph_runner"
        )

        return getattr(base_cuda_graph_runner, name)
    if name == "DecodeCudaGraphRunner":
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        return DecodeCudaGraphRunner
    if name == "PrefillCudaGraphRunner":
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        return PrefillCudaGraphRunner
    if name == "DeepEPCudaGraphRunnerAdapter":
        from sglang.srt.model_executor.runner_utils import DeepEPCudaGraphRunnerAdapter

        return DeepEPCudaGraphRunnerAdapter
    if name == "TCPCG_FAILURE_HINT":
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
            TCPCG_FAILURE_HINT,
        )

        return TCPCG_FAILURE_HINT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
