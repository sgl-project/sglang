"""Keep draft prefill capture out of target EPLB statistics on NPU.

Preserve the existing Full/Breakable implementations and their configuration;
this adapter only scopes recording during capture, not prefill graph support.
"""

from sglang.srt.hardware_backend.npu.eplb import isolate_npu_draft_recording
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.runtime_context import get_exec


class _DraftCaptureIsolation:
    @property
    def is_draft_worker(self):
        return self._cuda_graph_runner.model_runner.is_draft_worker

    @isolate_npu_draft_recording
    def capture_one(self, *args, **kwargs):
        return super().capture_one(*args, **kwargs)


class _NpuFullPrefillBackend(_DraftCaptureIsolation, FullCudaGraphBackend):
    pass


class _NpuBreakablePrefillBackend(_DraftCaptureIsolation, BreakableCudaGraphBackend):
    pass


def resolve_npu_prefill_backend(cuda_graph_runner, backend_name):
    enable_memory_saver = get_exec().features.enable_memory_saver
    if backend_name == Backend.BREAKABLE:
        return _NpuBreakablePrefillBackend(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            debug_eager=get_exec().graph.debug_cuda_graph,
        )
    if backend_name == Backend.FULL:
        return _NpuFullPrefillBackend(
            cuda_graph_runner,
            enable_memory_saver=enable_memory_saver,
            reuse_output_buffer=True,
        )
    raise ValueError(f"Unsupported prefill graph backend: {backend_name}")
