"""MLU graph primitives with the shared runner/capture lifecycle."""

import logging

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)

logger = logging.getLogger(__name__)


class DecodeMluGraphRunner(DecodeCudaGraphRunner):
    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
            record_shapes=True,
        )
        torch.mlu.memory._record_memory_history()
        return context

    def _post_process_after_profile(self, context):
        from sglang.srt.utils.profile_utils import export_cuda_graph_capture_trace

        torch.mlu.memory._dump_snapshot("mlu_graph_runner_memory_usage.pickle")
        torch.mlu.memory._record_memory_history(enabled=None)
        logger.info(
            context.key_averages(group_by_input_shape=True).table(
                sort_by="mlu_time_total", row_limit=10
            )
        )
        export_cuda_graph_capture_trace(
            context, runner_name=self._profile_runner_name()
        )


class FullMluGraphBackend(FullCudaGraphBackend):
    def __init__(self, runner, *, enable_memory_saver=False, reuse_output_buffer=False):
        if enable_memory_saver:
            raise ValueError("MLU graph execution does not support memory saver.")
        super().__init__(runner, reuse_output_buffer=reuse_output_buffer)

    def _capture_graph(self, graph_ctx, graph):
        return graph_ctx(mlu_graph=graph, pool=self._pool, stream=self._capture_stream)

    def _create_graph(self):
        return torch.mlu.MLUGraph()


class BreakableMluGraphBackend(BreakableCudaGraphBackend):
    def __init__(self, runner, *, enable_memory_saver=False, debug_eager=False):
        if enable_memory_saver:
            raise ValueError("MLU graph execution does not support memory saver.")
        super().__init__(runner, debug_eager=debug_eager)

    def build_deduped_cuda_graph(self):
        from sglang.srt.environ import envs

        if envs.SGLANG_ENABLE_CUDA_GRAPH_DEDUP.get():
            raise ValueError("MLU graph deduplication is not supported.")
