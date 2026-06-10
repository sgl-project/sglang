"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class BreakableCudaGraphBackend(BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        if is_hip():
            raise RuntimeError("Breakable CUDA graph is not supported on ROCm/HIP")
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            with self.replay_session():
                yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = BreakableCUDAGraph()
        captured_fn = (
            eager_on_graph(True)(forward_fn) if self._debug_eager else forward_fn
        )
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
        ):
            out = captured_fn()
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: Any) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
