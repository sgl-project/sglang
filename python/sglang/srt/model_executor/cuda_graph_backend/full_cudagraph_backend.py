"""FullCudaGraphBackend — captures the entire model forward as one
``torch.cuda.CUDAGraph`` per shape.

Backend owns its own ``_graphs[shape] -> CUDAGraph`` and
``_outputs[shape] -> LogitsProcessorOutput`` tables, plus the memory
pool handle for this backend instance.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """Single-graph capture: one ``torch.cuda.CUDAGraph`` per shape;
    attention metadata is captured *inside* the graph.

    Memory-saver-aware: when the ``TorchMemorySaverAdapter`` is enabled,
    capture goes through its wrapper so the graph allocation is tagged
    correctly.
    """

    def __init__(self, *, enable_memory_saver: bool = False) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = None
        self._tp_group = None
        self._memory_saver_adapter: Optional[Any] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._enable_memory_saver = enable_memory_saver

    def prepare(self, runner) -> None:
        self._device_module = runner.device_module
        self._tp_group = runner.model_runner.tp_group
        self._memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self._enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    def can_run(self, forward_batch: "ForwardBatch") -> bool:
        return True

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        """Bind ``stream`` and the (lazily-allocated) pool handle for the
        duration of the outer capture loop. Sets ``set_graph_pool_id``
        for symmetric-memory cooperation.
        """
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
    ) -> None:
        # Two jit warmups so kernels stay loaded and any one-time setup
        # cost is paid before the actual capture; then capture under
        # the cuda-graph context.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def has_shape(self, shape_key: Any) -> bool:
        return shape_key in self._graphs

    def replay(
        self,
        shape_key: Any,
        static_forward_batch: "ForwardBatch",
        **kwargs,
    ) -> Any:
        # static_forward_batch / kwargs are unused — Full backend replays
        # against the static buffers populated by the runner; output flows
        # via the cached ``_outputs[shape]`` whose tensors point into
        # those buffers.
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
